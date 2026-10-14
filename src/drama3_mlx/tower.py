from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from drama3_mlx.config import AudioDecoderConfig, Drama3Config
from drama3_mlx.layers import ZeroCenteredRMSNorm, grouped_attention

RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5


def sample_top_k_top_p(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.95,
) -> mx.array:
    """Sample one token index per row with top-k and top-p (nucleus) filtering.
    logits: [batch, vocab_size] -> returns [batch]
    """
    temp = max(temperature, MIN_TEMPERATURE)
    scaled_logits = logits.astype(mx.float32) / temp

    if top_k <= 1:
        return mx.argmax(scaled_logits, axis=-1)

    # Sort descending
    vocab_size = scaled_logits.shape[-1]
    top_k = min(top_k, vocab_size)

    indices = mx.argsort(-scaled_logits, axis=-1)
    sorted_logits = mx.take_along_axis(scaled_logits, indices, axis=-1)

    probs = mx.softmax(sorted_logits, axis=-1)
    cum_probs = mx.cumsum(probs, axis=-1)

    # Top-p thresholding: mask items whose cumulative probability exceeded top_p
    # Keep at least the first item
    mask = (cum_probs > top_p)
    mask = mx.concatenate([mx.zeros((mask.shape[0], 1), dtype=mx.bool_), mask[:, :-1]], axis=-1)

    if top_k < vocab_size:
        rank_mask = mx.arange(vocab_size)[None, :] >= top_k
        mask = mask | rank_mask

    filtered_logits = mx.where(mask, -1e9, sorted_logits)
    filtered_probs = mx.softmax(filtered_logits, axis=-1)

    sampled_sorted_idx = mx.random.categorical(mx.log(filtered_probs + 1e-10))
    sampled_orig_idx = mx.take_along_axis(indices, sampled_sorted_idx[:, None], axis=-1).squeeze(-1)
    return sampled_orig_idx


class TowerAttention(nn.Module):
    """Gated attention over time with interleaved RoPE and QK norm."""

    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.config = config
        self.q_width = config.n_head * config.head_dim
        self.kv_width = config.n_local_heads * config.head_dim

        self.wqkv = nn.Linear(config.dim, 2 * self.q_width + 2 * self.kv_width, bias=False)
        self.wo = nn.Linear(self.q_width, config.dim, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)

        self.k_cache: Optional[mx.array] = None
        self.v_cache: Optional[mx.array] = None

    def setup_cache(self, max_seq_len: int, dtype: mx.Dtype = mx.float16) -> None:
        shape = (1, self.config.n_local_heads, max_seq_len, self.config.head_dim)
        self.k_cache = mx.zeros(shape, dtype=dtype)
        self.v_cache = mx.zeros(shape, dtype=dtype)

    def rotate(self, x: mx.array, positions: mx.array) -> mx.array:
        """Interleaved (non-neox) partial RoPE in fp32."""
        width = self.config.rotary_dim
        exponents = mx.arange(0, width, 2, dtype=mx.float32)
        angles = positions.astype(mx.float32)[:, None] * (self.config.rope_base ** (-exponents / width))
        cos = mx.cos(angles)[:, None]
        sin = mx.sin(angles)[:, None]

        even = x[..., :width:2].astype(mx.float32)
        odd = x[..., 1:width:2].astype(mx.float32)

        rot_even = even * cos - odd * sin
        rot_odd = odd * cos + even * sin

        # Interleave back: rot_even and rot_odd into [..., width]
        stacked = mx.stack([rot_even, rot_odd], axis=-1)
        rotated = stacked.reshape(*x.shape[:-1], width).astype(x.dtype)
        passthrough = x[..., width:]
        return mx.concatenate([rotated, passthrough], axis=-1)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        cfg = self.config
        length = len(positions)

        qkv = self.wqkv(x)
        q = qkv[..., : self.q_width]
        gate = qkv[..., self.q_width : 2 * self.q_width]
        k = qkv[..., 2 * self.q_width : 2 * self.q_width + self.kv_width]
        v = qkv[..., 2 * self.q_width + self.kv_width :]

        q = self.q_norm(q.reshape(length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(length, cfg.n_local_heads, cfg.head_dim))

        q = self.rotate(q, positions).swapaxes(0, 1)[None]  # [1, H, T, D]
        k = self.rotate(k, positions).swapaxes(0, 1)[None]  # [1, H_kv, T, D]
        v = v.reshape(length, cfg.n_local_heads, cfg.head_dim).swapaxes(0, 1)[None]

        if self.k_cache is not None:
            end = int(positions[-1].item()) + 1
            self.k_cache[:, :, positions, :] = k
            self.v_cache[:, :, positions, :] = v
            k = self.k_cache[:, :, :end, :]
            v = self.v_cache[:, :, :end, :]
            key_positions = mx.arange(end)
        else:
            key_positions = positions

        mask_bool = key_positions[None, :] <= positions[:, None]
        mask = mx.where(mask_bool, 0.0, -1e9)[None, None, :, :]

        out = grouped_attention(q, k, v, mask=mask)
        out = out.squeeze(0).swapaxes(0, 1).reshape(length, self.q_width)
        return self.wo(out * nn.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.attention = TowerAttention(config)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.feed_forward = TowerMLP(config)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        residual = x + self.attention(self.attention_norm(x), positions)
        return residual + self.feed_forward(self.ffn_norm(residual))


class TowerLayer(nn.Module):
    """Mix code embedding into hidden state, then run transformer layer."""

    def __init__(self, config: AudioDecoderConfig, shared: bool = False):
        super().__init__()
        self.shared = shared
        self.vocab_size = config.vocab_size

        if shared:
            self.semantic_embedding = nn.Embedding(config.vocab_size, config.dim)
        else:
            self.codebook_embedding = nn.Embedding(config.vocab_size, config.dim)

        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(config)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        codes: mx.array,
        semantic: mx.array,
        positions: mx.array,
        shared_input: Optional[mx.array] = None,
    ) -> mx.array:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        clamped_codes = mx.clip(codes, 0, self.vocab_size - 1)
        emb = embedding(clamped_codes)

        e = mx.where(semantic[:, None], emb, 0.0)
        if shared_input is not None:
            e = e + shared_input

        combined = mx.concatenate([self.enorm(e), self.hnorm(hidden)], axis=-1)
        mixed = self.eh_proj(combined)
        out = self.final_layernorm(self.transformer_layer(mixed, positions))
        return out


class FishOmniAudioDecoder(nn.Module):
    """Drama 3 Audio Decoder: predicts codes 1..9 of each frame attending over time."""

    def __init__(self, config: Drama3Config):
        super().__init__()
        self.full_config = config
        cfg = self.config = config.audio_decoder
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id

        self.codebook_embeddings = nn.Embedding(cfg.vocab_size * cfg.num_codebooks, cfg.dim)
        self.shared_layers = [TowerLayer(cfg, shared=True) for _ in range(cfg.shared_enrichment_layers)]
        self.residual_layers = [TowerLayer(cfg, shared=False) for _ in range(cfg.num_codebooks - 1)]
        self.codebook_heads = [
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False) for _ in range(cfg.num_codebooks - 1)
        ]

        self._main_candidates: Optional[mx.array] = None

    def setup_caches(self, max_seq_len: int) -> None:
        for layer in (*self.shared_layers, *self.residual_layers):
            layer.transformer_layer.attention.setup_cache(max_seq_len)

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Sum embeddings of codes first..end-1 per frame."""
        if first == end:
            return mx.zeros((len(frames), self.config.dim))

        offsets = mx.arange(first, end) * self.config.vocab_size
        code_slice = frames[:, 1 + first : 1 + end] + offsets[None, :]

        # Flat indices for embedding lookup
        flat_indices = code_slice.reshape(-1)
        looked_up = self.codebook_embeddings(flat_indices).reshape(len(frames), end - first, self.config.dim)
        return mx.sum(looked_up, axis=1)

    def embed_backbone(self, frames: mx.array, text_embeddings: mx.array) -> mx.array:
        """Add codebook embeddings for semantic frames to backbone token embeddings."""
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0])
        return mx.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(
        self,
        hidden: mx.array,
        c0: mx.array,
        semantic: mx.array,
        positions: mx.array,
        frames: mx.array,
    ) -> mx.array:
        unseen = self.embedding_sum(
            frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks
        )
        unseen = mx.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)

        for index, layer in enumerate(self.shared_layers):
            hidden = layer(hidden, c0, semantic, positions, unseen if index == 0 else None)
        return hidden

    def prefill(
        self, hidden: mx.array, positions: mx.array, frames: mx.array, next_frames: mx.array
    ) -> None:
        """Teacher-force tower caches during prompt prefill."""
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, positions, frames)
        for layer, code in zip(self.residual_layers, next_frames[:, 1:].T):
            hidden = layer(hidden, code, semantic, positions)

    def decode_step(
        self,
        logits: mx.array,
        hidden: mx.array,
        positions: mx.array,
        frames: mx.array,
        history: mx.array,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.95,
    ) -> mx.array:
        """Decode step: sample main token, run shared layers, then 9 residual codebook layers."""
        if self._main_candidates is None:
            eos = mx.array([self.eos_id], dtype=mx.int32)
            band = mx.arange(self.semantic_begin, self.semantic_end + 1, dtype=mx.int32)
            candidates = mx.concatenate([eos, band])
            self._main_candidates = candidates[mx.argsort(candidates)]

        candidates = self._main_candidates
        candidate_logits = logits[:, candidates]

        first_idx = sample_top_k_top_p(candidate_logits, temperature, top_k, top_p)
        first = candidates[first_idx]

        retry_idx = sample_top_k_top_p(
            candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
        )
        retry = candidates[retry_idx]

        # Repetition-Aware Sampling
        recent_history = history[-RAS_WINDOW_FRAMES:]
        repeated = mx.any(recent_history == first[:, None], axis=-1)
        repeated = repeated & self.is_semantic(first)

        main = mx.where(repeated, retry, first)
        c0 = mx.clip(main - self.semantic_begin, 0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)

        hidden = self.run_shared(hidden, c0, semantic, positions, frames)

        codes = [c0]
        for layer, head in zip(self.residual_layers, self.codebook_heads):
            hidden = layer(hidden, codes[-1], semantic, positions)
            code_logits = head(hidden)
            next_code = sample_top_k_top_p(code_logits, temperature, top_k, top_p)
            codes.append(next_code)

        # Result frame: [1, 10 + 1] -> [main, c0, c1, ..., c9]
        return mx.stack([main, *codes], axis=-1)
