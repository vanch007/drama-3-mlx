from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from drama3_mlx.config import FishOmniTextConfig
from drama3_mlx.layers import (
    FullAttention,
    GatedDeltaNet,
    LayerContext,
    ZeroCenteredRMSNorm,
)
from drama3_mlx.moe import SparseMoeBlock


class DecoderLayer(nn.Module):
    def __init__(self, config: FishOmniTextConfig, layer_index: int, dtype: mx.Dtype = mx.bfloat16):
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "full_attention":
            self.self_attn = FullAttention(config)
        else:
            self.linear_attn = GatedDeltaNet(config)

        self.mlp = SparseMoeBlock(config, dtype=dtype)
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        mixer = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        x = x + mixer(self.input_layernorm(x), context)
        return x + self.mlp(self.post_attention_layernorm(x))


class FishOmniBackbone(nn.Module):
    """Drama 3 text backbone: hybrid linear/full attention with 256 MoE experts."""

    def __init__(self, config: FishOmniTextConfig, dtype: mx.Dtype = mx.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i, dtype=dtype) for i in range(config.num_hidden_layers)]
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE table
        self._cos_table: Optional[mx.array] = None
        self._sin_table: Optional[mx.array] = None
        self._init_rope_table(max(config.max_position_embeddings, 16384))

    def _init_rope_table(self, max_seq_len: int) -> None:
        dim = self.config.rotary_dim
        base = self.config.rope_theta
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        self._cos_table = mx.cos(freqs)
        self._sin_table = mx.sin(freqs)

    def allocate_caches(self, batch_size: int, max_seq_len: int) -> None:
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.allocate_cache(batch_size, max_seq_len, dtype=self.dtype)
            else:
                layer.linear_attn.allocate_cache(batch_size, dtype=self.dtype)

    def embed(self, token_ids: mx.array) -> mx.array:
        return self.embed_tokens(token_ids)

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        return self.lm_head(self.norm(hidden_states))

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        input_pos: Optional[mx.array] = None,
        use_cache: bool = True,
        continue_state: bool = False,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            x = self.embed(input_ids)
        else:
            x = inputs_embeds

        batch, length, _ = x.shape
        if input_pos is None:
            input_pos = mx.arange(length)

        # Extract RoPE frequencies for current positions
        if input_pos.ndim == 1:
            pos_indices = input_pos
        else:
            pos_indices = input_pos[0]

        cos = self._cos_table[pos_indices]
        sin = self._sin_table[pos_indices]

        kv_len = int(pos_indices[-1].item()) + 1 if len(pos_indices) > 0 else length
        context = LayerContext(
            cos=cos,
            sin=sin,
            input_pos=input_pos,
            kv_len=kv_len,
            use_cache=use_cache,
            continue_state=continue_state,
        )

        for layer in self.layers:
            x = layer(x, context)

        return x
