from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from drama3_mlx.config import FishOmniTextConfig

GDN_CHUNK_SIZE = 64


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        xf = x.astype(mx.float32)
        variance = mx.mean(mx.square(xf), axis=-1, keepdims=True)
        out = xf * mx.rsqrt(variance + self.eps)
        out = out * (1.0 + self.weight.astype(mx.float32))
        return out.astype(orig_dtype)


class GatedRMSNorm(nn.Module):
    """Gated DeltaNet output norm: weight * rmsnorm(x) * silu(gate)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        orig_dtype = x.dtype
        xf = x.astype(mx.float32)
        variance = mx.mean(mx.square(xf), axis=-1, keepdims=True)
        normed = xf * mx.rsqrt(variance + self.eps)
        normed = (self.weight * normed).astype(orig_dtype)
        gate_act = nn.silu(gate.astype(mx.float32)).astype(orig_dtype)
        return normed * gate_act


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """L2 normalization along the last dimension."""
    xf = x.astype(mx.float32)
    norm = mx.rsqrt(mx.sum(mx.square(xf), axis=-1, keepdims=True) + eps)
    return (xf * norm).astype(x.dtype)


def apply_neox_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the first rotary_dim channels of x [B, T, heads, head_dim] (half-split)."""
    rotary_dim = cos.shape[-1]
    rotary = x[..., :rotary_dim].astype(mx.float32)
    passthrough = x[..., rotary_dim:]
    d = rotary_dim // 2
    first = rotary[..., :d]
    second = rotary[..., d:]
    rotated = mx.concatenate([-second, first], axis=-1)

    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    out = rotary * cos + rotated * sin
    return mx.concatenate([out.astype(x.dtype), passthrough], axis=-1)


def grouped_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: Optional[mx.array] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> mx.array:
    """Grouped query attention.
    query: [B, H, T_q, D]
    key:   [B, H_kv, T_k, D]
    value: [B, H_kv, T_k, D_v]
    """
    B, H, T_q, D = query.shape
    _, H_kv, T_k, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if H != H_kv:
        reps = H // H_kv
        key = mx.repeat(key, reps, axis=1)
        value = mx.repeat(value, reps, axis=1)

    scores = mx.matmul(query * scale, key.swapaxes(-1, -2))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        scores = scores + mask
    elif is_causal and T_q > 1:
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T_q)
        scores = scores + causal_mask

    attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype)
    return mx.matmul(attn_weights, value)


def recurrent_gated_delta_rule(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    decay: mx.array,
    beta: mx.array,
    initial_state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Token-by-token gated delta rule in fp32.
    query, key: [B, T, H, dk]
    value:      [B, T, H, dv]
    decay, beta: [B, T, H]
    initial_state: [B, H, dk, dv]
    Returns (output [B, T, H, dv], final_state [B, H, dk, dv]).
    """
    orig_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]

    q_normed = l2norm(query.astype(mx.float32)) * (key_dim ** -0.5)
    k_normed = l2norm(key.astype(mx.float32))
    v_f = value.astype(mx.float32)
    b_f = beta.astype(mx.float32)
    d_f = decay.astype(mx.float32)

    if initial_state is None:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
    else:
        state = initial_state.astype(mx.float32)

    outputs = []
    for t in range(length):
        k_t = k_normed[:, t]  # [B, H, dk]
        q_t = q_normed[:, t]  # [B, H, dk]
        v_t = v_f[:, t]       # [B, H, dv]
        decay_step = mx.exp(d_f[:, t])[..., None, None]  # [B, H, 1, 1]
        beta_step = b_f[:, t][..., None]                  # [B, H, 1]

        state = state * decay_step
        remembered = mx.sum(state * k_t[..., None], axis=-2)  # [B, H, dv]
        delta = (v_t - remembered) * beta_step                 # [B, H, dv]
        state = state + (k_t[..., None] * delta[..., None, :]) # [B, H, dk, dv]

        out_step = mx.sum(state * q_t[..., None], axis=-2)     # [B, H, dv]
        outputs.append(out_step)

    out = mx.stack(outputs, axis=1).astype(orig_dtype)
    return out, state


def chunk_gated_delta_rule(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    decay: mx.array,
    beta: mx.array,
    initial_state: Optional[mx.array] = None,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> Tuple[mx.array, mx.array]:
    """Chunked gated delta rule for prefill.
    For arbitrary length, handles sequential chunks using recurrent updates at chunk boundaries.
    """
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]

    if length <= chunk_size:
        return recurrent_gated_delta_rule(
            query, key, value, decay, beta, initial_state=initial_state
        )

    # Process in chunks of chunk_size
    state = initial_state
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        q_chunk = query[:, start:end]
        k_chunk = key[:, start:end]
        v_chunk = value[:, start:end]
        d_chunk = decay[:, start:end]
        b_chunk = beta[:, start:end]

        out_chunk, state = recurrent_gated_delta_rule(
            q_chunk, k_chunk, v_chunk, d_chunk, b_chunk, initial_state=state
        )
        outputs.append(out_chunk)

    return mx.concatenate(outputs, axis=1), state


@dataclass
class LayerContext:
    cos: mx.array
    sin: mx.array
    input_pos: mx.array
    kv_len: int
    use_cache: bool = True
    continue_state: bool = False


class FullAttention(nn.Module):
    """Gated softmax attention: GQA, q/k norm, partial RoPE, sigmoid output gate."""

    def __init__(self, config: FishOmniTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)

        self.k_cache: Optional[mx.array] = None
        self.v_cache: Optional[mx.array] = None

    def allocate_cache(self, batch_size: int, max_seq_len: int, dtype: mx.Dtype = mx.float16) -> None:
        shape = (batch_size, self.num_kv_heads, max_seq_len, self.head_dim)
        self.k_cache = mx.zeros(shape, dtype=dtype)
        self.v_cache = mx.zeros(shape, dtype=dtype)

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        batch, length, _ = x.shape
        q_proj_out = self.q_proj(x)
        q_gate = q_proj_out.reshape(batch, length, self.num_heads, 2 * self.head_dim)
        query = q_gate[..., : self.head_dim]
        gate = q_gate[..., self.head_dim :].reshape(batch, length, self.num_heads * self.head_dim)

        query = self.q_norm(query)
        key = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key)
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)

        query = apply_neox_rope(query, context.cos, context.sin).swapaxes(1, 2)
        key = apply_neox_rope(key, context.cos, context.sin).swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        if context.use_cache:
            if self.k_cache is None:
                self.k_cache = key
                self.v_cache = value
            else:
                pos_val = context.input_pos
                start_pos = int(pos_val[0].item()) if (hasattr(pos_val, "ndim") and pos_val.ndim > 0) else int(pos_val.item() if hasattr(pos_val, "item") else pos_val)
                self.k_cache[:, :, start_pos : start_pos + length, :] = key
                self.v_cache[:, :, start_pos : start_pos + length, :] = value
                key = self.k_cache[:, :, : context.kv_len, :]
                value = self.v_cache[:, :, : context.kv_len, :]

            pos_indices = mx.arange(context.kv_len)[None, :]
            input_pos_col = context.input_pos.reshape(-1, 1)
            mask_bool = pos_indices <= input_pos_col
            mask = mx.where(mask_bool, 0.0, -1e9)[None, None, :, :]
            is_causal = False
        else:
            mask = None
            is_causal = length > 1

        attended = grouped_attention(
            query, key, value, mask=mask, is_causal=is_causal, scale=self.head_dim ** -0.5
        )
        attended = attended.swapaxes(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(attended * nn.sigmoid(gate))


class GatedDeltaNet(nn.Module):
    """Linear Attention (Gated DeltaNet) with depthwise Conv1d and recurrent delta rule."""

    def __init__(self, config: FishOmniTextConfig):
        super().__init__()
        hidden = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False)

        # Depthwise 1D convolution: weight shape in MLX conv1d is [out_channels, kernel_size, in_channels/groups]
        self.conv1d_weight = mx.zeros((self.conv_dim, self.conv_kernel_size, 1))

        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))

        self.norm = GatedRMSNorm(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

        self.conv_state: Optional[mx.array] = None
        self.recurrent_state: Optional[mx.array] = None

    def allocate_cache(self, batch_size: int, dtype: mx.Dtype = mx.float32) -> None:
        self.conv_state = mx.zeros((batch_size, self.conv_kernel_size - 1, self.conv_dim), dtype=dtype)
        self.recurrent_state = mx.zeros(
            (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=mx.float32
        )

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x)  # [B, T, conv_dim]
        history = self.conv_kernel_size - 1

        if context.continue_state and self.conv_state is not None:
            previous = self.conv_state[:batch]
        else:
            previous = mx.zeros((batch, history, self.conv_dim), dtype=mixed.dtype)

        padded = mx.concatenate([previous, mixed], axis=1)  # [B, history + T, conv_dim]

        if context.use_cache and self.conv_state is not None:
            self.conv_state[:batch] = padded[:, -history:, :]

        # Depthwise conv1d in MLX: input [B, T, C], weight [C, K, 1]
        conv_out = mx.conv1d(padded, self.conv1d_weight, groups=self.conv_dim)
        mixed_act = nn.silu(conv_out)

        query = mixed_act[..., : self.key_dim]
        key = mixed_act[..., self.key_dim : 2 * self.key_dim]
        value = mixed_act[..., 2 * self.key_dim :]

        query = query.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, length, self.num_v_heads, self.head_v_dim)

        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        beta = nn.sigmoid(self.in_proj_b(x).astype(mx.float32))

        # decay = -exp(A_log) * softplus(in_proj_a + dt_bias)
        decay = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(
            self.in_proj_a(x).astype(mx.float32) + self.dt_bias.astype(mx.float32)
        )

        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = mx.repeat(query, groups, axis=2)
            key = mx.repeat(key, groups, axis=2)

        initial_state = self.recurrent_state[:batch] if (context.continue_state and self.recurrent_state is not None) else None

        rule = (
            recurrent_gated_delta_rule
            if (context.continue_state and length == 1)
            else chunk_gated_delta_rule
        )
        core, final_state = rule(query, key, value, decay, beta, initial_state=initial_state)

        if context.use_cache and self.recurrent_state is not None:
            self.recurrent_state[:batch] = final_state

        core_normed = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        ).reshape(batch, length, self.value_dim)

        return self.out_proj(core_normed)
