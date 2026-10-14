from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from drama3_mlx.config import FishOmniTextConfig

FP8_BLOCK_SIZE = 128


def dequantize_fp8_blocks(
    weight: mx.array, scale_inv: mx.array, dtype: mx.Dtype = mx.bfloat16
) -> mx.array:
    """Dequantize Block-FP8 [..., rows, cols] with scale_inv [..., rows/128, cols/128]."""
    *lead, rows, cols = weight.shape
    row_blocks = rows // FP8_BLOCK_SIZE
    col_blocks = cols // FP8_BLOCK_SIZE

    blocks = weight.reshape(*lead, row_blocks, FP8_BLOCK_SIZE, col_blocks, FP8_BLOCK_SIZE)
    scaled = blocks.astype(mx.float32) * scale_inv[..., :, None, :, None]
    return scaled.reshape(*lead, rows, cols).astype(dtype)


class DenseMLP(nn.Module):
    """Standard SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.Module):
    """Routed experts stacked per layer.
    Seamlessly supports both unquantized (BF16/FP16) and affine quantized (8-bit / 4-bit).
    """

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        intermediate: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.dtype = dtype

        self.group_size = 64
        self.bits = 8

        # Initialize unquantized placeholders (load_weights replaces them when loading quantized weights)
        self.w13_weight = mx.zeros((num_experts, 2 * intermediate, hidden), dtype=dtype)
        self.w2_weight = mx.zeros((num_experts, hidden, intermediate), dtype=dtype)

        # Pre-register scales and biases buffers for quantized checkpoints
        self.w13_scales = mx.zeros((0,), dtype=dtype)
        self.w13_biases = mx.zeros((0,), dtype=dtype)
        self.w2_scales = mx.zeros((0,), dtype=dtype)
        self.w2_biases = mx.zeros((0,), dtype=dtype)

    def _is_quantized(self) -> bool:
        return self.w13_scales is not None and self.w13_scales.size > 0

    def _get_bits(self) -> int:
        if self.w13_weight.dtype == mx.uint32:
            return 32 // (self.hidden // self.w13_weight.shape[-1])
        return 16

    def _get_expert_weights(self, idx: int) -> Tuple[mx.array, mx.array]:
        if self._is_quantized():
            bits = self._get_bits()
            w13 = mx.dequantize(
                self.w13_weight[idx],
                self.w13_scales[idx],
                self.w13_biases[idx],
                group_size=self.group_size,
                bits=bits,
            )
            w2 = mx.dequantize(
                self.w2_weight[idx],
                self.w2_scales[idx],
                self.w2_biases[idx],
                group_size=self.group_size,
                bits=bits,
            )
            return w13, w2
        return self.w13_weight[idx], self.w2_weight[idx]

    def __call__(self, x: mx.array, topk_ids: mx.array, topk_weights: mx.array) -> mx.array:
        """x: [N, H], topk_ids: [N, k], topk_weights: [N, k] -> [N, H]."""
        N, H = x.shape
        k = topk_ids.shape[1]

        if N == 1:
            token_x = x[0]
            ids = topk_ids[0]
            weights = topk_weights[0]

            out = mx.zeros((H,), dtype=x.dtype)
            for idx, w in zip(ids.tolist(), weights.tolist()):
                w13, w2 = self._get_expert_weights(idx)
                gate_up = mx.matmul(w13, token_x)
                gate = gate_up[: self.intermediate]
                up = gate_up[self.intermediate :]
                act = nn.silu(gate) * up
                down = mx.matmul(w2, act)
                out = out + (down * w)
            return out[None, :]

        # Batched / multi-token path
        out = mx.zeros((N, H), dtype=x.dtype)
        if self._is_quantized():
            unique_indices = set(topk_ids.reshape(-1).tolist())
            expert_cache = {idx: self._get_expert_weights(idx) for idx in unique_indices}
            for i in range(k):
                expert_indices = topk_ids[:, i].tolist()
                weights = topk_weights[:, i : i + 1]
                w13_batch = mx.stack([expert_cache[idx][0] for idx in expert_indices], axis=0)
                w2_batch = mx.stack([expert_cache[idx][1] for idx in expert_indices], axis=0)
                gate_up = mx.matmul(w13_batch, x[..., None]).squeeze(-1)
                gate = gate_up[:, : self.intermediate]
                up = gate_up[:, self.intermediate :]
                act = nn.silu(gate) * up
                down = mx.matmul(w2_batch, act[..., None]).squeeze(-1)
                out = out + (down * weights)
            return out

        for i in range(k):
            expert_indices = topk_ids[:, i]
            weights = topk_weights[:, i : i + 1]
            w13_batch = self.w13_weight[expert_indices]
            w2_batch = self.w2_weight[expert_indices]
            gate_up = mx.matmul(w13_batch, x[..., None]).squeeze(-1)
            gate = gate_up[:, : self.intermediate]
            up = gate_up[:, self.intermediate :]
            act = nn.silu(gate) * up
            down = mx.matmul(w2_batch, act[..., None]).squeeze(-1)
            out = out + (down * weights)

        return out


class SparseMoeBlock(nn.Module):
    """Top-k routed MoE with sigmoid-gated shared expert."""

    def __init__(self, config: FishOmniTextConfig, dtype: mx.Dtype = mx.bfloat16):
        super().__init__()
        hidden = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(hidden, config.num_experts, bias=False)
        self.experts = MoEExperts(
            config.num_experts, hidden, config.moe_intermediate_size, dtype=dtype
        )
        self.shared_expert = DenseMLP(hidden, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])

        router_logits = self.gate(flat).astype(mx.float32)
        probs = mx.softmax(router_logits, axis=-1)

        # Top-k selection
        topk_ids = mx.argpartition(-probs, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        topk_weights = mx.take_along_axis(probs, topk_ids, axis=-1)

        # Sort top-k by descending probability for numerical consistency
        order = mx.argsort(-topk_weights, axis=-1)
        topk_ids = mx.take_along_axis(topk_ids, order, axis=-1)
        topk_weights = mx.take_along_axis(topk_weights, order, axis=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / mx.sum(topk_weights, axis=-1, keepdims=True)

        expert_out = self.experts(flat, topk_ids, topk_weights.astype(flat.dtype))
        shared_gate = nn.sigmoid(self.shared_expert_gate(flat))
        shared_out = shared_gate * self.shared_expert(flat)

        return (expert_out + shared_out).reshape(orig_shape)
