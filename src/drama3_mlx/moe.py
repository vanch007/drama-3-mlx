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
    """Routed experts stacked per layer:
    w13_weight: [E, 2*I, H] (gate then up stacked on dim 1)
    w2_weight:  [E, H, I]   (down proj, transposed for MLX Linear: x @ w2_weight.T or batch matmul)
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

        # w13: [E, 2*I, H], w2: [E, H, I]
        self.w13_weight = mx.zeros((num_experts, 2 * intermediate, hidden), dtype=dtype)
        self.w2_weight = mx.zeros((num_experts, hidden, intermediate), dtype=dtype)

    def __call__(self, x: mx.array, topk_ids: mx.array, topk_weights: mx.array) -> mx.array:
        """x: [N, H], topk_ids: [N, k], topk_weights: [N, k] -> [N, H]."""
        N, H = x.shape
        k = topk_ids.shape[1]

        if N == 1:
            # Fast decode path for single token
            token_x = x[0]  # [H]
            ids = topk_ids[0]  # [k]
            weights = topk_weights[0]  # [k]

            out = mx.zeros((H,), dtype=x.dtype)
            for idx, w in zip(ids.tolist(), weights.tolist()):
                w13 = self.w13_weight[idx]  # [2*I, H]
                gate_up = mx.matmul(w13, token_x)  # [2*I]
                gate = gate_up[: self.intermediate]
                up = gate_up[self.intermediate :]
                act = nn.silu(gate) * up  # [I]
                down = mx.matmul(self.w2_weight[idx], act)  # [H]
                out = out + (down * w)
            return out[None, :]

        # Batched / multi-token path
        out = mx.zeros((N, H), dtype=x.dtype)
        for i in range(k):
            expert_indices = topk_ids[:, i]  # [N]
            weights = topk_weights[:, i : i + 1]  # [N, 1]

            # Gather weights for each token's i-th expert: [N, 2*I, H]
            w13_batch = self.w13_weight[expert_indices]
            w2_batch = self.w2_weight[expert_indices]

            # gate_up: [N, 2*I] = sum_H (x[N, 1, H] * w13_batch[N, 2*I, H])
            gate_up = mx.matmul(w13_batch, x[..., None]).squeeze(-1)  # [N, 2*I]
            gate = gate_up[:, : self.intermediate]
            up = gate_up[:, self.intermediate :]
            act = nn.silu(gate) * up  # [N, I]

            down = mx.matmul(w2_batch, act[..., None]).squeeze(-1)  # [N, H]
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
