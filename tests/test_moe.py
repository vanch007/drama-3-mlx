import unittest
import mlx.core as mx
import numpy as np

from drama3_mlx.config import FishOmniTextConfig
from drama3_mlx.moe import (
    DenseMLP,
    MoEExperts,
    SparseMoeBlock,
    dequantize_fp8_blocks,
)


class TestMoE(unittest.TestCase):
    def test_dense_mlp(self):
        mlp = DenseMLP(hidden=16, intermediate=32)
        x = mx.random.normal((2, 16))
        out = mlp(x)
        self.assertEqual(out.shape, (2, 16))

    def test_moe_experts_single_and_batched(self):
        experts = MoEExperts(num_experts=4, hidden=16, intermediate=32, dtype=mx.float32)
        # random init weights
        experts.w13_weight = mx.random.normal((4, 64, 16))
        experts.w2_weight = mx.random.normal((4, 16, 32))

        # Single token test (N=1)
        x_single = mx.random.normal((1, 16))
        topk_ids_single = mx.array([[0, 2]])
        topk_weights_single = mx.array([[0.6, 0.4]])
        out_single = experts(x_single, topk_ids_single, topk_weights_single)
        self.assertEqual(out_single.shape, (1, 16))

        # Batched test (N=3)
        x_batch = mx.random.normal((3, 16))
        topk_ids_batch = mx.array([[0, 1], [2, 3], [1, 2]])
        topk_weights_batch = mx.array([[0.7, 0.3], [0.5, 0.5], [0.8, 0.2]])
        out_batch = experts(x_batch, topk_ids_batch, topk_weights_batch)
        self.assertEqual(out_batch.shape, (3, 16))

    def test_sparse_moe_block(self):
        cfg = FishOmniTextConfig(
            hidden_size=16,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=16,
        )
        block = SparseMoeBlock(cfg, dtype=mx.float32)
        x = mx.random.normal((1, 4, 16))
        out = block(x)
        self.assertEqual(out.shape, (1, 4, 16))


if __name__ == "__main__":
    unittest.main()
