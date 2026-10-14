import unittest
import mlx.core as mx
import numpy as np

from drama3_mlx.config import FishOmniTextConfig
from drama3_mlx.layers import (
    FullAttention,
    GatedDeltaNet,
    GatedRMSNorm,
    LayerContext,
    ZeroCenteredRMSNorm,
    apply_neox_rope,
    chunk_gated_delta_rule,
    grouped_attention,
    recurrent_gated_delta_rule,
)


class TestLayers(unittest.TestCase):
    def test_zero_centered_rmsnorm(self):
        norm = ZeroCenteredRMSNorm(16)
        x = mx.random.normal((2, 5, 16))
        out = norm(x)
        self.assertEqual(out.shape, (2, 5, 16))
        # Variance along last dim should be close to 1
        var = mx.mean(mx.square(out), axis=-1)
        np.testing.assert_allclose(np.array(var), 1.0, atol=1e-3)

    def test_gated_rmsnorm(self):
        norm = GatedRMSNorm(16)
        x = mx.random.normal((2, 5, 16))
        gate = mx.zeros((2, 5, 16))  # silu(0) = 0
        out = norm(x, gate)
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-5)

    def test_neox_rope(self):
        x = mx.random.normal((1, 4, 2, 16))
        cos = mx.ones((4, 8))
        sin = mx.zeros((4, 8))
        out = apply_neox_rope(x, cos, sin)
        # With cos=1 and sin=0, rotary part should be unchanged
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-5)

    def test_delta_rule_consistency(self):
        # Test that recurrent and chunked rules match
        batch, length, heads, dk, dv = 1, 8, 2, 4, 4
        query = mx.random.normal((batch, length, heads, dk))
        key = mx.random.normal((batch, length, heads, dk))
        value = mx.random.normal((batch, length, heads, dv))
        decay = -mx.abs(mx.random.normal((batch, length, heads))) * 0.1
        beta = mx.sigmoid(mx.random.normal((batch, length, heads)))

        out_rec, state_rec = recurrent_gated_delta_rule(query, key, value, decay, beta)
        out_chunk, state_chunk = chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=4)

        np.testing.assert_allclose(np.array(out_rec), np.array(out_chunk), atol=1e-4)
        np.testing.assert_allclose(np.array(state_rec), np.array(state_chunk), atol=1e-4)

    def test_full_attention(self):
        cfg = FishOmniTextConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            partial_rotary_factor=0.5,
        )
        attn = FullAttention(cfg)
        x = mx.random.normal((1, 4, 32))
        cos = mx.ones((4, 4))
        sin = mx.zeros((4, 4))
        ctx = LayerContext(cos=cos, sin=sin, input_pos=mx.arange(4), kv_len=4, use_cache=False)
        out = attn(x, ctx)
        self.assertEqual(out.shape, (1, 4, 32))


if __name__ == "__main__":
    unittest.main()
