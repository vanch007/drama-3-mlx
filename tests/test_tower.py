import unittest
import mlx.core as mx

from drama3_mlx.config import AudioDecoderConfig, Drama3Config, FishOmniTextConfig
from drama3_mlx.tower import (
    FishOmniAudioDecoder,
    TowerAttention,
    TowerLayer,
    TowerMLP,
    sample_top_k_top_p,
)


class TestTower(unittest.TestCase):
    def test_sample_top_k_top_p(self):
        logits = mx.array([[10.0, 1.0, 0.5, 0.1]])
        idx = sample_top_k_top_p(logits, temperature=0.01, top_k=1, top_p=0.9)
        self.assertEqual(int(idx[0].item()), 0)

    def test_tower_attention(self):
        cfg = AudioDecoderConfig(
            dim=16,
            n_head=2,
            n_local_heads=1,
            head_dim=8,
            rotary_percent=0.5,
        )
        attn = TowerAttention(cfg)
        x = mx.random.normal((3, 16))
        pos = mx.arange(3)
        out = attn(x, pos)
        self.assertEqual(out.shape, (3, 16))

    def test_audio_decoder(self):
        text_cfg = FishOmniTextConfig(vocab_size=100, hidden_size=16)
        audio_cfg = AudioDecoderConfig(
            dim=16,
            intermediate_size=32,
            n_head=2,
            n_local_heads=1,
            head_dim=8,
            vocab_size=16,
            num_codebooks=3,
            shared_enrichment_layers=1,
        )
        cfg = Drama3Config(
            text=text_cfg,
            audio_decoder=audio_cfg,
            semantic_start_token_id=20,
            semantic_end_token_id=35,
            eos_token_id=2,
        )
        decoder = FishOmniAudioDecoder(cfg)
        self.assertTrue(bool(decoder.is_semantic(mx.array([25]))[0].item()))
        self.assertFalse(bool(decoder.is_semantic(mx.array([5]))[0].item()))


if __name__ == "__main__":
    unittest.main()
