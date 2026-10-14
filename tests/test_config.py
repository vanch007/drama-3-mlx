import unittest
from drama3_mlx.config import Drama3Config, FishOmniTextConfig, AudioDecoderConfig

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = Drama3Config()
        self.assertEqual(cfg.text.hidden_size, 2048)
        self.assertEqual(cfg.text.num_hidden_layers, 40)
        self.assertEqual(cfg.text.num_experts, 256)
        self.assertEqual(cfg.text.num_experts_per_tok, 8)
        self.assertEqual(cfg.text.rotary_dim, 64)
        self.assertEqual(cfg.audio_decoder.num_codebooks, 10)
        self.assertEqual(cfg.audio_decoder.vocab_size, 4096)
        self.assertEqual(cfg.audio_decoder.rotary_dim, 64)
        self.assertEqual(cfg.semantic_start_token_id, 248099)
        self.assertEqual(cfg.semantic_end_token_id, 252194)
        self.assertEqual(cfg.eos_token_id, 248046)

    def test_from_dict(self):
        d = {
            "text_config": {
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "rope_parameters": {"rope_theta": 5000.0, "partial_rotary_factor": 0.5},
            },
            "audio_decoder_config": {
                "dim": 128,
                "vocab_size": 512,
                "num_codebooks": 4,
            },
            "semantic_start_token_id": 100,
            "semantic_end_token_id": 200,
        }
        cfg = Drama3Config.from_dict(d)
        self.assertEqual(cfg.text.vocab_size, 1000)
        self.assertEqual(cfg.text.hidden_size, 128)
        self.assertEqual(cfg.text.rope_theta, 5000.0)
        self.assertEqual(cfg.text.rotary_dim, 16)
        self.assertEqual(cfg.audio_decoder.num_codebooks, 4)

if __name__ == "__main__":
    unittest.main()
