import unittest
from types import SimpleNamespace
import mlx.core as mx
import numpy as np

from drama3_mlx.backbone import FishOmniBackbone
from drama3_mlx.codec import build_ae
from drama3_mlx.config import AudioDecoderConfig, Drama3Config, FishOmniTextConfig
from drama3_mlx.generate import generate_drama3
from drama3_mlx.model import Drama3Model
from drama3_mlx.prompt import build_drama3_prompt
from drama3_mlx.tower import FishOmniAudioDecoder


class MockTokenizer:
    def encode(self, text: str):
        return SimpleNamespace(ids=[hash(c) % 20 for c in text])


class TestEndToEnd(unittest.TestCase):
    def test_tiny_model_generation(self):
        text_cfg = FishOmniTextConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            max_position_embeddings=128,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
        )
        audio_cfg = AudioDecoderConfig(
            dim=16,
            intermediate_size=32,
            n_head=2,
            n_local_heads=1,
            head_dim=8,
            vocab_size=8,
            num_codebooks=3,
            shared_enrichment_layers=1,
            vq_backbone_codebooks=1,
            rope_base=10000.0,
            rotary_percent=0.5,
        )
        cfg = Drama3Config(
            text=text_cfg,
            audio_decoder=audio_cfg,
            semantic_start_token_id=25,
            semantic_end_token_id=32,
            eos_token_id=2,
        )

        tokenizer = MockTokenizer()
        backbone = FishOmniBackbone(text_cfg, dtype=mx.float32)
        tower = FishOmniAudioDecoder(cfg)
        codec = build_ae()

        model = Drama3Model(cfg, backbone, tower, codec, tokenizer)

        prompt = build_drama3_prompt("Hello", tokenizer, cfg)
        self.assertEqual(prompt.shape[1], 4)  # 1 + 3 codebooks

        codes, reason = generate_drama3(model, prompt, max_new_tokens=5, top_k=2)
        self.assertEqual(codes.shape[0], 3)
        self.assertIn(reason, ["eos", "length"])


if __name__ == "__main__":
    unittest.main()
