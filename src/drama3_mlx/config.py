from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

FISH_OMNI_MODEL_TYPE = "fish_omni"
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"


@dataclass
class FishOmniTextConfig:
    vocab_size: int = 252288
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    layer_types: List[str] = field(
        default_factory=lambda: [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 10
    )
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FishOmniTextConfig:
        fields = {
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "max_position_embeddings",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
        }
        kwargs = {k: v for k, v in data.items() if k in fields}
        if "rope_parameters" in data and isinstance(data["rope_parameters"], dict):
            rp = data["rope_parameters"]
            if "rope_theta" in rp:
                kwargs["rope_theta"] = float(rp["rope_theta"])
            if "partial_rotary_factor" in rp:
                kwargs["partial_rotary_factor"] = float(rp["partial_rotary_factor"])
        elif "rope_theta" in data:
            kwargs["rope_theta"] = float(data["rope_theta"])
        return cls(**kwargs)


@dataclass
class AudioDecoderConfig:
    dim: int = 2048
    intermediate_size: int = 6144
    n_head: int = 16
    n_local_heads: int = 2
    head_dim: int = 256
    norm_eps: float = 1e-6
    rope_base: float = 10000000.0
    rotary_percent: float = 0.25
    vocab_size: int = 4096
    num_codebooks: int = 10
    shared_enrichment_layers: int = 4
    vq_backbone_codebooks: int = 1

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioDecoderConfig:
        fields = {
            "dim",
            "intermediate_size",
            "n_head",
            "n_local_heads",
            "head_dim",
            "norm_eps",
            "rope_base",
            "rotary_percent",
            "vocab_size",
            "num_codebooks",
            "shared_enrichment_layers",
            "vq_backbone_codebooks",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class Drama3Config:
    text: FishOmniTextConfig = field(default_factory=FishOmniTextConfig)
    audio_decoder: AudioDecoderConfig = field(default_factory=AudioDecoderConfig)
    semantic_start_token_id: int = 248099
    semantic_end_token_id: int = 252194
    eos_token_id: int = 248046
    pad_token_id: int = 248044
    audio_pad_token_id: int = 248076
    speech_template_version: str = SPEECH_TEMPLATE_VERSION
    model_type: str = FISH_OMNI_MODEL_TYPE
    codec_sample_rate: int = 44100
    codec_num_codebooks: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Drama3Config:
        text_cfg = FishOmniTextConfig.from_dict(data.get("text_config", {}))
        audio_cfg = AudioDecoderConfig.from_dict(data.get("audio_decoder_config", {}))
        return cls(
            text=text_cfg,
            audio_decoder=audio_cfg,
            semantic_start_token_id=data.get("semantic_start_token_id", 248099),
            semantic_end_token_id=data.get("semantic_end_token_id", 252194),
            eos_token_id=data.get("eos_token_id", 248046),
            pad_token_id=data.get("pad_token_id", 248044),
            audio_pad_token_id=data.get("audio_pad_token_id", 248076),
            speech_template_version=data.get("speech_template_version", SPEECH_TEMPLATE_VERSION),
            model_type=data.get("model_type", FISH_OMNI_MODEL_TYPE),
        )

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path) -> Drama3Config:
        path = Path(path_or_repo)
        if not path.is_dir():
            raise FileNotFoundError(f"Configuration directory not found: {path}")
        with open(path / "config.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
