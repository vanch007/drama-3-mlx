"""Drama 3 MLX: Pure Apple Silicon MLX implementation of Fish Audio Drama 3."""

from drama3_mlx.backbone import FishOmniBackbone
from drama3_mlx.chunking import Drama3History, split_drama3_text
from drama3_mlx.codec import DAC, build_ae
from drama3_mlx.config import AudioDecoderConfig, Drama3Config, FishOmniTextConfig
from drama3_mlx.generate import (
    generate_drama3,
    generate_drama3_stream,
    generate_long_speech,
)
from drama3_mlx.model import Drama3Model
from drama3_mlx.prompt import build_drama3_prompt, with_speaker_tag
from drama3_mlx.tower import FishOmniAudioDecoder

__version__ = "0.1.0"

__all__ = [
    "Drama3Model",
    "Drama3Config",
    "FishOmniTextConfig",
    "AudioDecoderConfig",
    "FishOmniBackbone",
    "FishOmniAudioDecoder",
    "DAC",
    "build_ae",
    "build_drama3_prompt",
    "with_speaker_tag",
    "split_drama3_text",
    "Drama3History",
    "generate_drama3",
    "generate_drama3_stream",
    "generate_long_speech",
]
