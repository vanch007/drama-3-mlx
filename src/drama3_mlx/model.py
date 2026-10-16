from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from drama3_mlx.backbone import FishOmniBackbone
from drama3_mlx.codec import DAC, build_ae
from drama3_mlx.config import Drama3Config
from drama3_mlx.generate import (
    generate_drama3,
    generate_drama3_stream,
    generate_long_speech,
)
from drama3_mlx.prompt import build_drama3_prompt
from drama3_mlx.tower import FishOmniAudioDecoder


def map_tower_key(k: str) -> str:
    for prefix, repl in [
        ("audio_decoder_mtp_shared.layers.", "shared_layers."),
        ("audio_decoder_mtp.layers.", "residual_layers."),
        ("audio_decoder_mtp.heads.", "codebook_heads."),
        ("audio_decoder.codebook_embeddings.", "codebook_embeddings."),
    ]:
        if k.startswith(prefix):
            return repl + k[len(prefix):]
    return k


def map_backbone_key(k: str) -> str:
    if k.startswith("model.language_model."):
        k = k[len("model.language_model.") :]
    if ".linear_attn.conv1d.weight" in k:
        k = k.replace(".linear_attn.conv1d.weight", ".linear_attn.conv1d_weight")
    return k


class Drama3Model(nn.Module):
    """Unified pure MLX model for Fish Audio Drama 3.
    Combines text backbone, audio decoder tower, tokenizer, and DAC neural audio codec.
    """

    def __init__(
        self,
        config: Drama3Config,
        backbone: FishOmniBackbone,
        audio_decoder: FishOmniAudioDecoder,
        codec: DAC,
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.audio_decoder = audio_decoder
        self.codec = codec
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_repo: str | Path,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> Drama3Model:
        model_dir = Path(model_path_or_repo).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        config = Drama3Config.from_pretrained(model_dir)

        # 1. Tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Missing tokenizer.json in {model_dir}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # 2. Backbone & Tower
        backbone = FishOmniBackbone(config.text, dtype=dtype)
        audio_decoder = FishOmniAudioDecoder(config)

        # 3. DAC Codec
        codec = build_ae()
        codec_file = model_dir / "codec.safetensors"
        if codec_file.is_file():
            codec_weights = mx.load(str(codec_file))
            codec.load_weights(list(codec_weights.items()), strict=False)

        # 4. Stream load sharded weights into backbone and tower
        weight_files = sorted(model_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {model_dir}")

        for wf in weight_files:
            if wf.name == "codec.safetensors":
                continue
            shard = mx.load(str(wf))
            bb_weights = {}
            tw_weights = {}
            for k, v in shard.items():
                if "audio_decoder" in k:
                    tw_weights[map_tower_key(k)] = v
                else:
                    bb_weights[map_backbone_key(k)] = v

            if bb_weights:
                backbone.load_weights(list(bb_weights.items()), strict=False)
            if tw_weights:
                audio_decoder.load_weights(list(tw_weights.items()), strict=False)
            del shard, bb_weights, tw_weights

        return cls(config, backbone, audio_decoder, codec, tokenizer)

    def encode_audio(self, audio_source: Union[str, Path, np.ndarray], sample_rate: int = 44100) -> mx.array:
        """Encode audio into 10 codebooks: returns [num_codebooks, num_frames]."""
        if isinstance(audio_source, (str, Path)):
            audio, sr = sf.read(str(audio_source))
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            if sr != self.config.codec_sample_rate:
                import scipy.signal
                num_samples = int(len(audio) * self.config.codec_sample_rate / sr)
                audio = scipy.signal.resample(audio, num_samples)
        else:
            audio = audio_source

        audio_tensor = mx.array(audio[None, None, :], dtype=mx.float32)
        codes, _ = self.codec.encode(audio_tensor)
        return codes[0]

    def decode_codes_to_audio(self, codes: mx.array) -> np.ndarray:
        """Decode 10 codebooks [num_codebooks, num_frames] into 44.1kHz float32 audio."""
        if codes.shape[-1] == 0:
            return np.zeros((0,), dtype=np.float32)

        if codes.ndim == 2:
            codes_input = codes[None]
        else:
            codes_input = codes

        feature_lengths = mx.array([codes_input.shape[-1]], dtype=mx.int32)
        audio, lengths = self.codec.decode(codes_input, feature_lengths)
        length = int(lengths[0].item())
        audio = audio[..., :length]
        audio_np = np.array(audio.squeeze())
        return audio_np

    def tts(
        self,
        text: str,
        *,
        ref_audio: Optional[Union[str, Path, np.ndarray]] = None,
        ref_text: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> np.ndarray:
        """Single or multi-character text-to-speech with optional zero-shot reference cloning."""
        prompt_texts = []
        prompt_tokens = []

        if ref_audio is not None:
            if not ref_text:
                raise ValueError("ref_text transcript is required when ref_audio is provided")
            codes = self.encode_audio(ref_audio)
            prompt_texts.append(ref_text)
            prompt_tokens.append(codes)

        return generate_long_speech(
            self,
            text,
            prompt_texts=prompt_texts,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def tts_stream(
        self,
        text: str,
        *,
        ref_audio: Optional[Union[str, Path, np.ndarray]] = None,
        ref_text: Optional[str] = None,
        chunk_frames: int = 10,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> Generator[np.ndarray, None, None]:
        """Streaming TTS generator: yields numpy audio chunks in real-time."""
        prompt_texts = []
        prompt_tokens = []

        if ref_audio is not None:
            if not ref_text:
                raise ValueError("ref_text transcript is required when ref_audio is provided")
            codes = self.encode_audio(ref_audio)
            prompt_texts.append(ref_text)
            prompt_tokens.append(codes)

        prompt = build_drama3_prompt(
            text=text,
            tokenizer=self.tokenizer,
            config=self.config,
            prompt_texts=prompt_texts,
            prompt_tokens=prompt_tokens,
        )

        yield from generate_drama3_stream(
            self,
            prompt,
            chunk_frames=chunk_frames,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
