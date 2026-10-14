#!/usr/bin/env python3
"""Automated pipeline to upload all 3 Drama 3 MLX models to Hugging Face with updated READMEs."""

import os
import shutil
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

API = HfApi()
ROOT = Path("/Users/vanch/drama-3-mlx")
CHECKPOINT_DIR = ROOT / "checkpoints/drama-3"
LOG_FILE = ROOT / "outputs/hf_upload_pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


README_8BIT = """---
license: other
license_name: fish-audio-community-license
license_link: https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE
language:
- zh
- en
- ja
- ko
- es
- fr
- de
- it
- ru
- pt
pipeline_tag: text-to-speech
tags:
- mlx
- audio
- text-to-speech
- tts
- voice-cloning
- fish-audio
- drama-3
- moe
- 8bit
- apple-silicon
---

# Drama-3-MLX-8bit

This is the pure [Apple Silicon MLX](https://github.com/ml-explore/mlx) 8-bit affine quantized release of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3).

Drama 3 is Fish Audio's next-generation 35B parameter (3B active) Mixture-of-Experts (MoE) conversational text-to-speech model with hybrid linear/full attention and a 10-codebook Codebook-MTP audio decoder.

This 8-bit MLX release quantizes the 10,240 MoE experts using affine 8-bit quantization (`group_size=64`, `bits=8`), balancing generation speed, memory bandwidth efficiency, and studio-grade fidelity on Apple Silicon systems with 36GB+ unified memory.

## Performance on Apple Silicon (M3 Max 128GB)

| Precision | Model Size | Speed (80 frames / 3.72s audio) | Step Latency | RTF | Whisper ASR Accuracy |
|---|---|---|---|---|---|
| 4-bit | 23.2 GB | 18.15 s | 226.9 ms/step | 4.886 | 100.0% |
| **8-bit (this model)** | **32.6 GB** | **22.36 s** | **279.5 ms/step** | **6.018** | **100.0%** |
| BF16 | 69.0 GB | 34.36 s | 429.5 ms/step | 9.248 | 100.0% |

*The 8-bit model runs **1.54x faster than BF16** on Apple Silicon unified memory due to reduced memory bandwidth traffic.*

## Features

- **Balanced Fidelity & Speed**: Affine 8-bit MoE experts with full-precision DAC neural codec.
- **Zero-Shot Voice Cloning**: High-fidelity cloning with reference audio and transcript.
- **Multi-Speaker Characters**: Supports tags `<|speaker:0|>` to `<|speaker:15|>` (up to 16 distinct characters).
- **Point Tags & Emotion**: In-text cues like `[sighs]`, `[laughs]`, `[whispers]`.
- **44.1kHz High-Fidelity Audio**: 10-codebook Modded DAC neural audio codec.

## Quick Start

### Installation

```bash
git clone https://github.com/vanch007/drama-3-mlx.git
cd drama-3-mlx
pip install -e .
```

### Download and Run with CLI

```bash
hf download vanch007/Drama-3-MLX-8bit --local-dir models/Drama-3-MLX-8bit

drama3-mlx tts \
  --model models/Drama-3-MLX-8bit \
  --text "你好，世界！这是 Fish Audio Drama 3 的 8-bit MLX 版本。" \
  --output outputs/sample.wav
```

### Python API

```python
from drama3_mlx.model import Drama3Model
import soundfile as sf

model = Drama3Model.from_pretrained("models/Drama-3-MLX-8bit")

audio = model.tts(
    "Get the trust fund to the bank early, before the storm arrives tonight.",
    ref_audio="ref.wav",
    ref_text="Exact transcript of the reference audio clip."
)
sf.write("cloned_8bit.wav", audio, 44100)
```
"""


def upload_model(folder: Path, repo_id: str) -> None:
    log(f"Starting upload of {folder} to {repo_id}...")
    API.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
    )
    log(f"Successfully uploaded {repo_id}!")


def main():
    log("==========================================")
    log("Starting Drama 3 MLX Hugging Face Pipeline")
    log("==========================================")

    # 1. Wait for 4-bit upload if running
    log("Checking 4-bit upload status...")
    dir_4bit = ROOT / "models/drama-3-mlx-4bit"
    if dir_4bit.is_dir():
        upload_model(dir_4bit, "vanch007/Drama-3-MLX-4bit")

    # 2. Upload BF16 model
    dir_bf16 = ROOT / "models/drama-3-mlx-bf16"
    if dir_bf16.is_dir():
        upload_model(dir_bf16, "vanch007/Drama-3-MLX-BF16")
        log(f"Removing local BF16 directory {dir_bf16} to reclaim 69GB disk space...")
        shutil.rmtree(str(dir_bf16), ignore_errors=True)
        log("Reclaimed 69GB disk space!")

    # 3. Generate 8-bit model and upload
    dir_8bit = ROOT / "models/drama-3-mlx-8bit"
    if not dir_8bit.is_dir():
        log("Converting 8-bit model from raw checkpoints...")
        from drama3_mlx.convert import convert_drama3_model
        convert_drama3_model(CHECKPOINT_DIR, dir_8bit, quantize="8bit", dtype="bfloat16")
        log("8-bit model conversion complete!")

    with open(dir_8bit / "README.md", "w", encoding="utf-8") as f:
        f.write(README_8BIT)
    log("Saved README.md for 8-bit model.")

    upload_model(dir_8bit, "vanch007/Drama-3-MLX-8bit")

    log("==========================================================")
    log("ALL 3 MODELS SUCCESSFULLY CONVERTED AND UPLOADED TO HF!")
    log("  1. https://huggingface.co/vanch007/Drama-3-MLX-4bit")
    log("  2. https://huggingface.co/vanch007/Drama-3-MLX-BF16")
    log("  3. https://huggingface.co/vanch007/Drama-3-MLX-8bit")
    log("==========================================================")


if __name__ == "__main__":
    main()
