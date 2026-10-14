#!/usr/bin/env python3
"""Run demo synthesis with Drama 3 MLX."""

import argparse
import time
from pathlib import Path

import soundfile as sf
from drama3_mlx.model import Drama3Model

DEFAULT_PROMPTS = [
    "[sighs] Fine. If we miss it, it\'s on you. [laughs] Well, maybe that\'s fair.",
    "<|speaker:0|> 你好，欢迎使用 Drama 3 纯 MLX 版本！<|speaker:1|> 效果听起来真棒，完全在本地 Apple Silicon 实时运行。",
    "The quarterly report is due on Thursday, so please finish it by noon.",
]


def main():
    parser = argparse.ArgumentParser(description="Drama 3 MLX Demonstration")
    parser.add_argument("--model", "-m", default="models/drama-3-mlx", help="MLX checkpoint path")
    parser.add_argument("--out-dir", "-o", default="outputs/demo", help="Output directory")
    parser.add_argument("--ref-audio", "-r", help="Optional reference audio for cloning")
    parser.add_argument("--ref-text", help="Exact transcript of the reference audio")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Drama 3 MLX from {args.model}...")
    model = Drama3Model.from_pretrained(args.model)

    for i, text in enumerate(DEFAULT_PROMPTS, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Input: {text}")
        t0 = time.perf_counter()
        audio = model.tts(text, ref_audio=args.ref_audio, ref_text=args.ref_text)
        duration = len(audio) / model.config.codec_sample_rate
        elapsed = time.perf_counter() - t0
        rtf = elapsed / max(duration, 0.001)

        out_path = out_dir / f"sample_{i}.wav"
        sf.write(str(out_path), audio, model.config.codec_sample_rate)
        print(f"Saved: {out_path} ({duration:.2f}s, RTF: {rtf:.3f})")


if __name__ == "__main__":
    main()
