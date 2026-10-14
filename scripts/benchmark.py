#!/usr/bin/env python3
"""Benchmark Drama 3 MLX generation speed and RTF on Apple Silicon."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
from drama3_mlx.model import Drama3Model

BENCHMARK_SENTENCES = [
    "你好，欢迎使用 Apple Silicon 本地纯 MLX 版本的 Fish Audio Drama 3 语音合成系统。",
    "Get the trust fund to the bank early, before the storm arrives tonight.",
    "[sighs] It has been a very productive day. [laughs] Let\'s see what happens next.",
]


def main():
    parser = argparse.ArgumentParser(description="Drama 3 MLX Benchmark")
    parser.add_argument("--model", "-m", default="models/drama-3-mlx", help="MLX model path")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading Drama 3 MLX model from {args.model}...")
    model = Drama3Model.from_pretrained(args.model)

    for s_idx, text in enumerate(BENCHMARK_SENTENCES, 1):
        print(f"\nBenchmarking Sentence {s_idx}: {text[:50]}...")
        # Warmup
        for _ in range(args.warmup):
            _ = model.tts(text)

        durations = []
        elapsed_times = []
        for r in range(args.runs):
            mx.synchronize()
            t0 = time.perf_counter()
            audio = model.tts(text)
            mx.synchronize()
            elapsed = time.perf_counter() - t0
            duration = len(audio) / model.config.codec_sample_rate

            durations.append(duration)
            elapsed_times.append(elapsed)
            print(f"  Run {r + 1}: audio={duration:.2f}s, elapsed={elapsed:.2f}s, RTF={elapsed / duration:.3f}")

        mean_duration = sum(durations) / len(durations)
        mean_elapsed = sum(elapsed_times) / len(elapsed_times)
        mean_rtf = mean_elapsed / mean_duration
        print(f"  --> Average Duration: {mean_duration:.2f}s | Average RTF: {mean_rtf:.3f}")


if __name__ == "__main__":
    main()
