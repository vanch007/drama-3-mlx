#!/usr/bin/env python3
"""Monitors download of Drama 3 and automatically converts to MLX once complete."""

import os
import sys
import time
from pathlib import Path

CHECKPOINT_DIR = Path("/Users/vanch/drama-3-mlx/checkpoints/drama-3")
OUTPUT_DIR = Path("/Users/vanch/drama-3-mlx/models/drama-3-mlx")
TOTAL_SHARDS = 8


def check_complete() -> bool:
    for i in range(1, TOTAL_SHARDS + 1):
        fname = f"model-{i:05d}-of-{TOTAL_SHARDS:05d}.safetensors"
        if not (CHECKPOINT_DIR / fname).is_file():
            return False
    return True


def main():
    print("Watcher started: waiting for Drama 3 safetensors download to finish...")
    while True:
        if check_complete():
            print("\nAll 8 safetensors shards detected! Starting conversion...")
            from drama3_mlx.convert import convert_drama3_model
            convert_drama3_model(CHECKPOINT_DIR, OUTPUT_DIR, dtype="bfloat16")
            print("Automatic conversion finished successfully!")
            break

        # Check incomplete progress
        cache_dir = CHECKPOINT_DIR / ".cache/huggingface/download"
        downloaded = 0
        if cache_dir.is_dir():
            for f in cache_dir.glob("*.incomplete"):
                downloaded += f.stat().st_size
        for f in CHECKPOINT_DIR.glob("*.safetensors"):
            downloaded += f.stat().st_size

        gb = downloaded / (1024**3)
        pct = (gb / 36.88) * 100
        print(f"\r[Progress] Downloaded {gb:.2f} GB / 36.88 GB ({pct:.1f}%)...", end="", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
