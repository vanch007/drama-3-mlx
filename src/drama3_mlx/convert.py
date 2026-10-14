from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from drama3_mlx.codec import build_ae


def convert_codec_weights(codec_pth_path: Path, output_safetensors_path: Path) -> None:
    """Convert PyTorch DAC codec.pth to MLX codec.safetensors."""
    state_dict = torch.load(codec_pth_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    dac = build_ae()
    weights_np = {k: v.float().cpu().numpy() for k, v in state_dict.items()}
    sanitized = dac.sanitize(weights_np)
    mlx_weights = {k: mx.array(v) for k, v in sanitized.items()}

    mx.save_safetensors(str(output_safetensors_path), mlx_weights)
    print(f"Saved converted codec weights to {output_safetensors_path}")


def convert_drama3_model(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "bfloat16",
    quantize: str = "8bit",
) -> None:
    """Convert HuggingFace Drama 3 checkpoint to pure MLX format with optional 8-bit quantization."""
    src = Path(checkpoint_dir).expanduser()
    dst = Path(output_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Starting conversion from {src} -> {dst} (quantize={quantize}, dtype={dtype})...")

    # 1. Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (src / fname).is_file():
            shutil.copy(src / fname, dst / fname)

    # 2. Convert Codec
    codec_pth = src / "codec.pth"
    codec_dst = dst / "codec.safetensors"
    if codec_pth.is_file() and not codec_dst.is_file():
        convert_codec_weights(codec_pth, codec_dst)

    # 3. Read index
    index_file = src / "model.safetensors.index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"Missing {index_file}")

    with open(index_file, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    open_shards: Dict[str, Any] = {}

    def get_tensor(key: str) -> torch.Tensor:
        shard_name = weight_map[key]
        if shard_name not in open_shards:
            open_shards[shard_name] = safe_open(str(src / shard_name), framework="pt")
        return open_shards[shard_name].get_tensor(key)

    total_layers = 40
    shards_count = 8
    layers_per_shard = total_layers // shards_count  # 5 layers per shard

    out_weight_map = {}

    # Convert layer by layer into 8 shards
    for shard_idx in range(shards_count):
        start_layer = shard_idx * layers_per_shard
        end_layer = start_layer + layers_per_shard
        shard_filename = f"model-{shard_idx + 1:05d}-of-{shards_count:05d}.safetensors"
        print(f"\nProcessing Shard {shard_idx + 1}/{shards_count} (Layers {start_layer} to {end_layer - 1})...")
        t_shard = time.perf_counter()

        shard_weights: Dict[str, mx.array] = {}

        # 1. Backbone non-expert weights for these layers
        for layer in range(start_layer, end_layer):
            layer_keys = [k for k in weight_map if f"model.language_model.layers.{layer}." in k and "mlp.experts." not in k]
            for k in layer_keys:
                tensor = get_tensor(k)
                np_val = tensor.float().numpy()
                if "linear_attn.conv1d.weight" in k:
                    # [C, 1, K] -> [C, K, 1]
                    np_val = np_val.transpose(0, 2, 1)

                mlx_val = mx.array(np_val).astype(mx.bfloat16 if dtype == "bfloat16" else mx.float16)
                shard_weights[k] = mlx_val

            # 2. Dequantize and stack MoE experts for this layer
            w13_list = []
            w2_list = []
            for e in range(256):
                gw = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.gate_proj.weight")
                gs = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.gate_proj.weight_scale_inv")
                g_deq = (gw.view(4, 128, 16, 128).to(torch.float32) * gs[:, None, :, None]).view(512, 2048)

                uw = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.up_proj.weight")
                us = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.up_proj.weight_scale_inv")
                u_deq = (uw.view(4, 128, 16, 128).to(torch.float32) * us[:, None, :, None]).view(512, 2048)

                dw = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.down_proj.weight")
                ds = get_tensor(f"model.language_model.layers.{layer}.mlp.experts.{e}.down_proj.weight_scale_inv")
                d_deq = (dw.view(16, 128, 4, 128).to(torch.float32) * ds[:, None, :, None]).view(2048, 512)

                w13_list.append(torch.cat([g_deq, u_deq], dim=0).float().numpy())
                w2_list.append(d_deq.float().numpy())

            w13_stacked = mx.array(np.stack(w13_list, axis=0)).astype(mx.bfloat16)
            w2_stacked = mx.array(np.stack(w2_list, axis=0)).astype(mx.bfloat16)

            if quantize == "8bit":
                qw13, s13, b13 = mx.quantize(w13_stacked, group_size=64, bits=8)
                qw2, s2, b2 = mx.quantize(w2_stacked, group_size=64, bits=8)
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_weight"] = qw13
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_scales"] = s13
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_biases"] = b13
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_weight"] = qw2
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_scales"] = s2
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_biases"] = b2
            else:
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_weight"] = w13_stacked
                shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_weight"] = w2_stacked

        # In the first shard, also include embeddings
        if shard_idx == 0:
            for k in ["model.language_model.embed_tokens.weight"]:
                if k in weight_map:
                    shard_weights[k] = mx.array(get_tensor(k).float().numpy()).astype(mx.bfloat16)

        # In the last shard, include final norm, lm_head, and audio decoder tower
        if shard_idx == shards_count - 1:
            for k in ["model.language_model.norm.weight", "lm_head.weight"]:
                if k in weight_map:
                    shard_weights[k] = mx.array(get_tensor(k).float().numpy()).astype(mx.bfloat16)
            for k in weight_map:
                if k.startswith("audio_decoder"):
                    shard_weights[k] = mx.array(get_tensor(k).float().numpy()).astype(mx.bfloat16)

        out_path = dst / shard_filename
        mx.save_safetensors(str(out_path), shard_weights)
        for k in shard_weights:
            out_weight_map[k] = shard_filename

        print(f"Saved {shard_filename} ({len(shard_weights)} tensors, took {time.perf_counter() - t_shard:.2f}s)")

    # Save final index
    with open(dst / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"quantize": quantize, "dtype": dtype}, "weight_map": out_weight_map}, f, indent=2)

    print(f"\nAll {shards_count} shards successfully converted to {dst}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Fish Audio Drama 3 checkpoint to pure MLX format")
    parser.add_argument("--checkpoint", "-c", default="/Users/vanch/drama-3-mlx/checkpoints/drama-3", help="Input directory")
    parser.add_argument("--output", "-o", default="/Users/vanch/drama-3-mlx/models/drama-3-mlx", help="Output directory")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--quantize", default="8bit", choices=["8bit", "none"])
    args = parser.parse_args()

    convert_drama3_model(args.checkpoint, args.output, dtype=args.dtype, quantize=args.quantize)


if __name__ == "__main__":
    main()
