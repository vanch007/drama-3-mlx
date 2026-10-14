from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

from drama3_mlx.codec import DAC, build_ae
from drama3_mlx.config import Drama3Config


def dequantize_fp8_block(weight: np.ndarray, scale_inv: np.ndarray) -> np.ndarray:
    """Block-FP8 dequantizer in numpy."""
    rows, cols = weight.shape
    row_blocks = rows // 128
    col_blocks = cols // 128
    blocks = weight.reshape(row_blocks, 128, col_blocks, 128).astype(np.float32)
    scaled = blocks * scale_inv[:, None, :, None]
    return scaled.reshape(rows, cols)


def convert_codec_weights(codec_pth_path: Path, output_safetensors_path: Path) -> None:
    """Convert PyTorch DAC codec.pth to MLX codec.safetensors."""
    import torch

    state_dict = torch.load(codec_pth_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    dac = build_ae()
    weights_np = {k: v.cpu().numpy() for k, v in state_dict.items()}
    mlx_weights = dac.sanitize(weights_np)

    mx.save_safetensors(str(output_safetensors_path), mlx_weights)
    print(f"Saved converted codec weights to {output_safetensors_path}")


def convert_drama3_model(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "bfloat16",
    quantize: Optional[str] = None,
) -> None:
    """Convert HuggingFace Drama 3 checkpoint to pure MLX format."""
    src = Path(checkpoint_dir).expanduser()
    dst = Path(output_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (src / fname).is_file():
            shutil.copy(src / fname, dst / fname)

    # 2. Convert Codec
    codec_pth = src / "codec.pth"
    if codec_pth.is_file():
        convert_codec_weights(codec_pth, dst / "codec.safetensors")

    # 3. Read index or find safetensors
    from safetensors import safe_open

    safetensors_files = sorted(src.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files in {src}")

    target_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float16
    print(f"Converting Drama 3 weights to {dtype} in {dst}...")

    weights_to_save: Dict[str, mx.array] = {}
    experts_w13_dict: Dict[Tuple[int, int], np.ndarray] = {}
    experts_w2_dict: Dict[Tuple[int, int], np.ndarray] = {}

    for sf_path in safetensors_files:
        print(f"Processing {sf_path.name}...")
        with safe_open(str(sf_path), framework="pt") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)
                np_val = tensor.cpu().float().numpy()

                # Handle conv1d transpose for linear attention
                if "linear_attn.conv1d.weight" in k:
                    # PyTorch Conv1d: [C, 1, K] -> MLX Conv1d: [C, K, 1]
                    np_val = np_val.transpose(0, 2, 1)

                # Check if expert weight
                # format: model.language_model.layers.{L}.mlp.experts.{E}.{proj}.weight
                expert_match = re.search(r"layers.(d+).mlp.experts.(d+).(gate_proj|up_proj|down_proj).weight", k)
                if expert_match:
                    layer_idx = int(expert_match.group(1))
                    expert_idx = int(expert_match.group(2))
                    proj = expert_match.group(3)

                    if proj == "down_proj":
                        experts_w2_dict[(layer_idx, expert_idx)] = np_val
                    elif proj == "gate_proj":
                        gate_up = experts_w13_dict.get((layer_idx, expert_idx))
                        if gate_up is None:
                            gate_up = [np_val, None]
                            experts_w13_dict[(layer_idx, expert_idx)] = gate_up
                        else:
                            gate_up[0] = np_val
                    elif proj == "up_proj":
                        gate_up = experts_w13_dict.get((layer_idx, expert_idx))
                        if gate_up is None:
                            gate_up = [None, np_val]
                            experts_w13_dict[(layer_idx, expert_idx)] = gate_up
                        else:
                            gate_up[1] = np_val
                    continue

                weights_to_save[k] = mx.array(np_val).astype(target_dtype)

    # Assemble stacked expert weights
    layers_with_experts = set(l for l, _ in experts_w2_dict.keys())
    for layer_idx in sorted(layers_with_experts):
        experts_in_layer = sorted([e for l, e in experts_w2_dict.keys() if l == layer_idx])
        num_exp = len(experts_in_layer)

        w13_list = []
        w2_list = []
        for e in range(num_exp):
            gate, up = experts_w13_dict[(layer_idx, e)]
            w13_list.append(np.concatenate([gate, up], axis=0))
            w2_list.append(experts_w2_dict[(layer_idx, e)])

        w13_stacked = np.stack(w13_list, axis=0)
        w2_stacked = np.stack(w2_list, axis=0)

        weights_to_save[f"model.language_model.layers.{layer_idx}.mlp.experts.w13_weight"] = mx.array(w13_stacked).astype(target_dtype)
        weights_to_save[f"model.language_model.layers.{layer_idx}.mlp.experts.w2_weight"] = mx.array(w2_stacked).astype(target_dtype)

    out_file = dst / "model.safetensors"
    mx.save_safetensors(str(out_file), weights_to_save)
    print(f"Successfully converted Drama 3 checkpoint to {out_file} ({len(weights_to_save)} tensors).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Fish Audio Drama 3 checkpoint to MLX format")
    parser.add_argument("--checkpoint", "-c", required=True, help="Input directory containing official drama-3 safetensors and codec.pth")
    parser.add_argument("--output", "-o", required=True, help="Output directory for MLX checkpoint")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    convert_drama3_model(args.checkpoint, args.output, dtype=args.dtype)


if __name__ == "__main__":
    main()
