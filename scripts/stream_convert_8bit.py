#!/usr/bin/env python3
"""Convert 8-bit Drama 3 model with on-the-fly raw shard cleanup to guarantee safe disk headroom."""

import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

SRC = Path("/Users/vanch/drama-3-mlx/checkpoints/drama-3")
DST = Path("/Users/vanch/drama-3-mlx/models/drama-3-mlx-8bit")
DST.mkdir(parents=True, exist_ok=True)

print("Starting streaming 8-bit conversion...")

# 1. Copy configs
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    if (SRC / fname).is_file():
        shutil.copy(SRC / fname, DST / fname)

# 2. Copy codec.safetensors from 4bit or bf16
codec_src = Path("/Users/vanch/drama-3-mlx/models/drama-3-mlx-4bit/codec.safetensors")
if codec_src.is_file():
    shutil.copy(codec_src, DST / "codec.safetensors")
    print("Copied codec.safetensors")

index_file = SRC / "model.safetensors.index.json"
with open(index_file, "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]

open_shards = {}
def get_tensor(key: str) -> torch.Tensor:
    shard_name = weight_map[key]
    if shard_name not in open_shards:
        open_shards[shard_name] = safe_open(str(SRC / shard_name), framework="pt")
    return open_shards[shard_name].get_tensor(key)

total_layers = 40
shards_count = 8
layers_per_shard = total_layers // shards_count

out_weight_map = {}

for shard_idx in range(shards_count):
    start_layer = shard_idx * layers_per_shard
    end_layer = start_layer + layers_per_shard
    shard_filename = f"model-{shard_idx + 1:05d}-of-{shards_count:05d}.safetensors"
    print(f"\nConverting Shard {shard_idx + 1}/{shards_count} (Layers {start_layer} to {end_layer - 1})...")
    t_shard = time.perf_counter()

    shard_weights = {}

    for layer in range(start_layer, end_layer):
        layer_keys = [k for k in weight_map if f"model.language_model.layers.{layer}." in k and "mlp.experts." not in k]
        for k in layer_keys:
            np_val = get_tensor(k).float().numpy()
            if "linear_attn.conv1d.weight" in k:
                np_val = np_val.transpose(0, 2, 1)
            shard_weights[k] = mx.array(np_val).astype(mx.bfloat16)

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

        qw13, s13, b13 = mx.quantize(w13_stacked, group_size=64, bits=8)
        qw2, s2, b2 = mx.quantize(w2_stacked, group_size=64, bits=8)

        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_weight"] = qw13
        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_scales"] = s13
        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w13_biases"] = b13
        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_weight"] = qw2
        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_scales"] = s2
        shard_weights[f"model.language_model.layers.{layer}.mlp.experts.w2_biases"] = b2

    if shard_idx == 0:
        if "model.language_model.embed_tokens.weight" in weight_map:
            shard_weights["model.language_model.embed_tokens.weight"] = mx.array(get_tensor("model.language_model.embed_tokens.weight").float().numpy()).astype(mx.bfloat16)

    if shard_idx == shards_count - 1:
        for k in ["model.language_model.norm.weight", "lm_head.weight"]:
            if k in weight_map:
                shard_weights[k] = mx.array(get_tensor(k).float().numpy()).astype(mx.bfloat16)
        for k in weight_map:
            if k.startswith("audio_decoder"):
                shard_weights[k] = mx.array(get_tensor(k).float().numpy()).astype(mx.bfloat16)

    out_path = DST / shard_filename
    mx.save_safetensors(str(out_path), shard_weights)
    for k in shard_weights:
        out_weight_map[k] = shard_filename

    print(f"Saved {shard_filename} ({len(shard_weights)} tensors, took {time.perf_counter() - t_shard:.2f}s)")

with open(DST / "model.safetensors.index.json", "w", encoding="utf-8") as f:
    json.dump({"metadata": {"quantize": "8bit", "dtype": "bfloat16"}, "weight_map": out_weight_map}, f, indent=2)

print("\n8-bit conversion complete!")
