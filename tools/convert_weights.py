#!/usr/bin/env python3
"""
Convert HuggingFace Qwen 3.5 MoE safetensors to ingot mmap-friendly format.

Input:  HuggingFace model directory with safetensors shards
Output: ingot model directory with:
  - model_weights.bin    — shared weights (embedding, layernorms, attention, shared expert)
  - packed_experts/      — one file per layer, experts split for individual mmap
  - weight_index.json    — maps weight names to byte offsets in model_weights.bin
  - config.json          — copied from input
  - vocab.json           — copied from input
  - merges.txt           — copied from input (or extracted from tokenizer.json)
  - added_tokens.json    — copied from input

Weight format:
  All quantized weights are stored as-is (4-bit packed in U32 with BF16 scales/biases).
  The converter preserves the quantization — no dequantization happens here.
  Each weight is written contiguously: weight data, then scales, then biases.
  Alignment: each weight starts at a 4096-byte boundary (page-aligned for mmap).

Expert format (packed_experts/layer_XX.bin):
  For each expert i (0..num_experts-1):
    gate_proj.weight  [moe_intermediate, hidden/8]  (packed 4-bit)
    gate_proj.scales  [moe_intermediate, hidden/group_size]  (BF16)
    gate_proj.biases  [moe_intermediate, hidden/group_size]  (BF16)
    up_proj.weight    [moe_intermediate, hidden/8]
    up_proj.scales    [moe_intermediate, hidden/group_size]
    up_proj.biases    [moe_intermediate, hidden/group_size]
    down_proj.weight  [hidden, moe_intermediate/8]
    down_proj.scales  [hidden, moe_intermediate/group_size]
    down_proj.biases  [hidden, moe_intermediate/group_size]
  Expert offsets are regular — each expert occupies the same number of bytes,
  so expert_offset = expert_index * expert_stride.
"""

import argparse
import json
import os
import struct
import shutil
import sys
import numpy as np
from pathlib import Path


def read_safetensors_header(path):
    """Read safetensors file header (JSON metadata about tensors)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header, 8 + header_size  # return header and data offset


def read_tensor_raw(shard_path, data_offset, tensor_meta):
    """Read raw tensor bytes from a safetensors shard."""
    start, end = tensor_meta["data_offsets"]
    with open(shard_path, "rb") as f:
        f.seek(data_offset + start)
        return f.read(end - start)


def align_to(offset, alignment=4096):
    """Round up to alignment boundary."""
    return (offset + alignment - 1) & ~(alignment - 1)


def write_padded(f, data, alignment=4096):
    """Write data and pad to alignment boundary."""
    f.write(data)
    pad = align_to(len(data), alignment) - len(data)
    if pad > 0:
        f.write(b"\x00" * pad)
    return align_to(len(data), alignment)


class WeightConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.index = {}  # weight_name -> {offset, size, dtype, shape}

        # Load safetensors index
        idx_path = self.input_dir / "model.safetensors.index.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"No model.safetensors.index.json in {input_dir}")

        with open(idx_path) as f:
            self.sf_index = json.load(f)

        self.weight_map = self.sf_index["weight_map"]

        # Cache shard headers
        self.shard_headers = {}
        self.shard_data_offsets = {}

    def get_shard_header(self, shard_name):
        if shard_name not in self.shard_headers:
            path = self.input_dir / shard_name
            header, data_offset = read_safetensors_header(path)
            self.shard_headers[shard_name] = header
            self.shard_data_offsets[shard_name] = data_offset
        return self.shard_headers[shard_name], self.shard_data_offsets[shard_name]

    def read_weight(self, name):
        """Read a weight tensor's raw bytes."""
        shard_name = self.weight_map[name]
        header, data_offset = self.get_shard_header(shard_name)
        meta = header[name]
        raw = read_tensor_raw(self.input_dir / shard_name, data_offset, meta)
        return raw, meta

    def convert(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "packed_experts").mkdir(exist_ok=True)

        # Copy tokenizer files
        for fname in ["config.json", "vocab.json", "merges.txt",
                      "added_tokens.json", "tokenizer_config.json",
                      "generation_config.json"]:
            src = self.input_dir / fname
            dst = self.output_dir / fname
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                print(f"  copied {fname}")

        # Classify weights
        all_names = sorted(self.weight_map.keys())
        shared_names = []  # go into model_weights.bin
        expert_layers = {}  # layer_idx -> list of expert weight names

        for name in all_names:
            if "switch_mlp" in name:
                # Extract layer index
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layers") + 1])
                if layer_idx not in expert_layers:
                    expert_layers[layer_idx] = []
                expert_layers[layer_idx].append(name)
            else:
                shared_names.append(name)

        print(f"\n  Shared weights: {len(shared_names)}")
        print(f"  Expert layers:  {len(expert_layers)}")

        # Write shared weights
        print("\n  Writing model_weights.bin...")
        self._write_shared(shared_names)

        # Write expert files
        print(f"\n  Writing {len(expert_layers)} expert layer files...")
        for layer_idx in sorted(expert_layers.keys()):
            self._write_expert_layer(layer_idx, expert_layers[layer_idx])

        # Write weight index
        index_path = self.output_dir / "weight_index.json"
        with open(index_path, "w") as f:
            json.dump(self.index, f, indent=2)
        print(f"\n  Wrote weight_index.json ({len(self.index)} entries)")

    def _write_shared(self, names):
        path = self.output_dir / "model_weights.bin"
        offset = 0
        total_mb = 0

        with open(path, "wb") as f:
            for i, name in enumerate(names):
                raw, meta = self.read_weight(name)

                # Page-align
                pad = align_to(offset) - offset
                if pad > 0:
                    f.write(b"\x00" * pad)
                    offset += pad

                f.write(raw)

                short = name.replace("language_model.model.", "")
                self.index[short] = {
                    "file": "model_weights.bin",
                    "offset": offset,
                    "size": len(raw),
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                }

                offset += len(raw)
                total_mb = offset / (1024 * 1024)

                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(names)} weights ({total_mb:.0f} MB)")

        print(f"    Done: {total_mb:.1f} MB total")

    def _write_expert_layer(self, layer_idx, names):
        """Write a single expert layer file with per-expert layout."""
        path = self.output_dir / "packed_experts" / f"layer_{layer_idx:02d}.bin"

        # Determine number of experts from the shape
        # switch_mlp weights have shape [num_experts, ...]
        sample_name = [n for n in names if n.endswith(".weight")][0]
        _, meta = self.read_weight(sample_name)
        num_experts = meta["shape"][0]

        # Group by projection type
        projections = {}  # "gate_proj.weight" -> (raw_bytes, meta)
        for name in names:
            # Extract: gate_proj.weight, gate_proj.scales, etc.
            parts = name.split("switch_mlp.")[1]
            projections[parts] = self.read_weight(name)

        # Write experts sequentially
        # For each expert, write: gate_proj(w,s,b), up_proj(w,s,b), down_proj(w,s,b)
        with open(path, "wb") as f:
            expert_stride = 0

            for expert_idx in range(num_experts):
                expert_start = f.tell()

                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    for suffix in ["weight", "scales", "biases"]:
                        key = f"{proj}.{suffix}"
                        raw, meta = projections[key]
                        shape = meta["shape"]

                        # Slice out this expert's data
                        # Shape is [num_experts, dim1, dim2]
                        expert_size = 1
                        for s in shape[1:]:
                            expert_size *= s

                        dtype_bytes = {
                            "U32": 4, "F32": 4, "F16": 2, "BF16": 2,
                            "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                        }
                        elem_size = dtype_bytes.get(meta["dtype"], 4)
                        chunk_size = expert_size * elem_size

                        start = expert_idx * chunk_size
                        end = start + chunk_size
                        f.write(raw[start:end])

                if expert_idx == 0:
                    expert_stride = f.tell() - expert_start

            total_mb = f.tell() / (1024 * 1024)

        # Record in index
        short = f"layers.{layer_idx}.experts"
        self.index[short] = {
            "file": f"packed_experts/layer_{layer_idx:02d}.bin",
            "num_experts": num_experts,
            "expert_stride": expert_stride,
        }

        print(f"    layer {layer_idx:2d}: {num_experts} experts, "
              f"{expert_stride} bytes/expert, {total_mb:.1f} MB total")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Qwen 3.5 MoE to ingot format")
    parser.add_argument("--input", required=True,
                       help="HuggingFace model directory")
    parser.add_argument("--output", required=True,
                       help="Output ingot model directory")
    args = parser.parse_args()

    print(f"Converting: {args.input} -> {args.output}")

    converter = WeightConverter(args.input, args.output)
    converter.convert()

    print("\nDone! Model ready for ingot.")


if __name__ == "__main__":
    main()
