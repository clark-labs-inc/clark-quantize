#!/usr/bin/env python3
"""Streamed Qwen3.8-27B ternary preflight for a bounded CPU or GPU device.

This intentionally does not instantiate the full model or allocate an FP32
master. It inventories the immutable checkpoint, produces honest theoretical
and aligned-2-bit ledgers, and samples representative matrices to compare the
Clark optimal-MSE ternary initializer with the BitNet/Prism-style absmean rule.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clark_quantize.ternary import ternarize


GROUPS = {
    "attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "linear_attention": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    ),
    "embedding": ("embed_tokens",),
    "head": ("lm_head",),
}
EXCLUDED = ("visual", "vision", "mtp.", "in_proj_a", "in_proj_b", "conv1d")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def category(name: str, shape: list[int]) -> str | None:
    if len(shape) != 2 or any(token in name for token in EXCLUDED):
        return None
    for label, suffixes in GROUPS.items():
        if any(name.endswith(suffix + ".weight") or name.endswith(suffix) for suffix in suffixes):
            return label
    return None


def tensor_shapes(model_dir: Path, weight_map: dict[str, str]) -> dict[str, list[int]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)
    shapes: dict[str, list[int]] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                shapes[name] = list(handle.get_slice(name).get_shape())
    return shapes


def pick_samples(candidates: dict[str, list[tuple[str, str, list[int]]]], count: int):
    chosen: list[tuple[str, str, str, list[int]]] = []
    for label, rows in sorted(candidates.items()):
        layered = []
        other = []
        for row in rows:
            match = LAYER_RE.search(row[0])
            (layered if match else other).append((int(match.group(1)) if match else -1, row))
        if layered:
            layered.sort(key=lambda item: item[0])
            positions = [round(i * (len(layered) - 1) / max(count - 1, 1)) for i in range(count)]
            for position in dict.fromkeys(positions):
                name, shard, shape = layered[position][1]
                chosen.append((label, name, shard, shape))
        else:
            for _, (name, shard, shape) in other[:count]:
                chosen.append((label, name, shard, shape))
    return chosen


def sampled_tensor(model_dir: Path, shard: str, name: str, rows: int) -> torch.Tensor:
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        shape = view.get_shape()
        return view[: min(rows, shape[0]), :].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--samples-per-kind", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    config = json.loads((args.model_dir / "config.json").read_text())
    receipt_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    download_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    weight_map: dict[str, str] = index["weight_map"]
    shapes = tensor_shapes(args.model_dir, weight_map)

    total_params = 0
    quant_params = 0
    scale_count = 0
    quant_by_category: Counter[str] = Counter()
    kept_by_reason: Counter[str] = Counter()
    candidates: dict[str, list[tuple[str, str, list[int]]]] = defaultdict(list)
    for name, shape in shapes.items():
        params = math.prod(shape)
        total_params += params
        label = category(name, shape)
        if label is None:
            reason = "excluded_high_precision" if any(token in name for token in EXCLUDED) else "non_matrix_or_unmapped"
            kept_by_reason[reason] += params
            continue
        quant_params += params
        groups_per_row = math.ceil(shape[1] / args.group)
        scale_count += shape[0] * groups_per_row
        quant_by_category[label] += params
        candidates[label].append((name, weight_map[name], shape))

    retained_params = total_params - quant_params
    entropy_bits = math.log2(3)
    theoretical_bits = quant_params * entropy_bits + scale_count * 16 + retained_params * 16
    aligned_bits = quant_params * 2 + scale_count * 16 + retained_params * 16

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    probes = []
    for label, name, shard, shape in pick_samples(candidates, args.samples_per_kind):
        source = sampled_tensor(args.model_dir, shard, name, args.sample_rows).to(device, dtype=torch.float32)
        row = {"category": label, "name": name, "shape": shape, "sample_shape": list(source.shape)}
        for method in ("absmean", "optimal"):
            synchronize()
            before = time.perf_counter()
            _, stats = ternarize(source, group=args.group, method=method)
            synchronize()
            row[method] = {**stats, "seconds": time.perf_counter() - before}
        probes.append(row)
        del source

    result = {
        "schema": "clark_qwen38_27b_preflight_v1",
        "model_dir": str(args.model_dir),
        "download_receipt": download_receipt,
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "group_size": args.group,
        "tensor_count": len(shapes),
        "parameter_ledger": {
            "total": total_params,
            "ternary_payload": quant_params,
            "retained_high_precision": retained_params,
            "fp16_scale_count": scale_count,
            "ternary_by_category": dict(quant_by_category),
            "retained_by_reason": dict(kept_by_reason),
            "effective_bits_per_total_parameter_base3": theoretical_bits / total_params,
            "effective_bits_per_total_parameter_aligned2": aligned_bits / total_params,
            "base3_payload_bytes": math.ceil(theoretical_bits / 8),
            "aligned2_payload_bytes": math.ceil(aligned_bits / 8),
        },
        "probes": probes,
        "compute": {
            "device": device_name,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
        "cuda": {
            "device": device_name if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
