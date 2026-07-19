"""Pack a small set of Qwen AgentWorld tensors according to a rate policy.

This is a bounded smoke test for the real packer path. It supports whole-tensor
choices and virtual MoE expert-slice choices such as
``tensor.name::expert_042``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant import pack_bits, pack_trits_40, quantize_binary, quantize_ternary_mse

TERNARY_RATES = {"ternary40_log8", "ternary40_fp16"}
BINARY_RATES = {"binary_log8", "binary_fp16"}


def load_tensor(model_dir: Path, shard: str, name: str) -> torch.Tensor:
    with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def relative_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    err = (reconstructed.float() - original.float()).square().mean().sqrt()
    denom = original.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def write_npz(path: Path, **arrays: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def slice_for_choice(weight: torch.Tensor, choice: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    meta = choice.get("meta") or {}
    if "expert" not in meta:
        return weight, {}
    expert_idx = int(meta["expert"])
    if weight.ndim < 1 or expert_idx < 0 or expert_idx >= weight.shape[0]:
        raise IndexError(f"expert index {expert_idx} out of bounds for shape {tuple(weight.shape)}")
    return weight[expert_idx].contiguous(), {
        "parent_shape": list(weight.shape),
        "expert": expert_idx,
        "route_fraction": meta.get("route_fraction"),
    }


def pack_choice(
    *,
    model_dir: Path,
    tensor_summary: dict[str, Any],
    choice: dict[str, Any],
    out_dir: Path,
    group_size: int,
) -> dict[str, Any]:
    meta = choice.get("meta") or {}
    tensor_name = meta["tensor"]
    tensor_info = tensor_summary[tensor_name]
    parent_weight = load_tensor(model_dir, tensor_info["shard"], tensor_name)
    weight, slice_meta = slice_for_choice(parent_weight, choice)
    safe_name = choice["group"].replace(".", "__").replace("/", "_").replace("::", "__")
    rate = choice["rate"]
    source_bytes = int(tensor_info["bytes"])
    if slice_meta:
        source_bytes = int(source_bytes / int(slice_meta["parent_shape"][0]))
    result = {
        "group": choice["group"],
        "tensor": tensor_name,
        "rate": rate,
        "shape": list(weight.shape),
        "params": int(weight.numel()),
        "source_dtype": str(weight.dtype),
        "source_bytes": source_bytes,
        **slice_meta,
    }

    if rate in TERNARY_RATES:
        quant = quantize_ternary_mse(weight, group_size=group_size)
        packed, shape, pad = pack_trits_40(quant.codes.cpu().numpy())
        scales = quant.scales.cpu().numpy().astype(np.float16)
        packed_path = out_dir / f"{safe_name}.ternary40.npz"
        file_bytes = write_npz(
            packed_path,
            packed=packed,
            scales=scales,
            shape=np.array(shape, dtype=np.int64),
            pad=np.array([pad], dtype=np.int64),
            group_size=np.array([group_size], dtype=np.int64),
        )
        result.update(
            {
                "artifact": str(packed_path),
                "artifact_bytes": file_bytes,
                "scale_count": int(scales.size),
                "relative_rmse": relative_rmse(weight, quant.dequantize()),
            }
        )
        return result

    if rate in BINARY_RATES:
        quant = quantize_binary(weight, group_size=group_size)
        packed, shape, pad = pack_bits(quant.codes.cpu().numpy())
        scales = quant.scales.cpu().numpy().astype(np.float16)
        packed_path = out_dir / f"{safe_name}.binary.npz"
        file_bytes = write_npz(
            packed_path,
            packed=packed,
            scales=scales,
            shape=np.array(shape, dtype=np.int64),
            pad=np.array([pad], dtype=np.int64),
            group_size=np.array([group_size], dtype=np.int64),
        )
        result.update(
            {
                "artifact": str(packed_path),
                "artifact_bytes": file_bytes,
                "scale_count": int(scales.size),
                "relative_rmse": relative_rmse(weight, quant.dequantize()),
            }
        )
        return result

    result["skipped_reason"] = "rate is not a packed low-bit whole-tensor rate"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    parser.add_argument("--policy", default="/workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json", type=Path)
    parser.add_argument("--out-dir", default="/workspace/agentworld_runs/pack_smoke_qwen_agentworld", type=Path)
    parser.add_argument("--max-tensors", type=int, default=4)
    parser.add_argument("--max-params", type=int, default=12_000_000)
    parser.add_argument("--include-experts", action="store_true")
    parser.add_argument(
        "--rate",
        action="append",
        choices=sorted(TERNARY_RATES | BINARY_RATES),
        help="optional packed rate filter; repeatable",
    )
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    policy = json.loads(args.policy.read_text())
    tensor_summary = inventory["tensor_summary"]
    group_size = int(policy.get("group_size", 112))

    selected = []
    allowed_rates = set(args.rate or sorted(TERNARY_RATES | BINARY_RATES))
    for choice in policy["choices"]:
        meta = choice.get("meta") or {}
        tensor_name = meta.get("tensor")
        is_expert = "::expert_" in choice["group"]
        if not tensor_name:
            continue
        if is_expert and not args.include_experts:
            continue
        if choice["rate"] not in allowed_rates:
            continue
        if int(meta.get("params", 0)) > args.max_params:
            continue
        selected.append(choice)
        if len(selected) >= args.max_tensors:
            break

    packed_rows = []
    for choice in selected:
        row = pack_choice(
            model_dir=args.model_dir,
            tensor_summary=tensor_summary,
            choice=choice,
            out_dir=args.out_dir,
            group_size=group_size,
        )
        packed_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    by_rate: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "artifact_bytes": 0, "source_bytes": 0})
    for row in packed_rows:
        summary = by_rate[row["rate"]]
        summary["count"] += 1
        summary["artifact_bytes"] += int(row.get("artifact_bytes", 0))
        summary["source_bytes"] += int(row.get("source_bytes", 0))

    manifest = {
        "model_dir": str(args.model_dir),
        "inventory": str(args.inventory),
        "policy": str(args.policy),
        "group_size": group_size,
        "max_tensors": args.max_tensors,
        "max_params": args.max_params,
        "include_experts": args.include_experts,
        "rates": sorted(allowed_rates),
        "packed": packed_rows,
        "summary": {k: dict(v) for k, v in sorted(by_rate.items())},
        "note": "bounded smoke artifact; full policy packing must stream tensors and group expert slices",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "packed_tensors": len(packed_rows)}, indent=2))


if __name__ == "__main__":
    main()
