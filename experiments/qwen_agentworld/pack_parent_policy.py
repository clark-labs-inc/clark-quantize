"""Pack mixed-rate policy groups into parent-tensor artifacts.

The older smoke packer writes one artifact per policy group. This script groups
expert-slice choices by their parent tensor so a mixed BF16/binary/ternary MoE
tensor can be represented by one artifact plus a per-slice manifest.
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
from clark_air.quant.binary import BinaryQuantized
from clark_air.quant.optimal_ternary import TernaryQuantized


TERNARY_RATES = {"ternary40_log8", "ternary40_fp16"}
BINARY_RATES = {"binary_log8", "binary_fp16"}
PACKABLE_RATES = {"bf16", "fp8", *TERNARY_RATES, *BINARY_RATES}


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_bucket_group_sizes(values: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"group-size override must be BUCKET=SIZE, got {value!r}")
        bucket, size_text = value.split("=", 1)
        bucket = bucket.strip()
        if not bucket:
            raise ValueError(f"empty bucket in group-size override {value!r}")
        size = int(size_text)
        if size <= 0:
            raise ValueError(f"group size must be positive for {bucket!r}: {size}")
        overrides[bucket] = size
    return overrides


def choice_group_size(choice: dict[str, Any], default_group_size: int, bucket_group_sizes: dict[str, int]) -> int:
    meta = choice.get("meta") or {}
    explicit = choice.get("group_size", meta.get("group_size"))
    if explicit is not None:
        size = int(explicit)
    else:
        size = int(bucket_group_sizes.get(str(meta.get("bucket")), default_group_size))
    if size <= 0:
        raise ValueError(f"group size must be positive for {choice.get('group')}: {size}")
    return size


def load_tensor(model_dir: Path, shard: str, name: str) -> torch.Tensor:
    with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def load_weight_patch(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    rows: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            rows[key] = f.get_tensor(key)
    return rows


def patch_key_candidates(tensor_name: str) -> list[str]:
    candidates = [tensor_name]
    language_prefix = "model.language_model."
    model_prefix = "model."
    if tensor_name.startswith(language_prefix):
        candidates.append(model_prefix + tensor_name[len(language_prefix) :])
    elif tensor_name.startswith(model_prefix):
        candidates.append(language_prefix + tensor_name[len(model_prefix) :])
    return candidates


def apply_weight_patch(
    parent: torch.Tensor,
    tensor_name: str,
    weight_patch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, str | None]:
    for key in patch_key_candidates(tensor_name):
        patched = weight_patch.get(key)
        if patched is None:
            continue
        if tuple(patched.shape) != tuple(parent.shape):
            raise ValueError(
                f"patch tensor {key} has shape {tuple(patched.shape)}, "
                f"expected {tuple(parent.shape)} for {tensor_name}"
            )
        return patched.contiguous(), key
    return parent, None


def relative_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    err = (reconstructed.float() - original.float()).square().mean().sqrt()
    denom = original.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def safe_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "_").replace("::", "__")


def bf16_payload(weight: torch.Tensor) -> np.ndarray:
    return weight.contiguous().to(torch.bfloat16).view(torch.int16).cpu().numpy()


def bf16_dequantize(payload: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(payload.copy()).view(torch.bfloat16).reshape(shape).float()


def fp8_payload(weight: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(torch, "float8_e4m3fn"):
        fp8 = weight.contiguous().float().to(torch.float8_e4m3fn)
        return fp8.view(torch.uint8).cpu().numpy(), {"fp8_mode": "torch.float8_e4m3fn"}

    scale = weight.float().abs().amax().clamp_min(1e-8) / 127.0
    quant = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8)
    return quant.cpu().numpy(), {"fp8_mode": "int8_symmetric_fallback", "fp8_scale": float(scale.cpu())}


def fp8_dequantize(payload: np.ndarray, shape: tuple[int, ...], meta: dict[str, Any]) -> torch.Tensor:
    mode = meta.get("fp8_mode")
    if mode == "torch.float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required to verify this artifact")
        return torch.from_numpy(payload.copy()).view(torch.float8_e4m3fn).reshape(shape).float()
    if mode == "int8_symmetric_fallback":
        return torch.from_numpy(payload.copy()).float().reshape(shape) * float(meta["fp8_scale"])
    raise ValueError(f"unsupported fp8 mode: {mode}")


def choice_sort_key(choice: dict[str, Any]) -> tuple[int, str]:
    meta = choice.get("meta") or {}
    expert = meta.get("expert")
    return (int(expert) if expert is not None else -1, choice["group"])


def select_choices(
    policy: dict[str, Any],
    *,
    include_experts: bool,
    rates: set[str] | None,
    buckets: set[str] | None,
    max_params_per_group: int | None,
) -> dict[str, list[dict[str, Any]]]:
    by_tensor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for choice in policy["choices"]:
        meta = choice.get("meta") or {}
        tensor = meta.get("tensor")
        if not tensor:
            continue
        if choice["rate"] not in PACKABLE_RATES:
            continue
        if "::expert_" in choice["group"] and not include_experts:
            continue
        if rates is not None and choice["rate"] not in rates:
            continue
        if buckets is not None and meta.get("bucket") not in buckets:
            continue
        if max_params_per_group is not None and int(meta.get("params", 0)) > max_params_per_group:
            continue
        by_tensor[str(tensor)].append(choice)
    return {tensor: sorted(rows, key=choice_sort_key) for tensor, rows in sorted(by_tensor.items())}


def limit_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def slice_for_choice(parent: torch.Tensor, choice: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    meta = choice.get("meta") or {}
    if "expert" not in meta:
        return parent.contiguous(), {}
    expert = int(meta["expert"])
    if parent.ndim < 1 or expert < 0 or expert >= parent.shape[0]:
        raise IndexError(f"expert index {expert} out of bounds for shape {tuple(parent.shape)}")
    return parent[expert].contiguous(), {
        "expert": expert,
        "parent_shape": list(parent.shape),
        "route_fraction": meta.get("route_fraction"),
    }


def pack_group(
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
    weight: torch.Tensor,
    rate: str,
    group_size: int,
) -> dict[str, Any]:
    arrays[f"{prefix}_group_size"] = np.array([group_size], dtype=np.int64)
    row: dict[str, Any] = {
        "prefix": prefix,
        "rate": rate,
        "shape": list(weight.shape),
        "params": int(weight.numel()),
        "source_dtype": str(weight.dtype),
        "source_bytes": int(weight.numel() * weight.element_size()),
        "group_size": int(group_size),
    }

    if rate == "bf16":
        arrays[f"{prefix}_bf16"] = bf16_payload(weight)
        reconstructed = bf16_dequantize(arrays[f"{prefix}_bf16"], tuple(weight.shape))
        row["relative_rmse"] = relative_rmse(weight, reconstructed)
        return row

    if rate == "fp8":
        payload, fp8_meta = fp8_payload(weight)
        arrays[f"{prefix}_fp8"] = payload
        reconstructed = fp8_dequantize(arrays[f"{prefix}_fp8"], tuple(weight.shape), fp8_meta)
        row.update(fp8_meta)
        row["relative_rmse"] = relative_rmse(weight, reconstructed)
        return row

    if rate in TERNARY_RATES:
        quant = quantize_ternary_mse(weight, group_size=group_size)
        packed, shape, pad = pack_trits_40(quant.codes.cpu().numpy())
        stored_scales = quant.scales.to(torch.float16)
        arrays[f"{prefix}_packed"] = packed
        arrays[f"{prefix}_scales"] = stored_scales.cpu().numpy()
        arrays[f"{prefix}_shape"] = np.array(shape, dtype=np.int64)
        arrays[f"{prefix}_pad"] = np.array([pad], dtype=np.int64)
        row["scale_count"] = int(quant.scales.numel())
        row["scale_dtype"] = "fp16"
        row["relative_rmse"] = relative_rmse(
            weight,
            TernaryQuantized(
                codes=quant.codes,
                scales=stored_scales,
                original_shape=quant.original_shape,
                group_size=quant.group_size,
                pad=quant.pad,
            ).dequantize(),
        )
        return row

    if rate in BINARY_RATES:
        quant = quantize_binary(weight, group_size=group_size)
        packed, shape, pad = pack_bits(quant.codes.cpu().numpy())
        stored_scales = quant.scales.to(torch.float16)
        arrays[f"{prefix}_packed"] = packed
        arrays[f"{prefix}_scales"] = stored_scales.cpu().numpy()
        arrays[f"{prefix}_shape"] = np.array(shape, dtype=np.int64)
        arrays[f"{prefix}_pad"] = np.array([pad], dtype=np.int64)
        row["scale_count"] = int(quant.scales.numel())
        row["scale_dtype"] = "fp16"
        row["relative_rmse"] = relative_rmse(
            weight,
            BinaryQuantized(
                codes=quant.codes,
                scales=stored_scales,
                original_shape=quant.original_shape,
                group_size=quant.group_size,
                pad=quant.pad,
            ).dequantize(),
        )
        return row

    raise ValueError(f"unsupported rate: {rate}")


def pack_parent_tensor(
    *,
    model_dir: Path,
    tensor_summary: dict[str, Any],
    tensor_name: str,
    choices: list[dict[str, Any]],
    out_dir: Path,
    default_group_size: int,
    bucket_group_sizes: dict[str, int],
    max_groups_per_parent: int | None,
    weight_patch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    info = tensor_summary[tensor_name]
    parent = load_tensor(model_dir, info["shard"], tensor_name)
    parent, patch_key = apply_weight_patch(parent, tensor_name, weight_patch)
    original_choice_count = len(choices)
    if max_groups_per_parent is not None:
        choices = choices[:max_groups_per_parent]

    arrays: dict[str, np.ndarray] = {
        "group_size": np.array([default_group_size], dtype=np.int64),
    }
    groups = []
    source_bytes = 0
    for index, choice in enumerate(choices):
        weight, slice_meta = slice_for_choice(parent, choice)
        prefix = f"g{index:05d}"
        group_size = choice_group_size(choice, default_group_size, bucket_group_sizes)
        row = pack_group(
            arrays,
            prefix=prefix,
            weight=weight,
            rate=choice["rate"],
            group_size=group_size,
        )
        row.update(
            {
                "group": choice["group"],
                "tensor": tensor_name,
                "bucket": (choice.get("meta") or {}).get("bucket"),
                "damage": choice.get("damage"),
                **slice_meta,
            }
        )
        source_bytes += int(row["source_bytes"])
        groups.append(row)

    artifact = out_dir / f"{safe_name(tensor_name)}.mixed_parent.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, **arrays)
    return {
        "tensor": tensor_name,
        "weight_patch_key": patch_key,
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "source_bytes": source_bytes,
        "groups_packed": len(groups),
        "partial": len(groups) < original_choice_count,
        "available_groups": original_choice_count,
        "groups": groups,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rate: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "source_bytes": 0.0})
    total_artifact = 0
    total_source = 0
    for parent in rows:
        total_artifact += int(parent["artifact_bytes"])
        total_source += int(parent["source_bytes"])
        for group in parent["groups"]:
            rate = group["rate"]
            by_rate[rate]["groups"] += 1
            by_rate[rate]["source_bytes"] += int(group["source_bytes"])
    return {
        "parents": len(rows),
        "groups": sum(int(row["groups_packed"]) for row in rows),
        "artifact_bytes": total_artifact,
        "source_bytes": total_source,
        "compression": total_source / total_artifact if total_artifact else None,
        "by_rate": {key: dict(value) for key, value in sorted(by_rate.items())},
    }


def manifest_payload(
    *,
    args: argparse.Namespace,
    group_size: int,
    bucket_group_sizes: dict[str, int],
    rows: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "model_dir": str(args.model_dir),
        "inventory": str(args.inventory),
        "policy": str(args.policy),
        "weight_patch": str(args.weight_patch) if args.weight_patch else None,
        "group_size": group_size,
        "bucket_group_sizes": bucket_group_sizes,
        "include_experts": args.include_experts,
        "rates": sorted(parse_csv(args.rates) or PACKABLE_RATES),
        "buckets": sorted(parse_csv(args.buckets) or []),
        "max_parent_tensors": args.max_parent_tensors,
        "max_groups_per_parent": args.max_groups_per_parent,
        "max_params_per_group": args.max_params_per_group,
        "complete": complete,
        "packed": rows,
        "summary": summarize(rows),
        "note": "bounded parent-tensor artifact; remove max limits for full policy packing",
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def summarize_selection(selected_by_tensor: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_rate: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "params": 0.0, "source_bytes": 0.0})
    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "params": 0.0, "source_bytes": 0.0})
    groups = 0
    params = 0
    for choices in selected_by_tensor.values():
        for choice in choices:
            meta = choice.get("meta") or {}
            group_params = int(meta.get("params", 0))
            source_bytes = group_params * 2
            groups += 1
            params += group_params
            for table, key in ((by_rate, choice["rate"]), (by_bucket, str(meta.get("bucket", "unknown")))):
                table[key]["groups"] += 1
                table[key]["params"] += group_params
                table[key]["source_bytes"] += source_bytes
    return {
        "parents": len(selected_by_tensor),
        "groups": groups,
        "params": params,
        "source_bytes": params * 2,
        "by_rate": {key: dict(value) for key, value in sorted(by_rate.items())},
        "by_bucket": {key: dict(value) for key, value in sorted(by_bucket.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    parser.add_argument("--policy", default="/workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json", type=Path)
    parser.add_argument("--weight-patch", type=Path)
    parser.add_argument("--out-dir", default="/workspace/agentworld_runs/pack_parent_qwen_agentworld", type=Path)
    parser.add_argument("--include-experts", action="store_true")
    parser.add_argument("--rates")
    parser.add_argument("--buckets")
    parser.add_argument("--max-parent-tensors", type=int, default=1)
    parser.add_argument("--max-groups-per-parent", type=int, default=24)
    parser.add_argument("--max-params-per-group", type=int, default=2_200_000)
    parser.add_argument(
        "--bucket-group-size",
        action="append",
        help="override group size for one policy bucket, e.g. mlp=32; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    policy = json.loads(args.policy.read_text())
    weight_patch = load_weight_patch(args.weight_patch)
    tensor_summary = inventory["tensor_summary"]
    group_size = int(policy.get("group_size", 112))
    bucket_group_sizes = parse_bucket_group_sizes(args.bucket_group_size)

    selected_by_tensor = select_choices(
        policy,
        include_experts=args.include_experts,
        rates=parse_csv(args.rates),
        buckets=parse_csv(args.buckets),
        max_params_per_group=limit_or_none(args.max_params_per_group),
    )

    if args.dry_run:
        result = {
            "model_dir": str(args.model_dir),
            "inventory": str(args.inventory),
            "policy": str(args.policy),
            "include_experts": args.include_experts,
            "rates": sorted(parse_csv(args.rates) or PACKABLE_RATES),
            "buckets": sorted(parse_csv(args.buckets) or []),
            "max_parent_tensors": args.max_parent_tensors,
            "max_groups_per_parent": args.max_groups_per_parent,
            "max_params_per_group": args.max_params_per_group,
            "bucket_group_sizes": bucket_group_sizes,
            "dry_run": True,
            "summary": summarize_selection(selected_by_tensor),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dry_path = args.out_dir / "dry_run.json"
        dry_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"dry_run": str(dry_path), "summary": result["summary"]}, indent=2))
        return

    manifest_path = args.out_dir / "manifest.json"
    partial_manifest_path = args.out_dir / "manifest.partial.json"
    rows = []
    completed_tensors: set[str] = set()
    if args.resume:
        resume_path = manifest_path if manifest_path.exists() else partial_manifest_path
        if resume_path.exists():
            previous = json.loads(resume_path.read_text())
            rows = list(previous.get("packed", []))
            completed_tensors = {str(row["tensor"]) for row in rows}
            print(json.dumps({"resuming_from": str(resume_path), "completed_tensors": len(completed_tensors)}, indent=2), flush=True)

    max_parent_tensors = limit_or_none(args.max_parent_tensors)
    max_groups_per_parent = limit_or_none(args.max_groups_per_parent)
    for tensor_name, choices in selected_by_tensor.items():
        if tensor_name in completed_tensors:
            continue
        if max_parent_tensors is not None and len(rows) >= max_parent_tensors:
            break
        if not choices:
            continue
        row = pack_parent_tensor(
            model_dir=args.model_dir,
            tensor_summary=tensor_summary,
            tensor_name=tensor_name,
            choices=choices,
            out_dir=args.out_dir,
            default_group_size=group_size,
            bucket_group_sizes=bucket_group_sizes,
            max_groups_per_parent=max_groups_per_parent,
            weight_patch=weight_patch,
        )
        rows.append(row)
        write_manifest(
            partial_manifest_path,
            manifest_payload(
                args=args,
                group_size=group_size,
                bucket_group_sizes=bucket_group_sizes,
                rows=rows,
                complete=False,
            ),
        )
        print(json.dumps({k: row[k] for k in ("tensor", "artifact", "artifact_bytes", "source_bytes", "groups_packed")}, indent=2), flush=True)

    manifest = manifest_payload(
        args=args,
        group_size=group_size,
        bucket_group_sizes=bucket_group_sizes,
        rows=rows,
        complete=True,
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
