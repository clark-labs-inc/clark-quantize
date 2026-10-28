#!/usr/bin/env python3
"""Preregistered reconstruction gate for an honest approximately 1.59-bpp payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.base3 import pack_base3, unpack_base3


SCHEMA = "clark_qwen38_target_158_reconstruction_v1"


def tensor_inventory(model_dir: Path) -> tuple[dict[str, dict[str, Any]], int]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    inventory: dict[str, dict[str, Any]] = {}
    total = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                shape = list(handle.get_slice(name).get_shape())
                parameters = math.prod(shape)
                row = {"shard": shard, "shape": shape, "parameters": parameters}
                if len(shape) != 2:
                    row["dtype"] = str(handle.get_tensor(name).dtype)
                inventory[name] = row
                total += parameters
    return inventory, total


def retained_dtype_ledger(inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tensor_count_by_dtype: dict[str, int] = {}
    parameter_count_by_dtype: dict[str, int] = {}
    for row in inventory.values():
        if len(row["shape"]) == 2:
            continue
        dtype = str(row.get("dtype", "unknown"))
        tensor_count_by_dtype[dtype] = tensor_count_by_dtype.get(dtype, 0) + 1
        parameter_count_by_dtype[dtype] = parameter_count_by_dtype.get(dtype, 0) + int(row["parameters"])
    return {
        "tensor_count": sum(tensor_count_by_dtype.values()),
        "parameters": sum(parameter_count_by_dtype.values()),
        "tensor_count_by_dtype": tensor_count_by_dtype,
        "parameter_count_by_dtype": parameter_count_by_dtype,
        "all_bfloat16": set(tensor_count_by_dtype) == {"torch.bfloat16"},
    }


def candidate_spec(name: str, columns: int) -> tuple[int, str, int]:
    if name == "all_rank2_group1024_fp8_scales":
        return 1024, "float8_e4m3fn", 8
    if name == "all_rank2_group2048_fp8_scales":
        return 2048, "float8_e4m3fn", 8
    if name == "all_rank2_per_row_fp16_scales":
        return columns, "float16", 16
    raise ValueError(f"unknown sealed candidate: {name}")


def projected_ledger(inventory: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    total = sum(row["parameters"] for row in inventory.values())
    rank2 = sum(row["parameters"] for row in inventory.values() if len(row["shape"]) == 2)
    retained = total - rank2
    scales = 0
    scale_bits = None
    for row in inventory.values():
        shape = row["shape"]
        if len(shape) != 2:
            continue
        group, _, bits = candidate_spec(name, shape[1])
        scale_bits = bits
        scales += shape[0] * math.ceil(shape[1] / group)
    assert scale_bits is not None
    payload_bits = rank2 * math.log2(3) + retained * 16 + scales * scale_bits
    return {
        "total_parameters": total,
        "rank2_ternary_parameters": rank2,
        "rank_not_2_bf16_parameters": retained,
        "scale_count": scales,
        "scale_bits": scale_bits,
        "complete_payload_bits": payload_bits,
        "complete_effective_bits_per_parameter": payload_bits / total,
    }


@torch.no_grad()
def optimal_codes_scales(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or group <= 0:
        raise ValueError("optimal ternary sample requires a matrix and positive group")
    rows, columns = weight.shape
    padding = (-columns) % group
    padded = torch.nn.functional.pad(weight, (0, padding)) if padding else weight
    grouped = padded.reshape(rows, -1, group)
    absolute = grouped.abs()
    ordered, _ = torch.sort(absolute, dim=-1, descending=True)
    cumulative = ordered.cumsum(-1)
    counts = torch.arange(1, group + 1, device=weight.device, dtype=cumulative.dtype)
    best = (cumulative.square() / counts).argmax(-1)
    scales = (
        cumulative.gather(-1, best[..., None])
        / (best[..., None].to(cumulative.dtype) + 1)
    ).clamp_min(1e-8).squeeze(-1)
    threshold = ordered.gather(-1, best[..., None])
    codes = (torch.sign(grouped) * (absolute >= threshold)).to(torch.int8)
    return codes.reshape(rows, -1)[:, :columns].contiguous(), scales.contiguous()


def stored_scales(scales: torch.Tensor, dtype_name: str) -> tuple[torch.Tensor, bytes]:
    cpu = scales.detach().cpu()
    if dtype_name == "float16":
        stored = cpu.to(torch.float16)
        raw = stored.numpy().astype("<f2", copy=False).tobytes()
    elif dtype_name == "float8_e4m3fn":
        stored = cpu.to(torch.float8_e4m3fn)
        raw = stored.view(torch.uint8).numpy().tobytes()
    else:
        raise ValueError(f"unsupported scale dtype: {dtype_name}")
    return stored.float(), raw


@torch.no_grad()
def evaluate_candidate(weight: torch.Tensor, name: str) -> dict[str, Any]:
    rows, columns = weight.shape
    group, dtype_name, bits = candidate_spec(name, columns)
    codes, scales = optimal_codes_scales(weight.float(), group)
    stored, scale_raw = stored_scales(scales, dtype_name)
    groups_per_row = math.ceil(columns / group)
    code_groups = torch.nn.functional.pad(codes, (0, (-columns) % group)).reshape(rows, groups_per_row, group)
    reconstructed = (code_groups.float().cpu() * stored.unsqueeze(-1)).reshape(rows, -1)[:, :columns]
    source = weight.detach().float().cpu()
    difference = reconstructed - source
    relative_l2 = float(difference.norm() / source.norm().clamp_min(1e-12))
    mse = float(difference.square().mean())
    variance = float(source.square().mean())
    cosine = float(torch.nn.functional.cosine_similarity(source.reshape(1, -1), reconstructed.reshape(1, -1)).item())
    scale_relative_l2 = float((stored - scales.cpu()).norm() / scales.cpu().norm().clamp_min(1e-12))
    code_raw = codes.cpu().numpy().tobytes()
    frame = pack_base3(codes.cpu().numpy().reshape(-1))
    decoded = unpack_base3(frame)
    exact_roundtrip = bool(np.array_equal(decoded, codes.cpu().numpy().reshape(-1)))
    finite = all(math.isfinite(value) for value in (relative_l2, mse, variance, cosine, scale_relative_l2))
    return {
        "candidate": name,
        "group": group,
        "scale_dtype": dtype_name,
        "scale_bits": bits,
        "parameters": int(source.numel()),
        "scale_count": int(stored.numel()),
        "zero_fraction": float((codes == 0).float().mean().cpu()),
        "relative_l2_error": relative_l2,
        "normalized_mse": mse / max(variance, 1e-30),
        "cosine_similarity": cosine,
        "scale_relative_l2_error": scale_relative_l2,
        "codes_exact_ternary": bool(torch.all((codes >= -1) & (codes <= 1)).item()),
        "exact_code_roundtrip": exact_roundtrip,
        "codes_sha256": hashlib.sha256(code_raw).hexdigest(),
        "base3_frame_sha256": hashlib.sha256(frame).hexdigest(),
        "stored_scales_sha256": hashlib.sha256(scale_raw).hexdigest(),
        "finite": finite,
    }


def load_sample(model_dir: Path, metadata: dict[str, Any], sample: dict[str, Any], device: str) -> torch.Tensor:
    start = int(sample["row_start"])
    stop = start + int(sample["row_count"])
    shape = metadata["shape"]
    if len(shape) != 2 or start < 0 or stop > shape[0]:
        raise ValueError(f"invalid sealed row sample for {sample['tensor']}")
    with safe_open(model_dir / metadata["shard"], framework="pt", device="cpu") as handle:
        return handle.get_slice(sample["tensor"])[start:stop, :].contiguous().to(device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("target_158_preregistration.json"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-target-158-reconstruction.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    prereg_content = args.preregistration.read_bytes()
    prereg = json.loads(prereg_content)
    preflight = json.loads(args.preflight.read_text())
    revision = preflight["download_receipt"]["revision"]
    if prereg.get("schema") != "clark_qwen38_target_158_preregistration_v1" or prereg.get("model_revision") != revision:
        raise ValueError("target-1.58 preregistration identity mismatch")
    if prereg.get("promotion", {}).get("does_not_authorize_full_pack") is not True:
        raise ValueError("reconstruction gate may not authorize a full pack")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    inventory, total = tensor_inventory(args.model_dir)
    expected = prereg["fixed_inventory_expectations"]
    candidates = [row["name"] for row in prereg["comparison_candidates"]]
    ledgers = {name: projected_ledger(inventory, name) for name in candidates}
    sealed_projections = {
        row["name"]: float(row["projected_complete_effective_bits_per_parameter"])
        for row in prereg["comparison_candidates"]
    }
    if set(sealed_projections) != set(ledgers) or any(
        abs(ledgers[name]["complete_effective_bits_per_parameter"] - sealed_projections[name]) > 1e-14
        for name in ledgers
    ):
        raise ValueError("candidate projected ledger differs from preregistration")
    per_row = ledgers["all_rank2_per_row_fp16_scales"]
    retained_dtypes = retained_dtype_ledger(inventory)
    inventory_valid = (
        total == expected["total_parameters"]
        and per_row["rank2_ternary_parameters"] == expected["rank2_ternary_parameters"]
        and per_row["rank_not_2_bf16_parameters"] == expected["rank_not_2_bf16_parameters"]
        and retained_dtypes["parameters"] == expected["rank_not_2_bf16_parameters"]
        and retained_dtypes["tensor_count"] == expected["rank_not_2_bf16_tensor_count"]
        and retained_dtypes["all_bfloat16"]
        and per_row["scale_count"] == expected["per_row_fp16_scale_count"]
        and abs(per_row["complete_effective_bits_per_parameter"] - expected["projected_complete_effective_bits_per_parameter"]) <= 1e-14
    )
    if not inventory_valid:
        raise ValueError("pinned checkpoint inventory differs from sealed target-1.58 ledger")

    rows = []
    aggregate: dict[str, dict[str, float]] = {name: {"weighted_squared_error": 0.0, "parameters": 0} for name in candidates}
    for sample in prereg["reconstruction_samples"]:
        name = sample["tensor"]
        if name not in inventory:
            raise ValueError(f"sealed tensor absent: {name}")
        weight = load_sample(args.model_dir, inventory[name], sample, args.device)
        source = weight.detach().float().cpu().numpy().astype("<f4", copy=False).tobytes()
        evaluations = []
        source_energy = float(weight.float().square().sum().cpu())
        for candidate in candidates:
            result = evaluate_candidate(weight, candidate)
            evaluations.append(result)
            aggregate[candidate]["weighted_squared_error"] += result["normalized_mse"] * source_energy
            aggregate[candidate]["parameters"] += result["parameters"]
        rows.append({
            **sample,
            "tensor_shape": inventory[name]["shape"],
            "source_energy": source_energy,
            "source_float32_sha256": hashlib.sha256(source).hexdigest(),
            "candidates": evaluations,
        })
        del weight

    retained = prereg["retained_control"]
    retained_meta = inventory.get(retained["tensor"])
    if retained_meta is None or len(retained_meta["shape"]) != retained["expected_rank"]:
        raise ValueError("retained convolution control identity mismatch")
    with safe_open(args.model_dir / retained_meta["shard"], framework="pt", device="cpu") as handle:
        control = handle.get_tensor(retained["tensor"])
    control_raw = control.contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    retained_control = {
        **retained,
        "shape": retained_meta["shape"],
        "parameters": retained_meta["parameters"],
        "source_dtype": str(control.dtype),
        "source_sha256": hashlib.sha256(control_raw).hexdigest(),
        "retained_exactly": str(control.dtype) == "torch.bfloat16",
    }

    summaries = {}
    for name in candidates:
        metrics = [evaluation for row in rows for evaluation in row["candidates"] if evaluation["candidate"] == name]
        total_sample_parameters = sum(metric["parameters"] for metric in metrics)
        weighted_relative_l2 = math.sqrt(
            sum(metric["relative_l2_error"] ** 2 * metric["parameters"] for metric in metrics)
            / total_sample_parameters
        )
        summaries[name] = {
            "sample_parameters": total_sample_parameters,
            "parameter_weighted_relative_l2_error": weighted_relative_l2,
            "maximum_relative_l2_error": max(metric["relative_l2_error"] for metric in metrics),
            "minimum_cosine_similarity": min(metric["cosine_similarity"] for metric in metrics),
            "all_finite": all(metric["finite"] for metric in metrics),
            "all_codes_exact_ternary": all(metric["codes_exact_ternary"] for metric in metrics),
            "all_code_roundtrips_exact": all(metric["exact_code_roundtrip"] for metric in metrics),
        }
    ceiling = float(prereg["promotion"]["maximum_complete_effective_bits_per_parameter"])
    eligible = [
        name for name in candidates
        if ledgers[name]["complete_effective_bits_per_parameter"] <= ceiling
        and summaries[name]["all_finite"]
        and summaries[name]["all_codes_exact_ternary"]
        and summaries[name]["all_code_roundtrips_exact"]
    ]
    selected = min(
        eligible,
        key=lambda name: (
            summaries[name]["parameter_weighted_relative_l2_error"],
            ledgers[name]["complete_effective_bits_per_parameter"],
            name,
        ),
    ) if eligible else None
    categories = {row["category"] for row in prereg["reconstruction_samples"]}
    coverage = {
        "language": any(value.startswith("language_") for value in categories),
        "vision": any(value.startswith("vision_") for value in categories),
        "mtp": any(value.startswith("mtp_") for value in categories),
        "linear_attention": {"linear_attention_a", "linear_attention_b"}.issubset(categories),
        "convolution_retained_control": retained_control["retained_exactly"],
        "all_rank_not_2_bfloat16": retained_dtypes["all_bfloat16"],
    }
    passed = inventory_valid and selected is not None and all(coverage.values())
    receipt = {
        "schema": SCHEMA,
        "hypothesis": prereg["hypothesis"],
        "model_revision": revision,
        "preregistration": {"path": str(args.preregistration), "sha256": hashlib.sha256(prereg_content).hexdigest()},
        "inventory_valid": inventory_valid,
        "retained_dtype_ledger": retained_dtypes,
        "ledgers": ledgers,
        "samples": rows,
        "retained_control": retained_control,
        "coverage": coverage,
        "summaries": summaries,
        "selection_rule": prereg["selection_rule"],
        "selected_candidate": selected,
        "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "promotion": {
            "passed": passed,
            "scope": "storage and reconstruction candidate only; does not authorize a full pack or model-quality claim",
            "authorizes_full_pack": False,
        },
        "stop_condition": prereg["stop_condition"],
    }
    atomic_bytes(args.out, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
