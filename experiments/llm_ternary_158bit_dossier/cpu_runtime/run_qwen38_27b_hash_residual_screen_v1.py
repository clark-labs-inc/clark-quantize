#!/usr/bin/env python3
"""Frozen Clark-Hash-assisted ternary residual screen on Qwen3.8-27B weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    fit_hash_ternary_residual,
    projected_hash_residual_bits_per_weight,
    quantize_hash_coefficients_int4,
    sketch_rows,
)
from clark_quantize.qwen38_ternary_entropy import empirical_entropy_bits_per_trit
from clark_quantize.ternary import ternary_codes_scales


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen hash-residual source changed: {label}")


def histogram(codes: torch.Tensor) -> list[int]:
    counts = Counter(int(value) for value in codes.reshape(-1).tolist())
    return [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]


def baseline_metrics(source: torch.Tensor, group: int) -> dict:
    codes, scales = ternary_codes_scales(source, group=group, method="optimal")
    exported_scales = scales.to(torch.float16).to(torch.float32)
    reconstructed = codes.to(torch.float32) * exported_scales.repeat_interleave(group, dim=1)
    squared_error = float((source - reconstructed).square().sum(dtype=torch.float64).item())
    counts = histogram(codes)
    return {
        "squared_error": squared_error,
        "relative_rmse": math.sqrt(squared_error / float(source.square().sum(dtype=torch.float64))),
        "counts": counts,
        "empirical_entropy_bits_per_weight": empirical_entropy_bits_per_trit(counts),
    }


def candidate_metrics(
    source: torch.Tensor,
    *,
    sketch_dim: int,
    nonzero_per_group: int,
    group: int,
    hashes_per_input: int,
    seed: int,
    iterations: int,
    probe_seed: int,
) -> dict:
    fit = fit_hash_ternary_residual(
        source,
        sketch_dim=sketch_dim,
        hashes_per_input=hashes_per_input,
        seed=seed,
        group=group,
        nonzero_per_group=nonzero_per_group,
        iterations=iterations,
    )
    _, _, coefficients = quantize_hash_coefficients_int4(fit.hash_coefficients)
    scales = fit.ternary_scales.to(torch.float16).to(torch.float32)
    ternary = (
        fit.ternary_codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * scales[:, :, None]
    ).reshape_as(source)
    reconstructed = ternary + expand_sketch_rows(coefficients, fit.projection)
    squared_error = float((source - reconstructed).square().sum(dtype=torch.float64).item())
    if not math.isclose(squared_error, fit.exported_squared_error, rel_tol=2e-6, abs_tol=1e-7):
        raise RuntimeError("hash-residual export error ledger changed")
    generator = torch.Generator().manual_seed(probe_seed)
    probes = torch.randn(8, source.shape[1], generator=generator)
    kernel = probes @ ternary.T + sketch_rows(probes, fit.projection) @ coefficients.T
    dense = probes @ reconstructed.T
    identity_error = float((kernel - dense).abs().max().item())
    if identity_error > 5e-4:
        raise RuntimeError(f"hash-residual CPU algebra identity failed: {identity_error}")
    counts = histogram(fit.ternary_codes)
    return {
        "sketch_dim": sketch_dim,
        "nonzero_per_group": nonzero_per_group,
        "squared_error": squared_error,
        "relative_rmse": math.sqrt(squared_error / float(source.square().sum(dtype=torch.float64))),
        "counts": counts,
        "empirical_ternary_entropy_bits_per_weight": empirical_entropy_bits_per_trit(counts),
        "projected_component_bits_per_weight": projected_hash_residual_bits_per_weight(
            input_dim=source.shape[1],
            sketch_dim=sketch_dim,
            nonzero_per_group=nonzero_per_group,
            group=group,
        ),
        "fp32_objective_history": list(fit.fp32_objective_history),
        "kernel_identity_max_abs_error": identity_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_hash_residual_screen_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8-27B checkpoint identity changed")
    index_content = (args.model_dir / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_content).hexdigest() != INDEX_SHA256:
        raise RuntimeError("Qwen3.8-27B tensor index changed")
    weight_map = json.loads(index_content)["weight_map"]
    family = prereg["candidate_family"]
    group = int(family["group_size"])
    records = []
    aggregate = {
        candidate["name"]: {"squared_error": 0.0, "parameters": 0, "weighted_rate": 0.0}
        for candidate in family["candidates"]
    }
    baseline_total = 0.0
    for tensor_index, tensor_spec in enumerate(prereg["scope"]["tensors"]):
        name = tensor_spec["name"]
        shard = weight_map.get(name)
        if shard != tensor_spec["shard"]:
            raise RuntimeError(f"Qwen3.8 hash-screen tensor shard changed: {name}")
        start, stop = tensor_spec["row_span"]
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            if list(view.get_shape()) != tensor_spec["shape"]:
                raise RuntimeError(f"Qwen3.8 hash-screen tensor shape changed: {name}")
            source = view[start:stop].contiguous().reshape(stop - start, -1).float()
        baseline = baseline_metrics(source, group)
        baseline_total += baseline["squared_error"]
        candidates = {}
        for candidate_index, candidate in enumerate(family["candidates"]):
            sketch_dim = source.shape[1] // int(candidate["input_to_sketch_ratio"])
            observed = candidate_metrics(
                source,
                sketch_dim=sketch_dim,
                nonzero_per_group=int(candidate["nonzero_per_group"]),
                group=group,
                hashes_per_input=int(family["hashes_per_input"]),
                seed=int(family["seed"]),
                iterations=int(family["alternating_iterations"]),
                probe_seed=int(family["seed"]) + tensor_index * 10 + candidate_index,
            )
            candidates[candidate["name"]] = observed
            target = aggregate[candidate["name"]]
            target["squared_error"] += observed["squared_error"]
            target["parameters"] += source.numel()
            target["weighted_rate"] += observed["projected_component_bits_per_weight"] * source.numel()
        records.append(
            {
                "tensor": name,
                "row_start": start,
                "row_stop": stop,
                "parameters": source.numel(),
                "baseline": baseline,
                "candidates": candidates,
            }
        )

    summary = {}
    for name, totals in aggregate.items():
        summary[name] = {
            "squared_error": totals["squared_error"],
            "squared_error_reduction_vs_symmetric": 1.0 - totals["squared_error"] / baseline_total,
            "projected_component_bits_per_weight": totals["weighted_rate"] / totals["parameters"],
        }
    winner = min(summary, key=lambda name: summary[name]["squared_error"])
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "selected_candidate": winner,
        "observed_squared_error_reduction": summary[winner]["squared_error_reduction_vs_symmetric"],
        "observed_projected_component_bits_per_weight": summary[winner]["projected_component_bits_per_weight"],
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_tensor_spans"])
        and summary[winner]["squared_error_reduction_vs_symmetric"]
        >= float(gate_spec["minimum_exported_squared_error_reduction"])
        and summary[winner]["projected_component_bits_per_weight"]
        <= float(gate_spec["maximum_projected_component_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_hash_residual_screen_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "clark_hash_source": prereg["clark_hash_source"],
        "records": records,
        "baseline_aggregate_squared_error": baseline_total,
        "summary": summary,
        "gate": gate,
        "claim_boundary": {
            "real_Qwen38_27B_weights_measured": True,
            "sampled_rows_only": True,
            "hash_coefficients_exported_as_int4": True,
            "ternary_group_scales_exported_as_FP16": True,
            "row_normalized_4bit_scale_quality_only_projected": True,
            "activation_weighted_or_trained": False,
            "capability_preservation_proven": False,
            "complete_storage_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
