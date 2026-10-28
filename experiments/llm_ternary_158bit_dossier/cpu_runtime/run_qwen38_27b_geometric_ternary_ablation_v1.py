#!/usr/bin/env python3
"""Preregistered real-weight geometric ternary screen for Qwen3.8-27B."""
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

import numpy as np
from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_affine_ternary import (
    affine_ternary_codes_scales,
    reconstruct_affine_ternary,
)
from clark_quantize.qwen38_geometric_transform import block_rht, inverse_block_rht
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
            raise RuntimeError(f"frozen geometric source changed: {label}")


def entropy_bits_per_code(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty ternary histogram")
    return -sum(
        count / total * math.log2(count / total)
        for count in counts.values()
        if count
    )


def code_histogram(codes: torch.Tensor) -> Counter[int]:
    values, counts = torch.unique(codes, return_counts=True)
    return Counter({int(value): int(count) for value, count in zip(values, counts)})


def symmetric_fit(source: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    codes, scales = ternary_codes_scales(source, group=group, method="optimal")
    scale_fp16 = scales.to(torch.float16).to(torch.float32)
    reconstructed = scale_fp16.repeat_interleave(group, dim=1)[:, : source.shape[1]]
    reconstructed = reconstructed * codes.to(torch.float32)
    return codes, reconstructed


def affine_fit(
    source: torch.Tensor, group: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...]]:
    fit = affine_ternary_codes_scales(source, group=group, max_iterations=12)
    export_fit = type(fit)(
        codes=fit.codes,
        scales=fit.scales.to(torch.float16).to(torch.float32),
        centers=fit.centers.to(torch.float16).to(torch.float32),
        objective_history=fit.objective_history,
        iterations=fit.iterations,
        converged=fit.converged,
    )
    return fit.codes, reconstruct_affine_ternary(export_fit, group=group), fit.objective_history


def affine_kernel_output(
    activation: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    centers: torch.Tensor,
    group: int,
) -> torch.Tensor:
    batch = activation.shape[0]
    rows, columns = codes.shape
    groups = columns // group
    activation_groups = activation.reshape(batch, groups, group)
    code_groups = codes.to(torch.float32).reshape(rows, groups, group)
    dot = torch.einsum("bgk,rgk->brg", activation_groups, code_groups)
    return (dot * scales[None, :, :]).sum(dim=2) + activation.sum(dim=1)[:, None] * centers


def evaluate_span(
    source: torch.Tensor,
    *,
    group: int,
    seed: int,
    probe_seed: int,
) -> dict:
    source = source.to(torch.float32).contiguous()
    rows, columns = source.shape
    source_energy = float(source.square().sum(dtype=torch.float64).item())
    variants: dict[str, dict] = {}

    baseline_codes, baseline_reconstructed = symmetric_fit(source, group)
    variants["symmetric"] = {
        "codes": baseline_codes,
        "reconstructed": baseline_reconstructed,
        "center_bytes": 0,
        "coordinate_objectives": [],
        "rotated": False,
    }

    affine_result = affine_ternary_codes_scales(source, group=group, max_iterations=12)
    affine_scales = affine_result.scales.to(torch.float16).to(torch.float32)
    affine_centers = affine_result.centers.to(torch.float16).to(torch.float32)
    affine_export = type(affine_result)(
        codes=affine_result.codes,
        scales=affine_scales,
        centers=affine_centers,
        objective_history=affine_result.objective_history,
        iterations=affine_result.iterations,
        converged=affine_result.converged,
    )
    variants["affine"] = {
        "codes": affine_result.codes,
        "reconstructed": reconstruct_affine_ternary(affine_export, group=group),
        "center_bytes": rows * 2,
        "coordinate_objectives": list(affine_result.objective_history),
        "rotated": False,
        "scales": affine_scales,
        "centers": affine_centers,
    }

    rotated_source, signs = block_rht(source, group=group, seed=seed)
    rotated_codes, rotated_reconstructed = symmetric_fit(rotated_source, group)
    variants["rht_symmetric"] = {
        "codes": rotated_codes,
        "reconstructed": rotated_reconstructed,
        "center_bytes": 0,
        "coordinate_objectives": [],
        "rotated": True,
    }
    rotated_affine = affine_ternary_codes_scales(
        rotated_source, group=group, max_iterations=12
    )
    rotated_affine_scales = rotated_affine.scales.to(torch.float16).to(torch.float32)
    rotated_affine_centers = rotated_affine.centers.to(torch.float16).to(torch.float32)
    rotated_affine_export = type(rotated_affine)(
        codes=rotated_affine.codes,
        scales=rotated_affine_scales,
        centers=rotated_affine_centers,
        objective_history=rotated_affine.objective_history,
        iterations=rotated_affine.iterations,
        converged=rotated_affine.converged,
    )
    variants["rht_affine"] = {
        "codes": rotated_affine.codes,
        "reconstructed": reconstruct_affine_ternary(rotated_affine_export, group=group),
        "center_bytes": rows * 2,
        "coordinate_objectives": list(rotated_affine.objective_history),
        "rotated": True,
        "scales": rotated_affine_scales,
        "centers": rotated_affine_centers,
    }

    generator = torch.Generator().manual_seed(probe_seed)
    probes = torch.randn(8, columns, generator=generator, dtype=torch.float32)
    reference_output = probes @ source.T
    rotated_probes, observed_signs = block_rht(probes, group=group, signs=signs)
    if not torch.equal(observed_signs, signs):
        raise AssertionError("RHT signs changed between weight and activations")

    output = {}
    for name, variant in variants.items():
        reconstructed_domain = variant.pop("reconstructed")
        reconstructed_original = (
            inverse_block_rht(reconstructed_domain, signs, group=group)
            if variant["rotated"]
            else reconstructed_domain
        )
        residual = reconstructed_original - source
        squared_error = float(residual.square().sum(dtype=torch.float64).item())
        codes = variant.pop("codes")
        histogram = code_histogram(codes)
        activation_for_kernel = rotated_probes if variant["rotated"] else probes
        if "scales" in variant:
            kernel_output = affine_kernel_output(
                activation_for_kernel,
                codes,
                variant.pop("scales"),
                variant.pop("centers"),
                group,
            )
        else:
            kernel_output = activation_for_kernel @ reconstructed_domain.T
        dense_candidate_output = probes @ reconstructed_original.T
        kernel_error = float((kernel_output - dense_candidate_output).abs().max().item())
        output_error = float(
            (dense_candidate_output - reference_output).square().mean(dtype=torch.float64).item()
        )
        if kernel_error > 2e-4:
            raise RuntimeError(f"{name} CPU algebra identity failed: {kernel_error}")
        output[name] = {
            **variant,
            "parameters": source.numel(),
            "source_energy": source_energy,
            "squared_error": squared_error,
            "relative_rmse": math.sqrt(squared_error / max(source_energy, 1e-300)),
            "probe_output_mse": output_error,
            "kernel_identity_max_abs_error": kernel_error,
            "histogram": {str(symbol): histogram.get(symbol, 0) for symbol in (-1, 0, 1)},
            "empirical_entropy_bits_per_code": entropy_bits_per_code(histogram),
            "center_bits_per_weight": variant["center_bytes"] * 8 / source.numel(),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-geometric-ternary-ablation-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_geometric_ternary_ablation_preregistration_v1.json"
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

    group = int(prereg["candidate_family"]["group_size"])
    seed = int(prereg["candidate_family"]["rht_seed"])
    records = []
    aggregate = {
        name: {"parameters": 0, "source_energy": 0.0, "squared_error": 0.0, "counts": Counter(), "center_bytes": 0}
        for name in ("symmetric", "affine", "rht_symmetric", "rht_affine")
    }
    for tensor_index, tensor_spec in enumerate(prereg["scope"]["tensors"]):
        name = tensor_spec["name"]
        shard = weight_map.get(name)
        if shard != tensor_spec["shard"]:
            raise RuntimeError(f"Qwen3.8 tensor shard changed: {name}")
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            if list(view.get_shape()) != tensor_spec["shape"]:
                raise RuntimeError(f"Qwen3.8 tensor shape changed: {name}")
            for span_index, span in enumerate(tensor_spec["row_spans"]):
                start, stop = map(int, span)
                source = view[start:stop].contiguous().reshape(stop - start, -1).float()
                observed = evaluate_span(
                    source,
                    group=group,
                    seed=seed,
                    probe_seed=seed + tensor_index * 100 + span_index,
                )
                records.append({"tensor": name, "row_start": start, "row_stop": stop, "variants": observed})
                for variant_name, metrics in observed.items():
                    target = aggregate[variant_name]
                    target["parameters"] += int(metrics["parameters"])
                    target["source_energy"] += float(metrics["source_energy"])
                    target["squared_error"] += float(metrics["squared_error"])
                    target["center_bytes"] += int(metrics["center_bytes"])
                    target["counts"].update(
                        {int(symbol): int(count) for symbol, count in metrics["histogram"].items()}
                    )
                del source

    summary = {}
    for name, totals in aggregate.items():
        summary[name] = {
            "parameters": totals["parameters"],
            "source_energy": totals["source_energy"],
            "squared_error": totals["squared_error"],
            "relative_rmse": math.sqrt(totals["squared_error"] / totals["source_energy"]),
            "histogram": {str(symbol): totals["counts"].get(symbol, 0) for symbol in (-1, 0, 1)},
            "empirical_entropy_bits_per_code": entropy_bits_per_code(totals["counts"]),
            "center_bytes": totals["center_bytes"],
            "center_bits_per_weight": totals["center_bytes"] * 8 / totals["parameters"],
        }
    baseline_error = summary["symmetric"]["squared_error"]
    for name, metrics in summary.items():
        metrics["squared_error_reduction_vs_symmetric"] = 1.0 - metrics["squared_error"] / baseline_error
        metrics["entropy_plus_center_bits_per_weight"] = (
            metrics["empirical_entropy_bits_per_code"] + metrics["center_bits_per_weight"]
        )
    eligible = ["affine", "rht_symmetric", "rht_affine"]
    winner = min(eligible, key=lambda name: summary[name]["squared_error"])
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "selected_candidate": winner,
        "observed_squared_error_reduction": summary[winner]["squared_error_reduction_vs_symmetric"],
        "observed_entropy_plus_center_bits_per_weight": summary[winner]["entropy_plus_center_bits_per_weight"],
        "observed_symmetric_entropy_bits_per_weight": summary["symmetric"]["empirical_entropy_bits_per_code"],
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_spans"])
        and summary[winner]["squared_error_reduction_vs_symmetric"]
        >= float(gate_spec["minimum_aggregate_squared_error_reduction"])
        and summary[winner]["entropy_plus_center_bits_per_weight"]
        <= summary["symmetric"]["empirical_entropy_bits_per_code"]
        + float(gate_spec["maximum_candidate_rate_increase_bits_per_weight"])
    )
    receipt = {
        "schema": "qwen38_27b_geometric_ternary_ablation_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "model": {"id": MODEL_ID, "revision": REVISION, "index_sha256": INDEX_SHA256},
        "records": records,
        "summary": summary,
        "gate": gate,
        "claim_boundary": {
            "real_qwen38_27b_weights_measured": True,
            "sampled_rows_only": True,
            "activation_calibrated": False,
            "full_model_exported": False,
            "capability_preserved": False,
            "cpu_throughput_improved": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
