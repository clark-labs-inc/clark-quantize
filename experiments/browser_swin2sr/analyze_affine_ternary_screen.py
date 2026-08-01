#!/usr/bin/env python3
"""Weight-only affine-ternary screen; never reads development or final data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from affine_ternary import fit_affine_ternary_codes, quantize_affine_ternary_mse
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from quantizer_controls import quantize_ternary_absmean
from clark_air.quant.optimal_ternary import quantize_ternary_mse


FORMAT = "backdropkit-swin2sr-affine-ternary-weight-screen-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def current_projection(surface, master: torch.Tensor):
    if surface.quantizer.method == "absmean":
        return quantize_ternary_absmean(master, surface.quantizer.group_size)
    if surface.quantizer.method == "optimal_mse":
        return quantize_ternary_mse(master, surface.quantizer.group_size)
    raise ValueError(f"unsupported current quantizer: {surface.quantizer.method}")


def stored_affine_values(quantized) -> torch.Tensor:
    codes = quantized.codes.reshape(-1, quantized.group_size).float()
    shifts = quantized.shifts.half().float().reshape(-1, 1)
    scales = quantized.scales.half().float().reshape(-1, 1)
    values = (shifts + scales * codes).reshape(-1)
    if quantized.pad:
        values = values[:-quantized.pad]
    return values.reshape(quantized.original_shape)


def squared_error(target: torch.Tensor, restored: torch.Tensor) -> float:
    return float((target.float() - restored.float()).square().sum().item())


def summarize(errors: dict[str, float], energy: float) -> dict[str, float]:
    return {
        **errors,
        "relative_rmse": math.sqrt(errors["squared_error"] / max(energy, 1e-30)),
    }


@torch.no_grad()
def analyze(checkpoint: Path) -> dict:
    _student, surfaces, payload = load_checkpoint_student(checkpoint, "cpu")
    rows = []
    category_totals = defaultdict(lambda: defaultdict(float))
    storage = defaultdict(int)
    for surface in surfaces:
        master = surface.module.parametrizations.weight.original.detach().cpu().float()
        current = current_projection(surface, master)
        fixed = fit_affine_ternary_codes(
            master,
            current.codes,
            current.group_size,
            current.pad,
        )
        reassigned = quantize_affine_ternary_mse(
            master,
            group_size=current.group_size,
        )
        energy = float(master.square().sum().item())
        current_error = squared_error(master, current.dequantize())
        fixed_fp32_error = squared_error(master, fixed.dequantize())
        fixed_fp16_error = squared_error(master, stored_affine_values(fixed))
        reassigned_fp32_error = squared_error(master, reassigned.dequantize())
        reassigned_fp16_error = squared_error(master, stored_affine_values(reassigned))
        groups = int(current.scales.numel())
        row = {
            "surface": surface.name,
            "category": surface.category,
            "values": master.numel(),
            "group_size": current.group_size,
            "groups": groups,
            "current_method": surface.quantizer.method,
            "energy": energy,
            "current_squared_error": current_error,
            "fixed_codes_affine_fp32_squared_error": fixed_fp32_error,
            "fixed_codes_affine_fp16_squared_error": fixed_fp16_error,
            "reassigned_affine_fp32_squared_error": reassigned_fp32_error,
            "reassigned_affine_fp16_squared_error": reassigned_fp16_error,
            "fixed_codes_fp16_error_reduction_fraction": (
                (current_error - fixed_fp16_error) / max(current_error, 1e-30)
            ),
            "mean_abs_shift_over_scale": float(
                (fixed.shifts.abs() / fixed.scales.clamp_min(1e-12)).mean().item()
            ),
        }
        rows.append(row)
        totals = category_totals[surface.category]
        totals["values"] += master.numel()
        totals["groups"] += groups
        totals["energy"] += energy
        totals["current_squared_error"] += current_error
        totals["fixed_codes_affine_fp16_squared_error"] += fixed_fp16_error
        totals["reassigned_affine_fp16_squared_error"] += reassigned_fp16_error
        storage["current_scale_bytes"] += groups * 2
        storage["affine_scale_and_shift_bytes"] += groups * 4

    def aggregate(records) -> dict:
        energy = sum(row["energy"] for row in records)
        current = sum(row["current_squared_error"] for row in records)
        fixed = sum(row["fixed_codes_affine_fp16_squared_error"] for row in records)
        reassigned = sum(
            row["reassigned_affine_fp16_squared_error"] for row in records
        )
        return {
            "values": sum(row["values"] for row in records),
            "groups": sum(row["groups"] for row in records),
            "energy": energy,
            "current": summarize({"squared_error": current}, energy),
            "fixed_codes_affine_fp16": summarize({"squared_error": fixed}, energy),
            "reassigned_affine_fp16": summarize(
                {"squared_error": reassigned}, energy
            ),
            "fixed_codes_error_reduction_fraction": (
                (current - fixed) / max(current, 1e-30)
            ),
            "reassigned_error_reduction_fraction": (
                (current - reassigned) / max(current, 1e-30)
            ),
        }

    categories = {
        category: aggregate([row for row in rows if row["category"] == category])
        for category in sorted(category_totals)
    }
    rows.sort(
        key=lambda row: row["current_squared_error"]
        - row["fixed_codes_affine_fp16_squared_error"],
        reverse=True,
    )
    return {
        "format": FORMAT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "surface_step": int(payload["surface_step"]),
        "data_access": {
            "training": False,
            "development": False,
            "final_heldout": False,
        },
        "interpretation": (
            "Weight reconstruction is a cheap screen, not an output-quality result. "
            "fixed_codes isolates affine shift/scale from support changes; reassigned "
            "codes are exploratory. Affine parameters are FP16-roundtripped."
        ),
        "aggregate": aggregate(rows),
        "categories": categories,
        "storage": {
            **storage,
            "incremental_affine_bytes": (
                storage["affine_scale_and_shift_bytes"]
                - storage["current_scale_bytes"]
            ),
        },
        "top_surfaces_by_fixed_code_error_reduction": rows[:24],
        "surfaces": rows,
    }


def main() -> None:
    args = parse_args()
    report = analyze(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "aggregate": report["aggregate"],
        "categories": report["categories"],
        "storage": report["storage"],
    }, indent=2))


if __name__ == "__main__":
    main()
