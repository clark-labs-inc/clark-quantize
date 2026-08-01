#!/usr/bin/env python3
"""Measure convolution-aware ternary grouping before committing to QAT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from data_pipeline import fixed_validation
from quantizer_controls import (
    quantize_ternary_absmean,
)
from surface_model_state import (
    pretrained,
    recovery_dense_state,
)
from surface_quantization import surface_group_size
from surface_schedule import module_category
from train_swin2sr_qat import VALIDATION_FILES, validate


GROUPINGS = ("fp32", "flat", "output_channel", "kernel")
METHODS = ("absmean", "optimal_mse")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument(
        "--categories",
        choices=(
            "image_io_conv",
            "first_convolution",
            "upsample_conv",
            "all_conv",
        ),
        default="image_io_conv",
    )
    return parser.parse_args()


def selected(name: str, category: str | None, scope: str) -> bool:
    if scope == "image_io_conv":
        return category == "image_io_conv"
    if scope == "first_convolution":
        return name == "swin2sr.first_convolution"
    if scope == "upsample_conv":
        return name == "upsample.conv"
    return category in {"image_io_conv", "residual_body_conv"}


def quantized_weight(
    weight: torch.Tensor,
    group_size: int,
    method: str,
) -> torch.Tensor:
    if method == "absmean":
        return quantize_ternary_absmean(weight, group_size).dequantize()
    from clark_air.quant.optimal_ternary import quantize_ternary_mse

    return quantize_ternary_mse(weight, group_size).dequantize()


def apply_grouping(
    model: torch.nn.Module,
    grouping: str,
    method: str,
    default_group_size: int,
    scope: str,
) -> list[dict]:
    ledger = []
    if grouping == "fp32":
        return ledger
    with torch.no_grad():
        for name, module in model.named_modules():
            category = module_category(name, module)
            if not selected(name, category, scope):
                continue
            group_size = surface_group_size(
                module,
                category,
                default_group_size,
                grouping,
            )
            module.weight.copy_(
                quantized_weight(module.weight, group_size, method)
            )
            ledger.append({
                "name": name,
                "category": category,
                "shape": list(module.weight.shape),
                "values": module.weight.numel(),
                "group_size": group_size,
                "scale_count": (
                    module.weight.numel() + group_size - 1
                ) // group_size,
            })
    return ledger


def main() -> None:
    args = parse_args()
    payload = torch.load(args.base_checkpoint, map_location="cpu")
    dense_state = recovery_dense_state(payload, args.device)
    teacher = pretrained(args.device).eval()
    teacher.requires_grad_(False)
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    results = []
    for method in METHODS:
        for grouping in GROUPINGS:
            if grouping == "fp32" and method != METHODS[0]:
                continue
            student = pretrained(args.device).eval()
            student.load_state_dict(dense_state, strict=True)
            ledger = apply_grouping(
                student,
                grouping,
                method,
                args.group_size,
                args.categories,
            )
            metrics = validate(student, teacher, validation)
            results.append({
                "grouping": grouping,
                "method": None if grouping == "fp32" else method,
                "metrics": metrics,
                "ledger": ledger,
            })
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    baseline = next(row for row in results if row["grouping"] == "fp32")
    comparison_metrics = (
        "psnr_vs_ground_truth",
        "edge_psnr_vs_ground_truth",
        "psnr_vs_teacher",
    )
    for row in results:
        row["delta_vs_fp32"] = {
            name: row["metrics"][name] - baseline["metrics"][name]
            for name in comparison_metrics
            if name in row["metrics"] and name in baseline["metrics"]
        }
    report = {
        "format": "backdropkit-swin2sr-conv-grouping-probe-v1",
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "validation_data": str(args.validation_data),
        "categories": args.categories,
        "default_group_size": args.group_size,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
