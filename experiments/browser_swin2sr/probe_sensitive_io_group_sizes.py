#!/usr/bin/env python3
"""Measure ternary scale-group tradeoffs for one sensitive Swin2SR I/O layer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from analyze_surface_signatures import codes, signature_counts, summarize
from data_pipeline import fixed_validation
from evaluate_hard_started_checkpoints import (
    checkpoint_fork_provenance,
    evaluate,
    load_checkpoint_student,
    validation_receipts,
    verify_study_sources,
)
from full_surface_provenance import file_sha256
from surface_curriculum import enter_hard_shadow, restore_live_amounts
from surface_model_state import pretrained
from train_swin2sr_qat import VALIDATION_FILES


METHODS = ("optimal_mse", "absmean")
TARGETS = ("swin2sr.first_convolution", "upsample.conv")
IDEAL_TERNARY_BITS = math.log2(3)


def parse_group_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(","))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("group sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("group sizes must be unique")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    parser.add_argument(
        "--group-sizes",
        type=parse_group_sizes,
        default=parse_group_sizes("1,2,3,6,9,15,27,45,90,270,1620"),
    )
    return parser.parse_args()


def storage_receipt(
    values: int,
    group_size: int,
    total_quantized_values: int,
    reference_group_size: int,
) -> dict[str, float | int]:
    scale_count = math.ceil(values / group_size)
    reference_scales = math.ceil(values / reference_group_size)
    scale_bits = scale_count * 16
    return {
        "values": values,
        "group_size": group_size,
        "scale_count": scale_count,
        "ideal_code_bits": values * IDEAL_TERNARY_BITS,
        "scale_bits": scale_bits,
        "layer_bits_per_weight": IDEAL_TERNARY_BITS + scale_bits / values,
        "whole_model_bit_delta_vs_reference": (
            (scale_count - reference_scales) * 16 / total_quantized_values
        ),
    }


@torch.inference_mode()
def target_signature(
    base: torch.Tensor,
    master: torch.Tensor,
    quantizer,
) -> dict[str, float | int]:
    return summarize(signature_counts(
        base,
        master,
        codes(base, quantizer),
        codes(master, quantizer),
    ))


def study_sealed_source_root(
    study: dict,
    manifest_path: Path,
    repository: Path,
) -> Path:
    configured = study.get("sealed_source_root")
    path = (
        Path(configured)
        if configured
        else manifest_path.parent / "sealed_sources"
    )
    return path if path.is_absolute() else repository / path


def probe(
    checkpoint: Path,
    target_name: str,
    group_sizes: tuple[int, ...],
    teacher,
    samples,
    device: str,
    study: dict | None = None,
    repository: Path | None = None,
) -> dict:
    student, surfaces, payload = load_checkpoint_student(checkpoint, device)
    target = next(
        (surface for surface in surfaces if surface.name == target_name),
        None,
    )
    if target is None:
        raise ValueError(f"target surface is missing: {target_name}")
    if float(target.quantizer.amount.item()) <= 0:
        raise ValueError(f"target surface has not started: {target_name}")
    total_values = sum(surface.module.weight.numel() for surface in surfaces)
    base_weight = (
        teacher.get_submodule(target_name).weight.detach()
    )
    master_weight = (
        target.module.parametrizations.weight.original.detach()
    )
    live_amounts = enter_hard_shadow(surfaces)
    original_method = target.quantizer.method
    original_group_size = target.quantizer.group_size
    try:
        results = []
        for method in METHODS:
            for group_size in group_sizes:
                target.quantizer.method = method
                target.quantizer.group_size = group_size
                metrics = evaluate(student, teacher, samples)
                results.append({
                    "method": method,
                    "group_size": group_size,
                    "metrics": metrics,
                    "target_signature": target_signature(
                        base_weight,
                        master_weight,
                        target.quantizer,
                    ),
                    "storage": storage_receipt(
                        target.module.weight.numel(),
                        group_size,
                        total_values,
                        original_group_size,
                    ),
                })
    finally:
        target.quantizer.method = original_method
        target.quantizer.group_size = original_group_size
        restore_live_amounts(surfaces, live_amounts)
    ranked = sorted(
        results,
        key=lambda row: (
            row["metrics"]["psnr_vs_ground_truth"],
            row["metrics"]["psnr_vs_teacher"],
            -row["storage"]["whole_model_bit_delta_vs_reference"],
        ),
        reverse=True,
    )
    return {
        "surface_step": int(payload["surface_step"]),
        "target": target_name,
        "target_shape": list(target.module.weight.shape),
        "target_values": target.module.weight.numel(),
        "reference_method": original_method,
        "reference_group_size": original_group_size,
        "surface_group_sizes": dict(sorted(
            (
                payload["quantization"].get("surface_group_sizes", {})
                or {}
            ).items()
        )),
        "convolution_learning_rate_multiplier": float(
            payload["surface_training"].get(
                "convolution_learning_rate_multiplier",
                0.25,
            )
        ),
        "fork_provenance": (
            checkpoint_fork_provenance(payload, study, repository)
            if study is not None and repository is not None
            else payload.get("fork_provenance")
        ),
        "total_quantized_values": total_values,
        "results": results,
        "ranking": [
            {
                "rank": index + 1,
                "method": row["method"],
                "group_size": row["group_size"],
                "psnr_vs_ground_truth": row["metrics"][
                    "psnr_vs_ground_truth"
                ],
                "psnr_vs_teacher": row["metrics"]["psnr_vs_teacher"],
                "whole_model_bit_delta_vs_reference": row["storage"][
                    "whole_model_bit_delta_vs_reference"
                ],
            }
            for index, row in enumerate(ranked)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop size must be divisible by 16")
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    sealed_source_root = study_sealed_source_root(
        study,
        args.study_manifest,
        repository,
    )
    verified_sources = verify_study_sources(
        study,
        repository,
        sealed_source_root=sealed_source_root,
    )
    samples = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    teacher = pretrained(args.device).eval()
    report = {
        "format": "backdropkit-swin2sr-sensitive-io-group-probe-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "verified_sealed_sources": verified_sources,
        "probe_source_sha256": file_sha256(Path(__file__)),
        "signature_source_sha256": file_sha256(
            Path(__file__).with_name("analyze_surface_signatures.py")
        ),
        "device": args.device,
        "crop_size": args.crop_size,
        "validation_files": validation_receipts(
            args.validation_data,
            VALIDATION_FILES,
        ),
        **probe(
            args.checkpoint,
            args.target,
            args.group_sizes,
            teacher,
            samples,
            args.device,
            study,
            repository,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "surface_step": report["surface_step"],
        "target": report["target"],
        "top": report["ranking"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
