#!/usr/bin/env python3
"""Verify one materialized hard Swin2SR state across torch devices.

Checkpoint reprojection can choose different optimal-MSE supports when two
supports differ only at float32 rounding scale. Browser deployment does not
reproject masters: it consumes one materialized state. This diagnostic
materializes that state once on the training device, reloads the exact tensors
on both devices, and compares their outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_pipeline import fixed_validation
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from qat_objective import psnr
from surface_artifact_state import hard_surface_state_dict
from surface_model_state import pretrained
from train_swin2sr_qat import VALIDATION_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", default="mps")
    parser.add_argument("--comparison-device", default="cpu")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-absolute-tolerance", type=float, default=2e-4)
    parser.add_argument("--minimum-output-psnr-db", type=float, default=70.0)
    return parser.parse_args()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def compare(
    state: dict[str, torch.Tensor],
    samples,
    source_device: str,
    comparison_device: str,
) -> dict:
    source = pretrained(source_device).eval()
    comparison = pretrained(comparison_device).eval()
    source.load_state_dict(state, strict=True)
    comparison.load_state_dict(state, strict=True)
    output_psnr = []
    source_gt = []
    comparison_gt = []
    maximum = 0.0
    mean_absolute = []
    for low_cpu, high_cpu in samples:
        source_output = source(
            low_cpu.to(source_device)
        ).reconstruction.clamp(0, 1).cpu()
        comparison_output = comparison(
            low_cpu.to(comparison_device)
        ).reconstruction.clamp(0, 1).cpu()
        delta = source_output - comparison_output
        maximum = max(maximum, float(delta.abs().max()))
        mean_absolute.append(float(delta.abs().mean()))
        output_psnr.append(psnr(source_output, comparison_output))
        source_gt.append(psnr(source_output, high_cpu))
        comparison_gt.append(psnr(comparison_output, high_cpu))
    return {
        "samples": len(samples),
        "maximum_absolute_error": maximum,
        "mean_absolute_error": float(np.mean(mean_absolute)),
        "minimum_output_psnr_db": float(np.min(output_psnr)),
        "mean_output_psnr_db": float(np.mean(output_psnr)),
        "source_mean_gt_psnr_db": float(np.mean(source_gt)),
        "comparison_mean_gt_psnr_db": float(np.mean(comparison_gt)),
        "mean_gt_psnr_delta_db": float(
            np.mean(source_gt) - np.mean(comparison_gt)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    if args.max_absolute_tolerance <= 0:
        raise ValueError("maximum absolute tolerance must be positive")
    source, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        args.source_device,
    )
    state = hard_surface_state_dict(source, surfaces)
    samples = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        "cpu",
    )
    metrics = compare(
        state,
        samples,
        args.source_device,
        args.comparison_device,
    )
    passed = bool(
        metrics["maximum_absolute_error"] <= args.max_absolute_tolerance
        and metrics["minimum_output_psnr_db"]
        >= args.minimum_output_psnr_db
    )
    receipt = {
        "format": "backdropkit-swin2sr-materialized-device-parity-v1",
        "diagnostic_only": True,
        "interpretation": (
            "The checkpoint was projected once on source_device with every "
            "target surface hard. Both devices then consumed the same dense "
            "materialized tensors; no device reprojected continuous masters."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "source_device": args.source_device,
        "comparison_device": args.comparison_device,
        "materialized_state_sha256": state_sha256(state),
        "materialized_state_values": sum(
            tensor.numel() for tensor in state.values()
        ),
        "max_absolute_tolerance": args.max_absolute_tolerance,
        "minimum_output_psnr_requirement_db": args.minimum_output_psnr_db,
        "metrics": metrics,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
