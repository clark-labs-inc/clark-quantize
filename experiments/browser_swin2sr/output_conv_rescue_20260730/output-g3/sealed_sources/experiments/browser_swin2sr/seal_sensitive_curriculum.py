#!/usr/bin/env python3
"""Seal the development-only Swin2SR sensitive ternary curriculum."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from full_surface_cli import (
    normalize_surface_group_sizes,
    surface_group_size,
)
from full_surface_optimizer import FULL_SURFACE_OPTIMIZERS
from full_surface_provenance import file_sha256
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION
from training_data_contract import dataset_fingerprints, validation_receipts


SOURCE_FILES = (
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_qat8.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/full_surface_cli.py",
    "experiments/browser_swin2sr/full_surface_optimizer.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/recovery_controls.py",
    "experiments/browser_swin2sr/recovery_data.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/surface_artifact_state.py",
    "experiments/browser_swin2sr/surface_curriculum.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
    "experiments/browser_swin2sr/surface_run_state.py",
    "experiments/browser_swin2sr/surface_schedule.py",
    "experiments/browser_swin2sr/train_swin2sr_full_surface.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--data-receipt", type=Path, action="append", default=[])
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--schedule-steps", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument(
        "--surface-group-size",
        action="append",
        type=surface_group_size,
    )
    parser.add_argument("--quantizer-method", default="absmean")
    parser.add_argument("--conv-quantizer-method", default="optimal_mse")
    parser.add_argument("--conv-grouping", default="kernel")
    parser.add_argument("--schedule-profile", default="sensitive_io_v2")
    parser.add_argument("--core-rotation", default="hadamard")
    parser.add_argument(
        "--optimizer",
        choices=FULL_SURFACE_OPTIMIZERS,
        default="bonsai_qat8_muon",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--convolution-learning-rate-multiplier",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--correction-learning-rate-multiplier",
        type=float,
        default=0.10,
    )
    parser.add_argument("--loss-recipe", default="balanced")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.05)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.25)
    parser.add_argument("--hard-shadow-every", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    args.surface_group_sizes = normalize_surface_group_sizes(
        args.surface_group_size
    ) or {}
    del args.surface_group_size
    return args


def verify_development_contract(
    bundle_path: Path,
    q4_path: Path,
    device: str,
) -> tuple[dict, dict]:
    bundle = json.loads(bundle_path.read_text())
    q4 = json.loads(q4_path.read_text())
    if bundle.get("manifest_sha256") != q4.get("bundle_manifest_sha256"):
        raise ValueError("development bundle does not match q4 reference")
    if int(bundle["samples"]) != len(q4.get("samples", [])):
        raise ValueError("development bundle and q4 sample counts differ")
    if device != q4.get("device"):
        raise ValueError("training/evaluation device does not match q4 reference")
    return bundle, q4


def training_command(args: argparse.Namespace) -> list[str]:
    script = Path(__file__).with_name("train_swin2sr_full_surface.py")
    command = [
        sys.executable,
        str(script),
        "--validation-data", str(args.validation_data),
        "--out", str(args.out),
        "--steps", str(args.steps),
        "--schedule-steps", str(args.schedule_steps),
        "--batch-size", str(args.batch_size),
        "--crop-size", str(args.crop_size),
        "--group-size", str(args.group_size),
        "--quantizer-method", args.quantizer_method,
        "--conv-quantizer-method", args.conv_quantizer_method,
        "--conv-grouping", args.conv_grouping,
        "--schedule-profile", args.schedule_profile,
        "--core-rotation", args.core_rotation,
        "--optimizer", args.optimizer,
        "--learning-rate", str(args.learning_rate),
        "--convolution-learning-rate-multiplier",
        str(args.convolution_learning_rate_multiplier),
        "--correction-learning-rate-multiplier",
        str(args.correction_learning_rate_multiplier),
        "--loss-recipe", args.loss_recipe,
        "--validate-every", str(args.validate_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--hard-shadow-weight-start", str(args.hard_shadow_weight_start),
        "--hard-shadow-weight-end", str(args.hard_shadow_weight_end),
        "--hard-shadow-every", str(args.hard_shadow_every),
        "--device", args.device,
        "--seed", str(args.seed),
        "--initialization", "fp32",
    ]
    for name, group_size in sorted(args.surface_group_sizes.items()):
        command.extend(("--surface-group-size", f"{name}={group_size}"))
    for root in args.data:
        command.extend(("--data", str(root)))
    return command


def main() -> None:
    args = parse_args()
    bundle, q4 = verify_development_contract(
        args.bundle,
        args.q4_reference,
        args.device,
    )
    if args.out.exists():
        raise FileExistsError(f"sealed output already exists: {args.out}")
    repo = Path(__file__).resolve().parents[2]
    datasets = dataset_fingerprints(args.data)
    command = training_command(args)
    manifest = {
        "format": "backdropkit-swin2sr-sensitive-curriculum-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "teacher": {"model": MODEL_ID, "revision": MODEL_REVISION},
        "initialization": "fp32",
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "q4_reference_state_sha256": q4["state_sha256"],
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "training_data": datasets,
        "training_data_receipts": validation_receipts(
            args.data_receipt,
            datasets,
        ),
        "source_files": {
            name: file_sha256(repo / name)
            for name in SOURCE_FILES
        },
        "training_command": command,
        "acceptance": {
            "quality": "statistically non-inferior or better than shipped q4",
            "size": "materially smaller than shipped q4",
            "runtime": "WebGPU, WASM, and CPU viable",
            "final_heldout_locked": True,
        },
    }
    args.out.mkdir(parents=True)
    (args.out / "study-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.out / "command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    print(json.dumps({
        "manifest": str(args.out / "study-manifest.json"),
        "command": command,
    }, indent=2))


if __name__ == "__main__":
    main()
