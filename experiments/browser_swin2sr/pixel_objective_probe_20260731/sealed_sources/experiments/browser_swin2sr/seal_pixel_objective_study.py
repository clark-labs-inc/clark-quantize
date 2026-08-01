#!/usr/bin/env python3
"""Seal a matched all-hard Charbonnier versus PSNR-aligned probe."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

from fork_pixel_objective_checkpoint import write_fork
from full_surface_provenance import file_sha256
from seal_optimizer_transfer_study import SOURCE_FILES as TRAINING_SOURCE_FILES
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-pixel-objective-study-v1"
ARMS = {
    "charbonnier": "charbonnier",
    "psnr_aligned": "psnr_aligned",
}
SOURCE_FILES = tuple(dict.fromkeys((
    *TRAINING_SOURCE_FILES,
    "experiments/browser_swin2sr/decide_pixel_objective.py",
    "experiments/browser_swin2sr/fork_sensitive_io_checkpoint.py",
    "experiments/browser_swin2sr/fork_pixel_objective_checkpoint.py",
    "experiments/browser_swin2sr/seal_pixel_objective_study.py",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-study", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=26000)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def parent_contract(
    checkpoint: Path,
    report: dict,
    study: dict,
) -> tuple[dict, dict[str, int]]:
    if int(report.get("surface_step", -1)) != 24000:
        raise RuntimeError("parent report is not selected step 24000")
    if report.get("selection_source") != "development":
        raise RuntimeError("parent report is not development evidence")
    if report.get("model_promotion_authorized"):
        raise RuntimeError("parent report improperly authorizes promotion")
    if not report.get("final_heldout_locked"):
        raise RuntimeError("parent report does not lock final heldout")
    if file_sha256(checkpoint) != report["state_sha256"]:
        raise RuntimeError("parent report uses another checkpoint")
    if report["study_manifest_sha256"] != file_sha256(
        Path(report["study_manifest"])
    ):
        raise RuntimeError("parent report study identity changed")
    if Path(report["study_manifest"]).resolve() != (
        Path(study["_path"]).resolve()
    ):
        raise RuntimeError("parent report uses another study")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if len(payload.get("quantized_modules", [])) != 204:
        raise RuntimeError("parent checkpoint does not cover 204 surfaces")
    training = payload["surface_training"]
    quantization = payload["quantization"]
    if training.get("loss_recipe") != "balanced":
        raise RuntimeError("parent checkpoint does not use balanced loss")
    if training.get("pixel_objective", "charbonnier") != "charbonnier":
        raise RuntimeError("parent checkpoint is not the Charbonnier state")
    surface_map = {
        name: int(size)
        for name, size in (
            quantization.get("surface_group_sizes") or {}
        ).items()
    }
    config = {
        "group_size": int(quantization["group_size"]),
        "quantizer_method": quantization["method"],
        "conv_quantizer_method": quantization["conv_method"],
        "conv_grouping": quantization["conv_grouping"],
        "schedule_profile": quantization["schedule_profile"],
        "core_rotation": quantization["rotation"],
        "schedule_steps": int(training["schedule_steps"]),
        "batch_size": int(training["batch_size"]),
        "crop_size": int(training["crop_size"]),
        "learning_rate": float(training["optimizer"]["learning_rate"]),
        "convolution_learning_rate_multiplier": float(
            training["convolution_learning_rate_multiplier"]
        ),
        "correction_learning_rate_multiplier": float(
            training.get("correction_learning_rate_multiplier", 0.1)
        ),
        "loss_recipe": training["loss_recipe"],
        "seed": int(training["seed"]),
        "hard_shadow_weight_start": float(
            training["hard_shadow_weight_start"]
        ),
        "hard_shadow_weight_end": float(training["hard_shadow_weight_end"]),
        "hard_shadow_every": int(training["hard_shadow_every"]),
    }
    if config["schedule_steps"] != 24000:
        raise RuntimeError("parent checkpoint uses another surface schedule")
    return config, surface_map


def sealed_command(
    sealed_root: Path,
    script: str,
    *arguments: str,
) -> list[str]:
    browser = sealed_root / "experiments/browser_swin2sr"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={browser}:{sealed_root}",
        sys.executable,
        str(sealed_root / script),
        *arguments,
    ]


def training_command(
    args: argparse.Namespace,
    sealed_root: Path,
    config: dict,
    surface_map: dict[str, int],
    arm: str,
    fork: Path,
    training_roots: list[Path],
) -> list[str]:
    command = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/train_swin2sr_full_surface.py",
        "--base-checkpoint", str(fork),
        "--validation-data", str(args.validation_data),
        "--out", str(args.out / arm / "run"),
        "--steps", str(args.steps),
        "--schedule-steps", str(config["schedule_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--crop-size", str(config["crop_size"]),
        "--group-size", str(config["group_size"]),
        "--quantizer-method", config["quantizer_method"],
        "--conv-quantizer-method", config["conv_quantizer_method"],
        "--conv-grouping", config["conv_grouping"],
        "--schedule-profile", config["schedule_profile"],
        "--core-rotation", config["core_rotation"],
        "--optimizer", "bonsai_qat8_muon",
        "--learning-rate", str(config["learning_rate"]),
        "--convolution-learning-rate-multiplier",
        str(config["convolution_learning_rate_multiplier"]),
        "--correction-learning-rate-multiplier",
        str(config["correction_learning_rate_multiplier"]),
        "--loss-recipe", config["loss_recipe"],
        "--pixel-objective", ARMS[arm],
        "--ground-truth-mse-floor", "0.00032",
        "--teacher-mse-floor", "0.00002",
        "--validate-every", "500",
        "--checkpoint-every", "1000",
        "--hard-shadow-weight-start",
        str(config["hard_shadow_weight_start"]),
        "--hard-shadow-weight-end",
        str(config["hard_shadow_weight_end"]),
        "--hard-shadow-every", str(config["hard_shadow_every"]),
        "--device", args.device,
        "--seed", str(config["seed"]),
    )
    for name, size in sorted(surface_map.items()):
        command.extend(("--surface-group-size", f"{name}={size}"))
    for root in training_roots:
        command.extend(("--data", str(root)))
    return command


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"pixel-objective study exists: {args.out}")
    if args.steps != 26000 or args.device != "mps":
        raise ValueError("pixel-objective study is sealed at step 26000 on MPS")
    repository = Path(__file__).resolve().parents[2]
    report = json.loads(args.parent_report.read_text())
    study = json.loads(args.parent_study.read_text())
    study["_path"] = str(args.parent_study)
    config, surface_map = parent_contract(
        args.parent_checkpoint,
        report,
        study,
    )
    training_roots = [
        Path(record["root"])
        for record in study["training_data"]
    ]
    if dataset_fingerprints(training_roots) != study["training_data"]:
        raise RuntimeError("training corpus identities changed")
    bundle = Path(study["bundle"])
    q4_reference = Path(study["q4_reference"])
    if file_sha256(bundle) != study["bundle_sha256"]:
        raise RuntimeError("development bundle changed")
    if file_sha256(q4_reference) != study["q4_reference_sha256"]:
        raise RuntimeError("q4 reference changed")
    bundle_record = json.loads(bundle.read_text())

    args.out.mkdir(parents=True)
    sealed_root = args.out / "sealed_sources"
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = repository / relative
        destination = sealed_root / relative
        source_hashes[relative] = file_sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    arm_records = {}
    for arm, objective in ARMS.items():
        root = args.out / arm
        fork = root / "fork-step24000.pt"
        fork_receipt = root / "fork-receipt.json"
        receipt = write_fork(
            args.parent_checkpoint,
            fork,
            fork_receipt,
            pixel_objective=objective,
            ground_truth_mse_floor=3.2e-4,
            teacher_mse_floor=2e-5,
            expect_surface_step=24000,
        )
        training = training_command(
            args,
            sealed_root,
            config,
            surface_map,
            arm,
            fork,
            training_roots,
        )
        checkpoint = root / "run/checkpoint-final.pt"
        materialization = root / "canonical/materialization.json"
        materialize = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/materialize_surface_states.py",
            str(checkpoint),
            "--out", str(root / "canonical"),
            "--source-device", args.device,
        )
        evaluate = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/evaluate_stratified.py",
            str(root / "canonical/dense-hard.pt"),
            "--bundle", str(bundle),
            "--out", str(root / "development"),
            "--crop-size", "128",
            "--device", args.device,
        )
        arm_records[arm] = {
            "pixel_objective": objective,
            "ground_truth_mse_floor": 3.2e-4,
            "teacher_mse_floor": 2e-5,
            "fork_checkpoint": str(fork),
            "fork_checkpoint_sha256": receipt["fork_checkpoint_sha256"],
            "fork_receipt": str(fork_receipt),
            "fork_receipt_sha256": file_sha256(fork_receipt),
            "training_command": training,
            "final_checkpoint": str(checkpoint),
            "materialization_command": materialize,
            "materialization": str(materialization),
            "evaluation_command": evaluate,
            "development_report": str(root / "development/metrics.json"),
        }
        (root / "commands.json").write_text(json.dumps({
            "training": training,
            "materialization": materialize,
            "evaluation": evaluate,
        }, indent=2) + "\n")

    manifest_path = args.out / "study-manifest.json"
    decision_command = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/decide_pixel_objective.py",
        "--study-manifest", str(manifest_path),
        "--control", arm_records["charbonnier"]["development_report"],
        "--candidate", arm_records["psnr_aligned"]["development_report"],
        "--out", str(args.out / "pixel-objective-decision.json"),
    )
    manifest = {
        "format": FORMAT,
        "created_at_unix": time.time(),
        "selection_source": "development",
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_checkpoint_sha256": file_sha256(args.parent_checkpoint),
        "parent_report": str(args.parent_report),
        "parent_report_sha256": file_sha256(args.parent_report),
        "parent_study": str(args.parent_study),
        "parent_study_sha256": file_sha256(args.parent_study),
        "surface_group_sizes": surface_map,
        "config": config,
        "training_data": study["training_data"],
        "training_data_receipts": study["training_data_receipts"],
        "bundle": str(bundle),
        "bundle_sha256": file_sha256(bundle),
        "bundle_manifest_sha256": bundle_record["manifest_sha256"],
        "q4_reference": str(q4_reference),
        "q4_reference_sha256": file_sha256(q4_reference),
        "source_files": source_hashes,
        "sealed_source_root": str(sealed_root),
        "arms": arm_records,
        "selection_step": 26000,
        "decision_rule": {
            "bootstrap_seed": 20260731,
            "bootstrap_samples": 10000,
            "ground_truth_lower_95_min_db": 0.0,
            "edge_mean_min_db": -0.1,
            "teacher_mean_min_db": -0.1,
            "every_source_mean_min_db": -0.1,
        },
        "decision_command": decision_command,
        "single_variable": (
            "The forks preserve model, optimizer, sampler/run state, and "
            "quantization hashes. Only the pixel objective differs."
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "decision-command.json").write_text(
        json.dumps(decision_command, indent=2) + "\n"
    )
    print(json.dumps({
        "out": str(manifest_path),
        "arms": sorted(arm_records),
        "selection_step": 26000,
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
