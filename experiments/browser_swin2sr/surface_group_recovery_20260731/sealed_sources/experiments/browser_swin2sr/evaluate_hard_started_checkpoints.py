#!/usr/bin/env python3
"""Evaluate the staged hard branch trained by Swin2SR hard-shadow QAT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline import fixed_validation
from full_surface_provenance import file_sha256
from qat_objective import psnr
from surface_curriculum import enter_hard_shadow, restore_live_amounts
from surface_model_state import pretrained
from surface_quantization import (
    add_surface_parametrizations,
    apply_surface_group_sizes,
    set_surface_progress,
)
from train_swin2sr_qat import VALIDATION_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def verify_study_sources(
    study: dict,
    root: Path,
    sealed_source_root: Path | None = None,
) -> dict[str, str]:
    verified = {}
    for relative, expected in study["source_files"].items():
        path = root / relative
        actual = file_sha256(path)
        if actual != expected and sealed_source_root is not None:
            sealed_path = sealed_source_root / relative
            if sealed_path.exists():
                sealed_actual = file_sha256(sealed_path)
                if sealed_actual == expected:
                    actual = sealed_actual
        if actual != expected:
            raise RuntimeError(
                f"sealed source changed: {relative}: {actual} != {expected}"
            )
        verified[relative] = actual
    return verified


def validation_receipts(
    validation_data: Path,
    filenames: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        {
            "path": str(validation_data / filename),
            "sha256": file_sha256(validation_data / filename),
        }
        for filename in filenames
    ]


def manifest_path(value: str, repository: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def checkpoint_fork_provenance(
    payload: dict,
    study: dict,
    repository: Path,
) -> dict | None:
    fork_checkpoint_value = study.get("fork_checkpoint")
    if not fork_checkpoint_value:
        return payload.get("fork_provenance")
    fork_checkpoint = manifest_path(fork_checkpoint_value, repository)
    expected_checkpoint_hash = study["fork_checkpoint_sha256"]
    if file_sha256(fork_checkpoint) != expected_checkpoint_hash:
        raise RuntimeError("study fork checkpoint hash changed")
    resumed_from = payload["surface_training"].get("resume_checkpoint")
    if not resumed_from:
        raise RuntimeError("rescue checkpoint lost its resume checkpoint")
    resume_checkpoint = manifest_path(resumed_from, repository)
    if file_sha256(resume_checkpoint) != expected_checkpoint_hash:
        raise RuntimeError(
            "rescue checkpoint does not descend from its sealed fork"
        )
    fork_receipt = manifest_path(study["fork_receipt"], repository)
    expected_receipt_hash = study["fork_receipt_sha256"]
    if file_sha256(fork_receipt) != expected_receipt_hash:
        raise RuntimeError("study fork receipt hash changed")
    receipt = json.loads(fork_receipt.read_text())
    parent_hash = study["parent_checkpoint_sha256"]
    if receipt.get("parent_checkpoint_sha256") != parent_hash:
        raise RuntimeError("fork receipt parent does not match study manifest")
    existing = payload.get("fork_provenance") or {}
    existing_parent = existing.get("parent_checkpoint_sha256")
    if existing_parent not in (None, parent_hash):
        raise RuntimeError("checkpoint fork parent conflicts with study manifest")
    return {
        "parent_checkpoint_sha256": parent_hash,
        "fork_checkpoint": str(fork_checkpoint),
        "fork_checkpoint_sha256": expected_checkpoint_hash,
        "fork_receipt": str(fork_receipt),
        "fork_receipt_sha256": expected_receipt_hash,
        "training_state_sha256": study.get("training_state_sha256"),
        "recovered_from_sealed_study_manifest": True,
    }


def surface_amount_summary(surfaces) -> dict:
    categories: dict[str, dict[str, int]] = {}
    started = transitioning = fully_hard = 0
    for surface in surfaces:
        amount = float(surface.quantizer.amount.item())
        category = categories.setdefault(
            surface.category,
            {"surfaces": 0, "started": 0, "transitioning": 0, "fully_hard": 0},
        )
        category["surfaces"] += 1
        if amount > 0:
            started += 1
            category["started"] += 1
        if 0 < amount < 1:
            transitioning += 1
            category["transitioning"] += 1
        if amount >= 1:
            fully_hard += 1
            category["fully_hard"] += 1
    return {
        "surfaces": len(surfaces),
        "started_surfaces": started,
        "transitioning_surfaces": transitioning,
        "fully_hard_surfaces": fully_hard,
        "categories": categories,
    }


def restore_checkpoint_schedule(surfaces, payload: dict) -> float:
    training = payload["surface_training"]
    schedule_steps = int(training["schedule_steps"])
    if schedule_steps <= 0:
        raise ValueError("checkpoint schedule_steps must be positive")
    progress = min(1.0, int(payload["surface_step"]) / schedule_steps)
    set_surface_progress(
        surfaces,
        progress,
        hard_core=bool(training.get("hard_core_from_start", False)),
    )
    return progress


@torch.inference_mode()
def evaluate(student, teacher, samples) -> dict[str, float]:
    ground_truth = []
    teacher_scores = []
    for low, high in samples:
        teacher_output = teacher(low).reconstruction.clamp(0, 1)
        student_output = student(low).reconstruction.clamp(0, 1)
        ground_truth.append(psnr(student_output, high))
        teacher_scores.append(psnr(student_output, teacher_output))
    return {
        "psnr_vs_ground_truth": float(np.mean(ground_truth)),
        "psnr_vs_teacher": float(np.mean(teacher_scores)),
    }


def load_checkpoint_student(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location=device)
    quantization = payload["quantization"]
    student = pretrained(device)
    surfaces = add_surface_parametrizations(
        student,
        scope="all_matrix_and_conv",
        group_size=int(quantization["group_size"]),
        method=quantization["method"],
        core_rotation=quantization.get("rotation", "none"),
        rotation_block=int(quantization.get("rotation_block", 4)),
        conv_method=quantization.get("conv_method"),
        conv_grouping=quantization.get("conv_grouping", "flat"),
        schedule_profile=quantization.get("schedule_profile", "joint_io_v1"),
    )
    apply_surface_group_sizes(
        surfaces,
        quantization.get("surface_group_sizes"),
    )
    student.load_state_dict(payload["model"], strict=True)
    restore_checkpoint_schedule(surfaces, payload)
    return student.eval(), surfaces, payload


def evaluate_checkpoint(
    checkpoint: Path,
    teacher,
    samples,
    device: str,
    study: dict,
    repository: Path,
) -> dict:
    student, surfaces, payload = load_checkpoint_student(checkpoint, device)
    amounts = surface_amount_summary(surfaces)
    blended = evaluate(student, teacher, samples)
    live_amounts = enter_hard_shadow(surfaces)
    try:
        hard_started = evaluate(student, teacher, samples)
    finally:
        restore_live_amounts(surfaces, live_amounts)
    return {
        "surface_step": int(payload["surface_step"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "surface_group_sizes": dict(
            sorted((
                payload["quantization"].get("surface_group_sizes", {}) or {}
            ).items())
        ),
        "convolution_learning_rate_multiplier": float(
            payload["surface_training"].get(
                "convolution_learning_rate_multiplier",
                0.25,
            )
        ),
        "fork_provenance": checkpoint_fork_provenance(
            payload,
            study,
            repository,
        ),
        **amounts,
        "blended": blended,
        "hard_started": hard_started,
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    configured_sealed_root = study.get("sealed_source_root")
    sealed_source_root = (
        Path(configured_sealed_root)
        if configured_sealed_root
        else args.study_manifest.parent / "sealed_sources"
    )
    if not sealed_source_root.is_absolute():
        sealed_source_root = repository / sealed_source_root
    verified_sources = verify_study_sources(
        study,
        repository,
        sealed_source_root=sealed_source_root,
    )
    validation_files = validation_receipts(
        args.validation_data,
        VALIDATION_FILES,
    )
    samples = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    teacher = pretrained(args.device).eval()
    checkpoints = sorted([
        evaluate_checkpoint(
            checkpoint,
            teacher,
            samples,
            args.device,
            study,
            repository,
        )
        for checkpoint in args.checkpoints
    ], key=lambda checkpoint: checkpoint["surface_step"])
    steps = [checkpoint["surface_step"] for checkpoint in checkpoints]
    if steps != sorted(set(steps)):
        raise RuntimeError(f"checkpoint steps must be unique and ordered: {steps}")
    receipt = {
        "format": "backdropkit-swin2sr-hard-started-validation-v2",
        "diagnostic_only": True,
        "interpretation": (
            "Every scheduled surface that had started was forced fully hard; "
            "future surfaces remained dense. This is the branch optimized by "
            "hard-shadow loss, not a final all-surface promotion evaluation."
        ),
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "sealed_source_root": str(sealed_source_root),
        "verified_sealed_sources": verified_sources,
        "device": args.device,
        "crop_size": args.crop_size,
        "validation_files": validation_files,
        "checkpoints": checkpoints,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "steps": steps,
        "latest": checkpoints[-1],
    }, indent=2))


if __name__ == "__main__":
    main()
