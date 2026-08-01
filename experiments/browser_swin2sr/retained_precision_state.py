"""Reusable loader for retained-precision Swin2SR training states."""

from __future__ import annotations

from pathlib import Path

import torch

from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from retained_precision_islands import ISLAND_SPECS, initialize_trainable_island


CHECKPOINT_FORMAT = "backdropkit-swin2sr-retained-precision-training-v1"


def load_retained_precision_student(
    parent: Path,
    island_name: str,
    precision: str,
    device: str,
    resume: Path | None = None,
    int7_scale_method: str = "max",
    int7_scale_blend: float = 1.0,
):
    parent_hash = file_sha256(parent)
    student, surfaces, parent_payload = load_checkpoint_student(parent, device)
    initialization = initialize_trainable_island(
        student,
        surfaces,
        ISLAND_SPECS[island_name],
        precision,
        int7_scale_method,
        int7_scale_blend,
    )
    resume_payload = None
    if resume is not None:
        resume_payload = torch.load(resume, map_location=device, weights_only=False)
        if resume_payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("resume checkpoint has the wrong format")
        if resume_payload["parent_checkpoint_sha256"] != parent_hash:
            raise ValueError("resume checkpoint belongs to a different parent")
        config = resume_payload["retained_fp16_training"]
        if config["island"] != island_name:
            raise ValueError("resume checkpoint belongs to a different island")
        if config["precision"] != precision:
            raise ValueError("resume checkpoint belongs to a different precision arm")
        student.load_state_dict(resume_payload["model"], strict=True)
    return student, surfaces, parent_payload, initialization, resume_payload
