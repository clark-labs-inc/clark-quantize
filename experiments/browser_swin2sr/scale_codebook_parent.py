"""Parent-state contracts for scale-codebook QAT transfer."""

from __future__ import annotations

from pathlib import Path

import torch

from evaluate_hard_started_checkpoints import load_checkpoint_student
from retained_precision_state import (
    CHECKPOINT_FORMAT as RETAINED_PRECISION_FORMAT,
    load_retained_precision_student,
)


def load_scale_codebook_parent(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != RETAINED_PRECISION_FORMAT:
        return load_checkpoint_student(path, device)
    config = payload["retained_fp16_training"]
    if config["precision"] != "ternary":
        raise ValueError("scale-codebook transfer requires an all-ternary parent")
    parent = Path(payload["parent_checkpoint"])
    student, surfaces, _base_payload, _initialization, loaded = (
        load_retained_precision_student(
            parent,
            config["island"],
            config["precision"],
            device,
            path,
        )
    )
    if loaded.get("format") != RETAINED_PRECISION_FORMAT:
        raise RuntimeError("retained-precision parent changed while loading")
    return student, surfaces, loaded


def scale_parent_training(payload: dict) -> dict:
    training = dict(payload["surface_training"])
    if payload.get("format") == RETAINED_PRECISION_FORMAT:
        retained = payload["retained_fp16_training"]
        training.update({
            "seed": int(retained["seed"]),
            "batch_size": int(retained["batch_size"]),
            "crop_size": int(retained["crop_size"]),
            "loss_recipe": retained["loss_recipe"],
            "pixel_objective": retained["pixel_objective"],
        })
    return training


def scale_parent_sampler_step(payload: dict) -> int:
    if payload.get("format") == RETAINED_PRECISION_FORMAT:
        return int(payload["island_step"])
    return int(payload["surface_step"])


def scale_parent_total_step(payload: dict) -> int:
    if "step" in payload:
        return int(payload["step"])
    return int(payload["surface_step"])
