#!/usr/bin/env python3
"""Fork an all-hard checkpoint by changing only its pixel-loss contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from fork_sensitive_io_checkpoint import state_tree_sha256
from full_surface_provenance import file_sha256
from qat_objective import PIXEL_OBJECTIVES
from run_state import atomic_torch_save


FORMAT = "backdropkit-swin2sr-pixel-objective-fork-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pixel-objective", choices=PIXEL_OBJECTIVES, required=True)
    parser.add_argument("--ground-truth-mse-floor", type=float, required=True)
    parser.add_argument("--teacher-mse-floor", type=float, required=True)
    parser.add_argument("--expect-surface-step", type=int, required=True)
    return parser.parse_args()


def immutable_state_hashes(payload: dict) -> dict[str, str]:
    return {
        "model": state_tree_sha256(payload["model"]),
        "optimizer": state_tree_sha256(payload["optimizer"]),
        "run_state": state_tree_sha256(payload["run_state"]),
        "quantization": state_tree_sha256(payload["quantization"]),
    }


def fork_payload(
    payload: dict,
    *,
    parent_checkpoint: Path,
    parent_sha256: str,
    pixel_objective: str,
    ground_truth_mse_floor: float,
    teacher_mse_floor: float,
    expect_surface_step: int,
) -> tuple[dict, dict]:
    if pixel_objective not in PIXEL_OBJECTIVES:
        raise ValueError(f"unknown pixel objective: {pixel_objective}")
    if ground_truth_mse_floor <= 0 or teacher_mse_floor <= 0:
        raise ValueError("pixel-objective MSE floors must be positive")
    if payload["quantization"].get("scope") != "all_matrix_and_conv":
        raise ValueError("checkpoint is not a full-surface checkpoint")
    if int(payload.get("surface_step", -1)) != expect_surface_step:
        raise ValueError("checkpoint is not at the expected surface step")
    if len(payload.get("quantized_modules", [])) != 204:
        raise ValueError("checkpoint does not cover all 204 surfaces")
    training = payload["surface_training"]
    previous = {
        "pixel_objective": training.get("pixel_objective", "charbonnier"),
        "ground_truth_mse_floor": float(
            training.get("ground_truth_mse_floor", 3.2e-4)
        ),
        "teacher_mse_floor": float(
            training.get("teacher_mse_floor", 2.0e-5)
        ),
    }
    before = immutable_state_hashes(payload)
    training["pixel_objective"] = pixel_objective
    training["ground_truth_mse_floor"] = ground_truth_mse_floor
    training["teacher_mse_floor"] = teacher_mse_floor
    payload["pixel_objective_fork"] = {
        "format": FORMAT,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_surface_step": expect_surface_step,
        "previous": previous,
        "selected": {
            "pixel_objective": pixel_objective,
            "ground_truth_mse_floor": ground_truth_mse_floor,
            "teacher_mse_floor": teacher_mse_floor,
        },
    }
    after = immutable_state_hashes(payload)
    if after != before:
        raise RuntimeError(
            "pixel-objective fork changed model, optimizer, sampler, or quantization"
        )
    return payload, {
        "surface_step": expect_surface_step,
        "immutable_state_sha256": after,
        "previous": previous,
        "selected": payload["pixel_objective_fork"]["selected"],
    }


def write_fork(
    checkpoint: Path,
    out: Path,
    receipt_path: Path,
    *,
    pixel_objective: str,
    ground_truth_mse_floor: float,
    teacher_mse_floor: float,
    expect_surface_step: int,
) -> dict:
    if out.exists() or receipt_path.exists():
        raise FileExistsError("output checkpoint or receipt already exists")
    parent_sha256 = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    forked, details = fork_payload(
        payload,
        parent_checkpoint=checkpoint,
        parent_sha256=parent_sha256,
        pixel_objective=pixel_objective,
        ground_truth_mse_floor=ground_truth_mse_floor,
        teacher_mse_floor=teacher_mse_floor,
        expect_surface_step=expect_surface_step,
    )
    atomic_torch_save(forked, out)
    reloaded = torch.load(out, map_location="cpu", weights_only=False)
    if immutable_state_hashes(reloaded) != details["immutable_state_sha256"]:
        raise RuntimeError("serialized fork changed immutable training state")
    receipt = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "fork_checkpoint": str(out),
        "fork_checkpoint_sha256": file_sha256(out),
        **details,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    args = parse_args()
    receipt = write_fork(
        args.checkpoint,
        args.out,
        args.receipt,
        pixel_objective=args.pixel_objective,
        ground_truth_mse_floor=args.ground_truth_mse_floor,
        teacher_mse_floor=args.teacher_mse_floor,
        expect_surface_step=args.expect_surface_step,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
