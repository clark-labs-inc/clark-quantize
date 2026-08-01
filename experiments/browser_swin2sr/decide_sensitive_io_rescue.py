#!/usr/bin/env python3
"""Apply the preregistered checkpoint gate to sensitive-I/O rescue arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TARGET_SURFACE = "swin2sr.first_convolution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=19000)
    parser.add_argument("--margin-db", type=float, default=0.10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path: Path, step: int) -> tuple[dict, dict]:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != "backdropkit-swin2sr-hard-started-validation-v2":
        raise RuntimeError(f"unsupported hard-started receipt: {path}")
    if not receipt.get("diagnostic_only"):
        raise RuntimeError(f"receipt is not development-only: {path}")
    checkpoints = receipt.get("checkpoints") or []
    matching = [
        checkpoint
        for checkpoint in checkpoints
        if int(checkpoint["surface_step"]) == step
    ]
    if len(matching) != 1:
        raise RuntimeError(f"receipt must contain step {step} exactly once: {path}")
    return receipt, matching[0]


def validation_identity(receipt: dict) -> dict:
    return {
        "device": receipt["device"],
        "crop_size": int(receipt["crop_size"]),
        "validation_files": [
            (item["path"], item["sha256"])
            for item in receipt["validation_files"]
        ],
    }


def decide(
    control_path: Path,
    arm_paths: list[Path],
    *,
    step: int,
    margin_db: float,
) -> dict:
    control_receipt, control = load_receipt(control_path, step)
    identity = validation_identity(control_receipt)
    if control.get("surface_group_sizes") not in (
        {},
        {TARGET_SURFACE: 9},
    ):
        raise RuntimeError("control unexpectedly has a surface override")
    loaded_arms = []
    control_fork = control.get("fork_provenance") or {}
    parent_hash = control_fork.get("parent_checkpoint_sha256")
    multipliers = set()
    for path in arm_paths:
        receipt, checkpoint = load_receipt(path, step)
        if validation_identity(receipt) != identity:
            raise RuntimeError(f"validation protocol mismatch: {path}")
        if checkpoint.get("surface_group_sizes") != {TARGET_SURFACE: 2}:
            raise RuntimeError(f"arm is not the preregistered group-2 rescue: {path}")
        fork = checkpoint.get("fork_provenance") or {}
        current_parent = fork.get("parent_checkpoint_sha256")
        if not current_parent:
            raise RuntimeError(f"arm has no fork-parent identity: {path}")
        if parent_hash is None:
            parent_hash = current_parent
        elif current_parent != parent_hash:
            raise RuntimeError("rescue arms do not share one parent checkpoint")
        multiplier = float(checkpoint["convolution_learning_rate_multiplier"])
        if multiplier in multipliers:
            raise RuntimeError(f"duplicate convolution LR multiplier: {multiplier}")
        multipliers.add(multiplier)
        blended = checkpoint["blended"]
        hard = checkpoint["hard_started"]
        loaded_arms.append({
            "receipt": str(path),
            "receipt_sha256": sha256(path),
            "checkpoint": checkpoint["checkpoint"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "convolution_learning_rate_multiplier": multiplier,
            "blended_ground_truth_db": float(
                blended["psnr_vs_ground_truth"]
            ),
            "hard_ground_truth_db": float(
                hard["psnr_vs_ground_truth"]
            ),
            "hard_teacher_db": float(hard["psnr_vs_teacher"]),
        })
    if not loaded_arms:
        raise RuntimeError("no rescue arms were supplied")

    control_blended = float(control["blended"]["psnr_vs_ground_truth"])
    control_hard_gt = float(control["hard_started"]["psnr_vs_ground_truth"])
    control_hard_teacher = float(control["hard_started"]["psnr_vs_teacher"])
    best_hard_gt = max(arm["hard_ground_truth_db"] for arm in loaded_arms)
    for arm in loaded_arms:
        arm["beats_control_hard_ground_truth"] = (
            arm["hard_ground_truth_db"] > control_hard_gt
        )
        arm["beats_control_hard_teacher"] = (
            arm["hard_teacher_db"] > control_hard_teacher
        )
        arm["blended_within_margin"] = (
            arm["blended_ground_truth_db"] >= control_blended - margin_db
        )
        arm["within_margin_of_best_hard_ground_truth"] = (
            arm["hard_ground_truth_db"] >= best_hard_gt - margin_db
        )
        arm["checkpoint_eligible"] = all((
            arm["beats_control_hard_ground_truth"],
            arm["beats_control_hard_teacher"],
            arm["blended_within_margin"],
            arm["within_margin_of_best_hard_ground_truth"],
        ))
    shortlist = sorted(
        (arm for arm in loaded_arms if arm["checkpoint_eligible"]),
        key=lambda arm: (
            arm["convolution_learning_rate_multiplier"],
            -arm["hard_ground_truth_db"],
        ),
    )
    return {
        "format": "backdropkit-swin2sr-sensitive-io-checkpoint-decision-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "surface_step": step,
        "margin_db": margin_db,
        "validation_identity": identity,
        "shared_fork_parent_sha256": parent_hash,
        "control": {
            "receipt": str(control_path),
            "receipt_sha256": sha256(control_path),
            "checkpoint": control["checkpoint"],
            "checkpoint_sha256": control["checkpoint_sha256"],
            "surface_group_sizes": control["surface_group_sizes"],
            "fork_provenance": control.get("fork_provenance"),
            "blended_ground_truth_db": control_blended,
            "hard_ground_truth_db": control_hard_gt,
            "hard_teacher_db": control_hard_teacher,
        },
        "arms": sorted(
            loaded_arms,
            key=lambda arm: arm["convolution_learning_rate_multiplier"],
        ),
        "checkpoint_shortlist": [
            arm["convolution_learning_rate_multiplier"]
            for arm in shortlist
        ],
        "provisional_smallest_lr": (
            shortlist[0]["convolution_learning_rate_multiplier"]
            if shortlist
            else None
        ),
        "final_selection_authorized": False,
        "required_next_gate": (
            "Evaluate the best two checkpoint-eligible arms on the 3,040-sample "
            "development bundle and enforce paired per-source deltas."
        ),
    }


def main() -> None:
    args = parse_args()
    receipt = decide(
        args.control,
        args.arms,
        step=args.step,
        margin_db=args.margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "checkpoint_shortlist": receipt["checkpoint_shortlist"],
        "final_selection_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
