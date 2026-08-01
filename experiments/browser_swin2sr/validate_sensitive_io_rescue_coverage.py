#!/usr/bin/env python3
"""Fail closed unless the step gate contains every preregistered input arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decide_sensitive_io_rescue import (
    TARGET_SURFACE,
    load_receipt,
    sha256,
    validation_identity,
)


EXPECTED_MULTIPLIERS = {0.25, 1.0, 4.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=19000)
    return parser.parse_args()


def validate_coverage(
    control_path: Path,
    arm_paths: list[Path],
    *,
    step: int,
) -> dict:
    if len(arm_paths) != len(EXPECTED_MULTIPLIERS):
        raise RuntimeError("coverage gate requires exactly three rescue arms")
    control_receipt, control = load_receipt(control_path, step)
    identity = validation_identity(control_receipt)
    if control.get("surface_group_sizes") not in (
        {},
        {TARGET_SURFACE: 9},
    ):
        raise RuntimeError("coverage control is not group 9")
    if float(control["convolution_learning_rate_multiplier"]) != 0.25:
        raise RuntimeError("coverage control is not the matched 0.25x arm")
    control_parent = (
        control.get("fork_provenance") or {}
    ).get("parent_checkpoint_sha256")
    if not control_parent:
        raise RuntimeError("coverage control has no shared parent")
    arms = []
    seen_multipliers = set()
    seen_studies = {control_receipt["study_manifest_sha256"]}
    for path in arm_paths:
        receipt, checkpoint = load_receipt(path, step)
        if validation_identity(receipt) != identity:
            raise RuntimeError("coverage validation protocol differs")
        if checkpoint.get("surface_group_sizes") != {TARGET_SURFACE: 2}:
            raise RuntimeError("coverage arm is not group 2")
        parent = (
            checkpoint.get("fork_provenance") or {}
        ).get("parent_checkpoint_sha256")
        if parent != control_parent:
            raise RuntimeError("coverage arms do not share one parent")
        multiplier = float(
            checkpoint["convolution_learning_rate_multiplier"]
        )
        if multiplier in seen_multipliers:
            raise RuntimeError("coverage contains a duplicate LR arm")
        seen_multipliers.add(multiplier)
        study_hash = receipt["study_manifest_sha256"]
        if study_hash in seen_studies:
            raise RuntimeError("coverage receipts reuse a study manifest")
        seen_studies.add(study_hash)
        arms.append({
            "receipt": str(path),
            "receipt_sha256": sha256(path),
            "study_manifest": receipt["study_manifest"],
            "study_manifest_sha256": study_hash,
            "checkpoint": checkpoint["checkpoint"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "convolution_learning_rate_multiplier": multiplier,
        })
    if seen_multipliers != EXPECTED_MULTIPLIERS:
        raise RuntimeError("coverage does not contain 0.25x, 1x, and 4x arms")
    source = Path(__file__)
    selector = source.with_name("decide_sensitive_io_rescue.py")
    return {
        "format": "backdropkit-swin2sr-sensitive-io-coverage-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "surface_step": step,
        "validation_identity": identity,
        "shared_fork_parent_sha256": control_parent,
        "control": {
            "receipt": str(control_path),
            "receipt_sha256": sha256(control_path),
            "study_manifest": control_receipt["study_manifest"],
            "study_manifest_sha256": (
                control_receipt["study_manifest_sha256"]
            ),
            "checkpoint": control["checkpoint"],
            "checkpoint_sha256": control["checkpoint_sha256"],
            "convolution_learning_rate_multiplier": 0.25,
        },
        "arms": sorted(
            arms,
            key=lambda arm: arm["convolution_learning_rate_multiplier"],
        ),
        "validator_sha256": sha256(source),
        "sealed_selector_sha256": sha256(selector),
        "exact_coverage_verified": True,
        "final_selection_authorized": False,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }


def main() -> None:
    args = parse_args()
    receipt = validate_coverage(
        args.control,
        args.arms,
        step=args.step,
    )
    if args.out.exists():
        raise FileExistsError(f"coverage receipt already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "surface_step": receipt["surface_step"],
        "arms": len(receipt["arms"]),
        "exact_coverage_verified": True,
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
