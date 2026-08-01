#!/usr/bin/env python3
"""Seal one matched output-convolution rescue arm before training."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_sensitive_curriculum import SOURCE_FILES
from seal_sensitive_io_rescue import (
    training_command,
    verify_parent_and_fork,
)


TARGET_SURFACE = "upsample.conv"
ARM_GROUPS = {
    "output-g9": 9,
    "output-g3": 3,
    "output-g2": 2,
}
EXTRA_SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/fork_sensitive_io_checkpoint.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/island_policy.py",
    "experiments/browser_swin2sr/recovery_corpus_identity.py",
    "experiments/browser_swin2sr/recovery_resume_policy.py",
    "experiments/browser_swin2sr/seal_sensitive_curriculum.py",
    "experiments/browser_swin2sr/seal_sensitive_io_rescue.py",
    "experiments/browser_swin2sr/seal_output_conv_rescue.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/training_data_contract.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--input-decision", type=Path, required=True)
    parser.add_argument("--fork-checkpoint", type=Path, required=True)
    parser.add_argument("--fork-receipt", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_GROUPS), required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--schedule-steps", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--loss-recipe", default="balanced")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.05)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.25)
    parser.add_argument("--hard-shadow-every", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def verify_selected_parent(
    receipt: dict,
    decision: dict,
    *,
    arm: str,
) -> tuple[int, dict[str, int]]:
    if (
        decision.get("format")
        != "backdropkit-swin2sr-sensitive-io-development-decision-v1"
    ):
        raise RuntimeError("unsupported sensitive-input decision")
    if not decision.get("sensitive_recipe_selected"):
        raise RuntimeError("sensitive-input decision selected no recipe")
    if decision.get("model_promotion_authorized"):
        raise RuntimeError("input decision improperly authorizes promotion")
    if not decision.get("final_heldout_locked"):
        raise RuntimeError("input decision does not lock final heldout")
    if (
        receipt.get("parent_checkpoint_sha256")
        != decision.get("selected_checkpoint_sha256")
    ):
        raise RuntimeError("output fork does not descend from selected input")
    if int(receipt.get("parent_surface_step", -1)) != 19000:
        raise RuntimeError("output fork parent is not at step 19000")
    change = (
        (receipt.get("fork_provenance") or {}).get("change")
        or receipt.get("change")
        or {}
    )
    if change.get("surface") != TARGET_SURFACE:
        raise RuntimeError("fork does not target the output convolution")
    expected_group = ARM_GROUPS[arm]
    if int(change.get("group_size", -1)) != expected_group:
        raise RuntimeError("fork output group does not match arm")
    if int(change.get("previous_group_size", -1)) != 9:
        raise RuntimeError("output fork does not start from matched g9")
    if change.get("convolution_learning_rate") is not None:
        raise RuntimeError("output fork changed the selected learning rate")
    surface_map = {
        name: int(size)
        for name, size in (receipt.get("surface_group_sizes") or {}).items()
    }
    if surface_map.get(TARGET_SURFACE) != expected_group:
        raise RuntimeError("fork receipt output group metadata differs")
    non_output = {
        name: size
        for name, size in surface_map.items()
        if name != TARGET_SURFACE
    }
    if not non_output:
        raise RuntimeError("output fork does not preserve an input recipe")
    return expected_group, non_output


def main() -> None:
    args = parse_args()
    if args.steps != 24000 or args.schedule_steps != 24000:
        raise ValueError("output rescue must end all-hard at step 24000")
    arm_root = args.study_root / args.arm
    manifest_path = arm_root / "study-manifest.json"
    command_path = arm_root / "command.json"
    sealed_root = arm_root / "sealed_sources"
    if (
        manifest_path.exists()
        or command_path.exists()
        or sealed_root.exists()
    ):
        raise FileExistsError("output arm was already sealed")
    parent, receipt = verify_parent_and_fork(args)
    decision = json.loads(args.input_decision.read_text())
    output_group, non_output_map = verify_selected_parent(
        receipt,
        decision,
        arm=args.arm,
    )
    repository = Path(__file__).resolve().parents[2]
    source_files = tuple(dict.fromkeys((*SOURCE_FILES, *EXTRA_SOURCE_FILES)))
    source_hashes = {
        name: file_sha256(repository / name)
        for name in source_files
    }
    for name in source_files:
        destination = sealed_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / name, destination)
    command = training_command(args)
    manifest = {
        "format": "backdropkit-swin2sr-output-conv-rescue-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "parent_study_manifest": str(args.parent_manifest),
        "parent_study_manifest_sha256": file_sha256(args.parent_manifest),
        "input_decision": str(args.input_decision),
        "input_decision_sha256": file_sha256(args.input_decision),
        "fork_receipt": str(args.fork_receipt),
        "fork_receipt_sha256": file_sha256(args.fork_receipt),
        "fork_checkpoint": str(args.fork_checkpoint),
        "fork_checkpoint_sha256": file_sha256(args.fork_checkpoint),
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "parent_surface_step": receipt["parent_surface_step"],
        "output_group_size": output_group,
        "surface_group_sizes": receipt["surface_group_sizes"],
        "non_output_surface_group_sizes": non_output_map,
        "training_state_sha256": receipt["training_state_sha256"],
        "training_data": parent["training_data"],
        "training_data_receipts": parent["training_data_receipts"],
        "q4_reference": parent["q4_reference"],
        "q4_reference_sha256": parent["q4_reference_sha256"],
        "bundle": parent["bundle"],
        "bundle_sha256": parent["bundle_sha256"],
        "source_files": source_hashes,
        "sealed_source_root": str(sealed_root),
        "arms": {args.arm: {"training_command": command}},
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    arm_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "command_path": str(command_path),
        "arm": args.arm,
        "output_group_size": output_group,
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
