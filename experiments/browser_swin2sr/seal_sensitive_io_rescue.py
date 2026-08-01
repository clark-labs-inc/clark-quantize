#!/usr/bin/env python3
"""Seal one development-only sensitive-I/O rescue arm before training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_sensitive_curriculum import SOURCE_FILES
from training_data_contract import dataset_fingerprints


EXTRA_SOURCE_FILES = (
    "experiments/browser_swin2sr/fork_sensitive_io_checkpoint.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--fork-checkpoint", type=Path, required=True)
    parser.add_argument("--fork-receipt", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest-name", default="study-manifest.json")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=19000)
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


def training_command(args: argparse.Namespace) -> list[str]:
    script = Path(__file__).with_name("train_swin2sr_full_surface.py")
    arm_out = args.study_root / args.arm / "run"
    command = [
        sys.executable,
        str(script),
        "--base-checkpoint", str(args.fork_checkpoint),
        "--validation-data", str(args.validation_data),
        "--out", str(arm_out),
        "--steps", str(args.steps),
        "--schedule-steps", str(args.schedule_steps),
        "--batch-size", str(args.batch_size),
        "--crop-size", str(args.crop_size),
        "--learning-rate", str(args.learning_rate),
        "--loss-recipe", args.loss_recipe,
        "--validate-every", str(args.validate_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--hard-shadow-weight-start", str(args.hard_shadow_weight_start),
        "--hard-shadow-weight-end", str(args.hard_shadow_weight_end),
        "--hard-shadow-every", str(args.hard_shadow_every),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    for root in args.data:
        command.extend(("--data", str(root)))
    return command


def verify_parent_and_fork(args: argparse.Namespace) -> tuple[dict, dict]:
    parent = json.loads(args.parent_manifest.read_text())
    receipt = json.loads(args.fork_receipt.read_text())
    if parent.get("selection_source") != "development":
        raise ValueError("parent study is not development-only")
    if receipt.get("fork_checkpoint") != str(args.fork_checkpoint):
        raise ValueError("fork receipt points to a different checkpoint")
    if receipt.get("fork_checkpoint_sha256") != file_sha256(args.fork_checkpoint):
        raise ValueError("fork checkpoint hash does not match its receipt")
    if receipt.get("parent_checkpoint_sha256") is None:
        raise ValueError("fork receipt does not identify its parent")
    current_data = dataset_fingerprints(args.data)
    if current_data != parent.get("training_data"):
        raise ValueError("training datasets drifted from the parent study")
    return parent, receipt


def main() -> None:
    args = parse_args()
    parent, receipt = verify_parent_and_fork(args)
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError("manifest name must be a filename")
    manifest_path = args.study_root / args.manifest_name
    command_path = args.study_root / args.arm / "command.json"
    if manifest_path.exists() or command_path.exists():
        raise FileExistsError("rescue manifest or arm command already exists")
    repo = Path(__file__).resolve().parents[2]
    source_files = tuple(dict.fromkeys((*SOURCE_FILES, *EXTRA_SOURCE_FILES)))
    source_hashes = {
        name: file_sha256(repo / name)
        for name in source_files
    }
    sealed_root = (
        args.study_root / "sealed_sources"
        if args.manifest_name == "study-manifest.json"
        else args.study_root / f"sealed_sources-{args.arm}"
    )
    for name in source_files:
        destination = sealed_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
    command = training_command(args)
    manifest = {
        "format": "backdropkit-swin2sr-sensitive-io-rescue-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "parent_study_manifest": str(args.parent_manifest),
        "parent_study_manifest_sha256": file_sha256(args.parent_manifest),
        "fork_receipt": str(args.fork_receipt),
        "fork_receipt_sha256": file_sha256(args.fork_receipt),
        "fork_checkpoint": str(args.fork_checkpoint),
        "fork_checkpoint_sha256": file_sha256(args.fork_checkpoint),
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "surface_group_sizes": receipt["surface_group_sizes"],
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
        "final_heldout_locked": True,
    }
    args.study_root.mkdir(parents=True, exist_ok=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "command_path": str(command_path),
        "command": command,
    }, indent=2))


if __name__ == "__main__":
    main()
