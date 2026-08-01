#!/usr/bin/env python3
"""Verify a completed sealed-source sensitive-I/O checkpoint continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from full_surface_provenance import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def option_value(arguments: list[str], flag: str) -> str:
    indices = [
        index for index, value in enumerate(arguments)
        if value == flag
    ]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise RuntimeError(f"recorded command does not contain one {flag}")
    return arguments[indices[0] + 1]


def latest_resume_start(
    log_path: Path,
    *,
    surface_step: int,
) -> dict:
    starts = []
    with log_path.open() as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("event") == "start"
                and int(record.get("start_surface_step", -1)) == surface_step
            ):
                starts.append(record)
    if len(starts) != 1:
        raise RuntimeError("training log has no unique resume start event")
    return starts[0]


def verify_resume(receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "backdropkit-swin2sr-sealed-resume-v1":
        raise RuntimeError("unsupported sealed resume receipt")
    if receipt.get("status") != "complete":
        raise RuntimeError("sealed resume is not complete")
    if receipt.get("model_promotion_authorized"):
        raise RuntimeError("sealed resume improperly authorizes promotion")
    if not receipt.get("final_heldout_locked"):
        raise RuntimeError("sealed resume does not lock final heldout")
    for identity in receipt["local_import_closure"].values():
        path = Path(identity["path"])
        if file_sha256(path) != identity["sha256"]:
            raise RuntimeError(f"resume source changed: {path}")
    runner = Path(receipt["runner"])
    if file_sha256(runner) != receipt["runner_sha256"]:
        raise RuntimeError("sealed resume runner changed")
    checkpoint = Path(receipt["target_checkpoint"])
    if file_sha256(checkpoint) != receipt["target_checkpoint_sha256"]:
        raise RuntimeError("target checkpoint hash changed")
    payload = torch.load(checkpoint, map_location="cpu")
    if int(payload["surface_step"]) != int(receipt["target_step"]):
        raise RuntimeError("target checkpoint is at a different step")
    policy = receipt["lineage_policy"]
    training = payload["surface_training"]
    if (
        training.get("resume_checkpoint")
        != policy["evaluator_resume_checkpoint"]
    ):
        raise RuntimeError("target checkpoint lost direct fork identity")
    chain = training.get("resume_chain") or []
    expected_chain = [{
        "checkpoint": policy["resume_chain_checkpoint"],
        "checkpoint_sha256": policy["resume_chain_checkpoint_sha256"],
    }]
    if chain != expected_chain:
        raise RuntimeError("target checkpoint has a different resume chain")
    command = receipt["command"]
    run_out = Path(option_value(command, "--out"))
    start = latest_resume_start(
        run_out / "train.jsonl",
        surface_step=int(receipt["surface_step"]),
    )
    if not start.get("exact_data_resume"):
        raise RuntimeError("training log does not prove exact data resume")
    if (
        Path(start["resume_checkpoint"]).resolve()
        != Path(receipt["checkpoint"]).resolve()
    ):
        raise RuntimeError("training log resumed another checkpoint")
    checkpoint_receipt = json.loads(
        Path(receipt["checkpoint_receipt"]).read_text()
    )
    matching = [
        record
        for record in checkpoint_receipt["checkpoints"]
        if int(record["surface_step"]) == int(receipt["surface_step"])
    ]
    if len(matching) != 1:
        raise RuntimeError("source checkpoint receipt is ambiguous")
    expected_baseline = matching[0]["blended"]
    for metric in ("psnr_vs_ground_truth", "psnr_vs_teacher"):
        if abs(float(start["baseline"][metric]) - float(
            expected_baseline[metric]
        )) > 1e-9:
            raise RuntimeError("resume baseline differs from checkpoint receipt")
    return {
        "format": "backdropkit-swin2sr-sealed-resume-verification-v1",
        "selection_source": "development",
        "resume_receipt": str(receipt_path),
        "resume_receipt_sha256": file_sha256(receipt_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": receipt["target_checkpoint_sha256"],
        "surface_step": int(receipt["target_step"]),
        "source_checkpoint": receipt["checkpoint"],
        "source_checkpoint_sha256": receipt["checkpoint_sha256"],
        "exact_data_resume": True,
        "baseline_reproduced": True,
        "resume_chain_verified": True,
        "local_imports_verified": len(receipt["local_import_closure"]),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"resume verification already exists: {args.out}")
    verification = verify_resume(args.resume_receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
