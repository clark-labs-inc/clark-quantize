#!/usr/bin/env python3
"""Seal and verify the corpus-aware source boundary before recovery stage two."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FILES = (
    "launch_recovery_full_surface_after_stage1.sh",
    "recovery_corpus_identity.py",
    "recovery_data.py",
    "recovery_resume_policy.py",
    "run_recovery_factorial.py",
    "seal_stage2_source.py",
    "train_swin2sr_recovery.py",
    "training_data_contract.py",
    "validate_training_corpus.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gate(
    stage1_manifest_path: Path,
    stage1_summary_path: Path,
    training_receipt_path: Path,
) -> dict:
    stage1_manifest = json.loads(stage1_manifest_path.read_text())
    stage1_summary = json.loads(stage1_summary_path.read_text())
    training_receipt = json.loads(training_receipt_path.read_text())
    if (
        stage1_manifest.get("format")
        != "backdropkit-swin2sr-recovery-factorial-v1"
        or stage1_manifest.get("stage") != "stage1"
    ):
        raise ValueError("invalid stage-one study manifest")
    if (
        stage1_summary.get("completed") != 24
        or stage1_summary.get("failed") != 0
        or len(stage1_summary.get("ranking", [])) != 24
    ):
        raise ValueError("stage one is not complete and failure-free")
    if (
        training_receipt.get("format")
        != "backdropkit-swin2sr-training-corpus-receipt-v1"
        or training_receipt.get("images") != 102000
        or training_receipt.get("languages") != 204
        or training_receipt.get("heldout_hash_overlap") != 0
    ):
        raise ValueError("glyph-safe training receipt does not satisfy the gate")
    source_root = Path(__file__).resolve().parent
    return {
        "format": "backdropkit-swin2sr-stage2-source-gate-v1",
        "stage1_manifest": str(stage1_manifest_path),
        "stage1_manifest_sha256": file_sha256(stage1_manifest_path),
        "stage1_summary": str(stage1_summary_path),
        "stage1_summary_sha256": file_sha256(stage1_summary_path),
        "stage1_training_roots": stage1_manifest["config"]["data"],
        "stage2_training_receipt": str(training_receipt_path),
        "stage2_training_receipt_sha256": file_sha256(training_receipt_path),
        "stage2_training_root": training_receipt["root"],
        "stage2_training_manifest_sha256": training_receipt["manifest_sha256"],
        "reset_data_stream_required": True,
        "source_files": {
            name: file_sha256(source_root / name)
            for name in SOURCE_FILES
        },
    }


def verify_gate(path: Path) -> dict:
    gate = json.loads(path.read_text())
    if gate.get("format") != "backdropkit-swin2sr-stage2-source-gate-v1":
        raise ValueError("invalid stage-two source gate")
    if gate.get("reset_data_stream_required") is not True:
        raise ValueError("stage-two source gate does not require a data reset")
    for path_key, hash_key in (
        ("stage1_manifest", "stage1_manifest_sha256"),
        ("stage1_summary", "stage1_summary_sha256"),
        ("stage2_training_receipt", "stage2_training_receipt_sha256"),
    ):
        source = Path(gate[path_key])
        if not source.is_file() or file_sha256(source) != gate[hash_key]:
            raise ValueError(f"stage-two gate input changed: {path_key}")
    source_root = Path(__file__).resolve().parent
    expected_sources = {
        name: file_sha256(source_root / name)
        for name in SOURCE_FILES
    }
    if gate.get("source_files") != expected_sources:
        raise ValueError("stage-two source files changed after the gate was sealed")
    rebuilt = build_gate(
        Path(gate["stage1_manifest"]),
        Path(gate["stage1_summary"]),
        Path(gate["stage2_training_receipt"]),
    )
    if rebuilt != gate:
        raise ValueError("stage-two source gate content is not reproducible")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-manifest", type=Path)
    parser.add_argument("--stage1-summary", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        gate = verify_gate(args.verify)
    else:
        required = (
            args.stage1_manifest,
            args.stage1_summary,
            args.training_receipt,
            args.out,
        )
        if any(value is None for value in required):
            raise ValueError("building a gate requires all input and output paths")
        gate = build_gate(
            args.stage1_manifest,
            args.stage1_summary,
            args.training_receipt,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
