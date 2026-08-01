#!/usr/bin/env python3
"""Capture immutable inputs and environment metadata for a QAT run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


SOURCE_FILES = (
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/artifact_ledger.py",
    "experiments/browser_swin2sr/verify_browser_artifact.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_air/quant/packing.py",
    "clark_quantize/bonsai_qat8.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--launch-command", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def data_record(root: Path) -> dict:
    manifest = root / "manifest.jsonl"
    summary = root / "summary.json"
    if not manifest.exists():
        raise ValueError(f"data root lacks manifest: {root}")
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "summary": (
            json.loads(summary.read_text())
            if summary.exists()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    source_hashes = {
        name: sha256(repo / name)
        for name in SOURCE_FILES
    }
    package_names = ("torch", "transformers", "numpy", "Pillow")
    manifest = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "launch_command": args.launch_command,
        "resume": (
            {
                "path": str(args.resume),
                "sha256": sha256(args.resume),
            }
            if args.resume
            else None
        ),
        "training_data": [data_record(root) for root in args.data],
        "source": {
            "git_commit": git(repo, "rev-parse", "HEAD"),
            "git_status": git(repo, "status", "--short"),
            "file_sha256": source_hashes,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_mps_available": torch.backends.mps.is_available(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in package_names
            },
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "run-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": str(destination),
        "resume_sha256": (
            manifest["resume"]["sha256"]
            if manifest["resume"]
            else None
        ),
        "data_manifest_sha256": [
            item["manifest_sha256"]
            for item in manifest["training_data"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
