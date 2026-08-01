#!/usr/bin/env python3
"""Seal the MixFrag-inspired Swin2SR scale-allocation calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from full_surface_provenance import file_sha256
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-output-fragility-study-v1"
OVERLAY_SOURCES = (
    "experiments/browser_swin2sr/calibrate_scale_fragility.py",
    "experiments/browser_swin2sr/mixfrag_scale_allocation.py",
    "experiments/browser_swin2sr/seal_output_fragility_study.py",
    "experiments/browser_swin2sr/mixfrag_scale_allocation_20260731/PROTOCOL.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-study", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"output-fragility study exists: {args.out}")
    base = json.loads(args.base_study.read_text())
    if base.get("format") != "backdropkit-swin2sr-scale-codebook-qat-study-v1":
        raise ValueError("unsupported base scale-codebook study")
    base_sources = args.base_study.parent / "sealed_sources"
    for relative, expected in base["source_sha256"].items():
        path = base_sources / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"base sealed source changed: {relative}")
    parent = Path(base["parent_checkpoint"])
    if file_sha256(parent) != base["parent_checkpoint_sha256"]:
        raise RuntimeError("output-fragility parent identity changed")
    repository = Path(__file__).resolve().parents[2]
    overlay = args.out / "sealed_overlay"
    for relative in OVERLAY_SOURCES:
        source = repository / relative
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    overlay_browser = overlay / "experiments/browser_swin2sr"
    base_browser = base_sources / "experiments/browser_swin2sr"
    pythonpath = f"{overlay_browser}:{base_browser}:{base_sources}"
    calibration_out = args.out / "calibration"
    command = [
        "/usr/bin/env",
        f"PYTHONPATH={pythonpath}",
        sys.executable,
        str(overlay / "experiments/browser_swin2sr/calibrate_scale_fragility.py"),
        "--parent", str(parent),
        "--out", str(calibration_out),
        "--samples", "64",
        "--seed", "20260731",
        "--batch-size", "8",
        "--bit-width", "1",
        "--bit-width", "2",
        "--bit-width", "4",
        "--bit-width", "8",
        "--budget-bytes", "10012",
        "--budget-bytes", "20024",
        "--budget-bytes", "75356",
        "--device", "mps",
    ]
    for data in args.data:
        command.extend(("--data", str(data)))
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_calibration_results",
        "model_promotion_authorized": False,
        "development_access_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "upstream_method": {
            "name": "MixFrag",
            "paper": "https://arxiv.org/abs/2607.28589v1",
            "submitted": "2026-07-30",
            "transfer": "isolated output fragility plus exact MCKP allocation only",
            "caution": "one-day-old v1 preprint; Swin2SR metric and backend are independent",
        },
        "base_study": str(args.base_study),
        "base_study_sha256": file_sha256(args.base_study),
        "base_sealed_sources": str(base_sources),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "training_data": dataset_fingerprints(args.data),
        "calibration_samples": 64,
        "seed": 20260731,
        "batch_size": 8,
        "bit_widths": [1, 2, 4, 8],
        "scale_storage_budgets": [10012, 20024, 75356],
        "expected_uniform_storage_bytes": {
            "q1": 5716,
            "q2": 11400,
            "q4": 25668,
            "q8": 65228,
            "fp16": 75356,
        },
        "fragility": "normalized_output_mse + 0.25 * normalized_edge_mse",
        "calibration_command": command,
        "calibration_output": str(calibration_out),
        "overlay_source_sha256": {
            relative: file_sha256(overlay / relative)
            for relative in OVERLAY_SOURCES
        },
    }
    path = args.out / "study-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "study_manifest": str(path),
        "study_manifest_sha256": file_sha256(path),
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "calibration_samples": manifest["calibration_samples"],
        "budgets": manifest["scale_storage_budgets"],
    }, indent=2))


if __name__ == "__main__":
    main()
