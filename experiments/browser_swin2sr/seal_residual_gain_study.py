#!/usr/bin/env python3
"""Preregister a zero-parameter residual-gain development probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_surface_provenance import file_sha256


SOURCE_FILES = (
    "experiments/browser_swin2sr/residual_gain.py",
    "experiments/browser_swin2sr/evaluate_residual_gain.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/decide_pixel_objective.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"residual-gain study exists: {args.out}")
    bundle = json.loads(args.bundle.read_text())
    q4 = json.loads(args.q4_reference.read_text())
    if bundle.get("manifest_sha256") != q4.get("bundle_manifest_sha256"):
        raise RuntimeError("q4 and residual-gain bundle identities differ")
    protocol = {
        "format": "backdropkit-swin2sr-residual-gain-protocol-v1",
        "status": "preregistered_before_residual_gain_results",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "state": str(args.state),
        "state_sha256": file_sha256(args.state),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "crop_size": 128,
        "torch_threads": 2,
        "fold_seed": 20260731,
        "screen_per_source": 64,
        "confirmation_per_source": 128,
        "gains": [0.9, 0.95, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15],
        "decision_rule": {
            "bootstrap_seed": 20260731,
            "bootstrap_samples": 5000,
            "ground_truth_lower_95_min_db": 0.0,
            "edge_mean_min_db": -0.05,
            "teacher_mean_min_db": -0.1,
            "every_source_ground_truth_mean_min_db": -0.1,
        },
        "passing_result": "selected gain advances to full development only",
        "failing_result": "retain unmodified ternary output",
        "final_heldout_access_authorized": False,
        "source_sha256": {
            name: file_sha256(Path(name)) for name in SOURCE_FILES
        },
    }
    args.out.mkdir(parents=True)
    protocol_path = args.out / "protocol.json"
    result_path = args.out / "result.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    command = [
        "python3",
        "experiments/browser_swin2sr/evaluate_residual_gain.py",
        "--protocol", str(protocol_path),
        "--out", str(result_path),
    ]
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps({
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "command": command,
    }, indent=2))


if __name__ == "__main__":
    main()
