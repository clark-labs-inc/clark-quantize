#!/usr/bin/env python3
"""Apply the preregistered affine-ternary output-screen gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORMAT = "backdropkit-swin2sr-affine-ternary-screen-decision-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def decide(study: dict, comparison: dict) -> dict:
    gate = study["pass_gate"]
    metrics = comparison["metrics"]
    ground_truth = metrics["psnr_vs_ground_truth"]
    edge = metrics["edge_psnr_vs_ground_truth"]
    teacher = metrics["psnr_vs_teacher"]
    checks = {
        "ground_truth_lower95_positive": (
            ground_truth["paired_bootstrap_95ci"][0]
            > gate["ground_truth_delta_lower95_minimum"]
        ),
        "edge_mean_nonnegative": (
            edge["candidate_minus_reference_mean"]
            >= gate["edge_delta_mean_minimum"]
        ),
        "teacher_mean_nonnegative": (
            teacher["candidate_minus_reference_mean"]
            >= gate["teacher_delta_mean_minimum"]
        ),
    }
    samples = {metrics[name]["samples"] for name in (
        "psnr_vs_ground_truth",
        "edge_psnr_vs_ground_truth",
        "psnr_vs_teacher",
    )}
    if samples != {study["development_samples"]}:
        raise ValueError(f"paired sample count mismatch: {samples}")
    passed = all(checks.values())
    return {
        "format": FORMAT,
        "passed": passed,
        "checks": checks,
        "metrics": {
            "ground_truth": ground_truth,
            "edge": edge,
            "teacher": teacher,
        },
        "outcome": (
            "advance_frozen_state_to_separately_sealed_full_development"
            if passed
            else "reject_static_affine_restoration"
        ),
        "training_authorized": False,
        "final_heldout_access_authorized": False,
        "runtime_implementation_authorized": False,
        "model_promotion_authorized": False,
    }


def main() -> None:
    args = parse_args()
    study = json.loads(args.study_manifest.read_text())
    comparison_path = Path(study["comparison"])
    comparison = json.loads(comparison_path.read_text())
    result = decide(study, comparison)
    result["study_manifest"] = str(args.study_manifest)
    result["comparison"] = str(comparison_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
