#!/usr/bin/env python3
"""Apply the sealed retained-precision discovery gate.

The discovery comparison is FP16 candidate minus the same-location ternary
control.  This script only authorizes preregistered confirmation arms; it can
never authorize final-heldout access or model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "backdropkit-swin2sr-retained-precision-discovery-decision-v1"
STUDY_FORMAT = "backdropkit-swin2sr-retained-precision-study-v2"
ISLANDS = ("image_io_fp16", "rstb_fusion_fp16")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(command: list[str]) -> Path:
    try:
        return Path(command[command.index("--out") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("comparison command has no --out path") from error


def island_gate(comparison: dict) -> dict:
    metrics = comparison["metrics"]
    ground_truth = metrics["psnr_vs_ground_truth"]
    edge = metrics["edge_psnr_vs_ground_truth"]
    teacher = metrics["psnr_vs_teacher"]
    lower = float(ground_truth["paired_bootstrap_95ci"][0])
    edge_mean = float(edge["candidate_minus_reference_mean"])
    teacher_mean = float(teacher["candidate_minus_reference_mean"])
    checks = {
        "ground_truth_lower_95_positive": lower > 0.0,
        "edge_mean_nonnegative": edge_mean >= 0.0,
        "teacher_mean_nonnegative": teacher_mean >= 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ground_truth_lower_95": lower,
        "ground_truth_mean": float(
            ground_truth["candidate_minus_reference_mean"]
        ),
        "edge_mean": edge_mean,
        "teacher_mean": teacher_mean,
    }


def verify_comparison(
    comparison: dict,
    ternary_report: Path,
    fp16_report: Path,
) -> None:
    reference = comparison["reference"]
    candidate = comparison["candidate"]
    if Path(reference["path"]).resolve() != ternary_report.resolve():
        raise ValueError("comparison reference is not the paired ternary report")
    if Path(candidate["path"]).resolve() != fp16_report.resolve():
        raise ValueError("comparison candidate is not the paired FP16 report")
    ternary = json.loads(ternary_report.read_text())
    fp16 = json.loads(fp16_report.read_text())
    if reference["state_sha256"] != ternary["state_sha256"]:
        raise ValueError("comparison reference state hash differs from report")
    if candidate["state_sha256"] != fp16["state_sha256"]:
        raise ValueError("comparison candidate state hash differs from report")
    sample_counts = {
        int(metric["samples"])
        for metric in comparison["metrics"].values()
    }
    if sample_counts != {3040}:
        raise ValueError("discovery comparison must cover all 3,040 samples")


def decide(study_path: Path) -> dict:
    study = json.loads(study_path.read_text())
    if study.get("format") != STUDY_FORMAT:
        raise ValueError("unsupported retained-precision study format")
    if study.get("model_promotion_authorized"):
        raise ValueError("discovery study cannot authorize model promotion")
    if not study.get("final_heldout_locked"):
        raise ValueError("final heldout must remain locked")

    discovery_seed = int(study["design"]["discovery_seed"])
    confirmation_seeds = [
        int(seed) for seed in study["design"]["confirmation_seeds"]
    ]
    results = {}
    authorized = []
    for island in ISLANDS:
        ternary_name = f"{island}__ternary__seed{discovery_seed}"
        fp16_name = f"{island}__fp16__seed{discovery_seed}"
        ternary_arm = study["arms"][ternary_name]
        fp16_arm = study["arms"][fp16_name]
        comparison_path = command_output(fp16_arm["control_comparison_command"])
        comparison = json.loads(comparison_path.read_text())
        ternary_report = Path(ternary_arm["development_report"])
        fp16_report = Path(fp16_arm["development_report"])
        verify_comparison(comparison, ternary_report, fp16_report)
        gate = island_gate(comparison)
        results[island] = {
            **gate,
            "comparison": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "ternary_report_sha256": file_sha256(ternary_report),
            "fp16_report_sha256": file_sha256(fp16_report),
        }
        if gate["passed"]:
            for seed in confirmation_seeds:
                authorized.extend(
                    [
                        f"{island}__ternary__seed{seed}",
                        f"{island}__fp16__seed{seed}",
                    ]
                )

    if any(name not in study["arms"] for name in authorized):
        raise ValueError("decision produced an arm absent from the sealed study")
    return {
        "format": FORMAT,
        "selection_source": "development_discovery_seed",
        "study_manifest": str(study_path),
        "study_manifest_sha256": file_sha256(study_path),
        "discovery_seed": discovery_seed,
        "confirmation_seeds": confirmation_seeds,
        "islands": results,
        "authorized_confirmation_arms": authorized,
        "confirmation_required": bool(authorized),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"discovery decision already exists: {args.out}")
    decision = decide(args.study_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "authorized_confirmation_arms": decision[
            "authorized_confirmation_arms"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
