#!/usr/bin/env python3
"""Apply the preregistered three-seed retained-precision confirmation rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decide_retained_precision_discovery import (
    FORMAT as DISCOVERY_FORMAT,
    ISLANDS,
    STUDY_FORMAT,
    command_output,
    island_gate,
    verify_comparison,
)


FORMAT = "backdropkit-swin2sr-retained-precision-confirmation-decision-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_means(seed_results: dict[int, dict]) -> dict:
    names = ("ground_truth_mean", "edge_mean", "teacher_mean")
    return {
        name: sum(float(result[name]) for result in seed_results.values())
        / len(seed_results)
        for name in names
    }


def final_gate(seed_results: dict[int, dict]) -> dict:
    checks = {
        "all_seed_ground_truth_lower_95_positive": all(
            result["checks"]["ground_truth_lower_95_positive"]
            for result in seed_results.values()
        ),
        "all_seed_edge_means_nonnegative": all(
            result["checks"]["edge_mean_nonnegative"]
            for result in seed_results.values()
        ),
        "all_seed_teacher_means_nonnegative": all(
            result["checks"]["teacher_mean_nonnegative"]
            for result in seed_results.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "three_seed_means": metric_means(seed_results),
    }


def decide(study_path: Path, discovery_path: Path) -> dict:
    study = json.loads(study_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    if study.get("format") != STUDY_FORMAT:
        raise ValueError("unsupported retained-precision study format")
    if discovery.get("format") != DISCOVERY_FORMAT:
        raise ValueError("unsupported discovery decision format")
    if discovery.get("study_manifest_sha256") != file_sha256(study_path):
        raise ValueError("discovery decision belongs to another study")
    if study.get("model_promotion_authorized") or not study.get(
        "final_heldout_locked"
    ):
        raise ValueError("retained-precision study violates promotion isolation")

    discovery_seed = int(study["design"]["discovery_seed"])
    confirmation_seeds = [
        int(seed) for seed in study["design"]["confirmation_seeds"]
    ]
    all_seeds = [discovery_seed, *confirmation_seeds]
    authorized = set(discovery["authorized_confirmation_arms"])
    results = {}
    supported = []
    for island in ISLANDS:
        discovery_passed = bool(discovery["islands"][island]["passed"])
        seed_results = {}
        if discovery_passed:
            expected = {
                f"{island}__{precision}__seed{seed}"
                for seed in confirmation_seeds
                for precision in ("ternary", "fp16")
            }
            if not expected.issubset(authorized):
                raise ValueError("discovery decision omitted a required confirmation arm")
            for seed in all_seeds:
                ternary_name = f"{island}__ternary__seed{seed}"
                fp16_name = f"{island}__fp16__seed{seed}"
                ternary_arm = study["arms"][ternary_name]
                fp16_arm = study["arms"][fp16_name]
                comparison_path = command_output(
                    fp16_arm["control_comparison_command"]
                )
                comparison = json.loads(comparison_path.read_text())
                ternary_report = Path(ternary_arm["development_report"])
                fp16_report = Path(fp16_arm["development_report"])
                verify_comparison(comparison, ternary_report, fp16_report)
                seed_results[seed] = {
                    **island_gate(comparison),
                    "comparison": str(comparison_path),
                    "comparison_sha256": file_sha256(comparison_path),
                }
            gate = final_gate(seed_results)
            if gate["passed"]:
                supported.append(island)
        else:
            gate = {
                "passed": False,
                "checks": {"discovery_gate_passed": False},
                "three_seed_means": None,
            }
        results[island] = {
            "discovery_passed": discovery_passed,
            "seed_results": {str(seed): value for seed, value in seed_results.items()},
            **gate,
        }

    supported.sort(
        key=lambda island: (
            results[island]["three_seed_means"]["ground_truth_mean"],
            results[island]["three_seed_means"]["teacher_mean"],
            results[island]["three_seed_means"]["edge_mean"],
            island,
        ),
        reverse=True,
    )
    selected = supported[0] if supported else None
    return {
        "format": FORMAT,
        "selection_source": "three_seed_development_confirmation",
        "study_manifest": str(study_path),
        "study_manifest_sha256": file_sha256(study_path),
        "discovery_decision": str(discovery_path),
        "discovery_decision_sha256": file_sha256(discovery_path),
        "islands": results,
        "supported_islands": supported,
        "selected_precision_location": selected,
        "precision_location_supported": selected is not None,
        "required_next_gate": (
            "preregister_compression_and_runtime_study_for_selected_location"
            if selected
            else "revise_precision_or_training_hypothesis_without_final_heldout"
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--discovery-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"confirmation decision already exists: {args.out}")
    decision = decide(args.study_manifest, args.discovery_decision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "supported_islands": decision["supported_islands"],
        "selected_precision_location": decision["selected_precision_location"],
    }, indent=2))


if __name__ == "__main__":
    main()
