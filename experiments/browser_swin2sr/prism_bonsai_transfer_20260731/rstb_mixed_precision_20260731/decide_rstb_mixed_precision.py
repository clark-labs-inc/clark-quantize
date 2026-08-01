#!/usr/bin/env python3
"""Choose the smallest low-bit RSTB format that preserves the FP16 gain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "backdropkit-swin2sr-rstb-mixed-precision-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-mixed-precision-protocol-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_comparison(
    comparison: dict,
    reference_report_path: Path,
    candidate_report_path: Path,
) -> None:
    reference = json.loads(reference_report_path.read_text())
    candidate = json.loads(candidate_report_path.read_text())
    if Path(comparison["reference"]["path"]).resolve() != reference_report_path.resolve():
        raise ValueError("comparison reference path differs from protocol")
    if Path(comparison["candidate"]["path"]).resolve() != candidate_report_path.resolve():
        raise ValueError("comparison candidate path differs from protocol")
    if comparison["reference"]["state_sha256"] != reference["state_sha256"]:
        raise ValueError("comparison reference state hash differs from report")
    if comparison["candidate"]["state_sha256"] != candidate["state_sha256"]:
        raise ValueError("comparison candidate state hash differs from report")
    if {
        int(comparison["metrics"][metric]["samples"])
        for metric in METRICS
    } != {3040}:
        raise ValueError("mixed-precision comparison must cover all 3,040 samples")


def candidate_gate(comparison: dict, fp16: dict, retention: float) -> dict:
    metrics = comparison["metrics"]
    fp16_metrics = fp16["metrics"]
    fp16_means = {
        metric: float(fp16_metrics[metric]["candidate_minus_reference_mean"])
        for metric in METRICS
    }
    if any(value <= 0.0 for value in fp16_means.values()):
        raise ValueError("FP16 RSTB effect must be positive on every metric")
    means = {
        metric: float(metrics[metric]["candidate_minus_reference_mean"])
        for metric in METRICS
    }
    ratios = {
        metric: means[metric] / fp16_means[metric]
        for metric in METRICS
    }
    checks = {
        "ground_truth_lower_95_positive": float(
            metrics["psnr_vs_ground_truth"]["paired_bootstrap_95ci"][0]
        ) > 0.0,
        "edge_mean_nonnegative": means["edge_psnr_vs_ground_truth"] >= 0.0,
        "teacher_mean_nonnegative": means["psnr_vs_teacher"] >= 0.0,
        **{
            f"{metric}_retains_at_least_{int(retention * 100)}pct": ratio >= retention
            for metric, ratio in ratios.items()
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "means": means,
        "fp16_retention_ratios": ratios,
    }


def choose(results: dict, storage: dict) -> str:
    passing = [name for name, result in results.items() if result["passed"]]
    if not passing:
        return "fp16"
    return min(passing, key=lambda name: (storage[name], name))


def decide(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported RSTB mixed-precision protocol")
    if not protocol.get("final_heldout_locked") or protocol.get("model_promotion_authorized"):
        raise ValueError("mixed-precision study must remain development-only")
    for name, contract in protocol["source_files"].items():
        path = Path(contract["path"])
        if file_sha256(path) != contract["sha256"]:
            raise ValueError(f"sealed mixed-precision source changed: {name}")

    fp16_contract = protocol["comparisons"]["fp16_vs_ternary"]
    fp16_path = Path(fp16_contract["output"])
    fp16 = json.loads(fp16_path.read_text())
    verify_comparison(
        fp16,
        Path(fp16_contract["reference_report"]),
        Path(fp16_contract["candidate_report"]),
    )
    retention = float(protocol["selection_rule"]["minimum_fp16_effect_retention"])
    results = {}
    for name in protocol["arms"]:
        contract = protocol["comparisons"][f"{name}_vs_ternary"]
        path = Path(contract["output"])
        comparison = json.loads(path.read_text())
        verify_comparison(
            comparison,
            Path(contract["reference_report"]),
            Path(contract["candidate_report"]),
        )
        results[name] = {
            **candidate_gate(comparison, fp16, retention),
            "comparison": str(path),
            "comparison_sha256": file_sha256(path),
        }
    storage = protocol["projected_incremental_data_bytes"]
    selected = choose(results, storage)
    return {
        "format": FORMAT,
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "fp16_comparison_sha256": file_sha256(fp16_path),
        "minimum_fp16_effect_retention": retention,
        "candidates": results,
        "selected_rstb_precision": selected,
        "selected_projected_incremental_data_bytes": storage[selected],
        "interpretation": (
            "smallest low-bit RSTB representation preserving the preregistered "
            "fraction of every FP16 quality gain"
            if selected != "fp16"
            else "no low-bit RSTB arm preserved the preregistered FP16 gains"
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"mixed-precision decision already exists: {args.out}")
    decision = decide(args.protocol)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_rstb_precision": decision["selected_rstb_precision"],
        "projected_incremental_data_bytes": (
            decision["selected_projected_incremental_data_bytes"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
