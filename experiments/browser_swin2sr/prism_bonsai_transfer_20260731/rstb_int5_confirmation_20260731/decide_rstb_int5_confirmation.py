#!/usr/bin/env python3
"""Require the selected RSTB int5 geometry to reproduce on both new seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


MIXED_ROOT = Path(__file__).parents[1] / "rstb_mixed_precision_20260731"
sys.path.insert(0, str(MIXED_ROOT))
from decide_rstb_mixed_precision import (  # noqa: E402
    METRICS,
    candidate_gate,
    verify_comparison,
)


FORMAT = "backdropkit-swin2sr-rstb-int5-confirmation-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-int5-confirmation-protocol-v1"
MATERIALIZATION_FORMAT = "backdropkit-swin2sr-rstb-int5-confirmation-materialization-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp16_gate(comparison: dict) -> dict:
    metrics = comparison["metrics"]
    means = {
        metric: float(metrics[metric]["candidate_minus_reference_mean"])
        for metric in METRICS
    }
    checks = {
        "ground_truth_lower_95_positive": float(
            metrics["psnr_vs_ground_truth"]["paired_bootstrap_95ci"][0]
        ) > 0.0,
        "edge_mean_nonnegative": means["edge_psnr_vs_ground_truth"] >= 0.0,
        "teacher_mean_nonnegative": means["psnr_vs_teacher"] >= 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks, "means": means}


def verify_materialization(path: Path, protocol_path: Path, seed: int, increment: int) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != MATERIALIZATION_FORMAT:
        raise ValueError("int5 confirmation materialization has the wrong format")
    if int(receipt.get("seed")) != seed:
        raise ValueError("int5 confirmation materialization has the wrong seed")
    if receipt.get("quantization") != {"bits": 5, "granularity": "per_kernel"}:
        raise ValueError("int5 confirmation materialization has the wrong geometry")
    if receipt.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("int5 confirmation materialization has the wrong protocol")
    if int(receipt["storage"]["incremental_data_bytes"]) != increment:
        raise ValueError("int5 confirmation materialization has the wrong storage")
    state = Path(receipt["dense_state"]["path"])
    if file_sha256(state) != receipt["dense_state"]["file_sha256"]:
        raise ValueError("int5 confirmation dense state changed")
    return receipt


def decide(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported RSTB int5 confirmation protocol")
    if not protocol.get("final_heldout_locked") or protocol.get("model_promotion_authorized"):
        raise ValueError("confirmation must remain development-only")
    for name, contract in protocol["source_files"].items():
        path = Path(contract["path"])
        if file_sha256(path) != contract["sha256"]:
            raise ValueError(f"sealed confirmation source changed: {name}")

    retention = float(protocol["selection_rule"]["minimum_fp16_effect_retention"])
    increment = int(protocol["storage"]["expected_incremental_data_bytes"])
    results = {}
    for seed_text, contract in protocol["seeds"].items():
        seed = int(seed_text)
        fp16_path = Path(contract["fp16_vs_ternary_comparison"])
        int5_path = Path(contract["int5_vs_ternary_comparison"])
        fp16 = json.loads(fp16_path.read_text())
        int5 = json.loads(int5_path.read_text())
        verify_comparison(
            fp16,
            Path(contract["ternary_development_report"]),
            Path(contract["fp16_development_report"]),
        )
        verify_comparison(
            int5,
            Path(contract["ternary_development_report"]),
            Path(contract["int5_development_report"]),
        )
        receipt = verify_materialization(
            Path(contract["int5_materialization"]), protocol_path, seed, increment
        )
        fp16_result = fp16_gate(fp16)
        int5_result = candidate_gate(int5, fp16, retention)
        results[seed_text] = {
            "passed": fp16_result["passed"] and int5_result["passed"],
            "fp16": fp16_result,
            "int5": int5_result,
            "fp16_comparison_sha256": file_sha256(fp16_path),
            "int5_comparison_sha256": file_sha256(int5_path),
            "int5_materialization_sha256": file_sha256(
                Path(contract["int5_materialization"])
            ),
            "int5_dense_state_sha256": receipt["dense_state"]["file_sha256"],
        }
    confirmed = all(result["passed"] for result in results.values())
    return {
        "format": FORMAT,
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "required_confirmation_seeds": [int(seed) for seed in protocol["seeds"]],
        "minimum_fp16_effect_retention": retention,
        "seeds": results,
        "int5_per_kernel_confirmed": confirmed,
        "authorized_effect": (
            "use_as_development_supported_precision_island_in_a_separately_"
            "preregistered_joint_training_study"
            if confirmed else "none"
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
        raise FileExistsError(f"confirmation decision already exists: {args.out}")
    result = decide(args.protocol)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "int5_per_kernel_confirmed": result["int5_per_kernel_confirmed"],
    }, indent=2))


if __name__ == "__main__":
    main()
