#!/usr/bin/env python3
"""Select the smallest RSTB fallback that passes both confirmation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


MIXED_ROOT = Path(__file__).parents[1] / "rstb_mixed_precision_20260731"
sys.path.insert(0, str(MIXED_ROOT))
from decide_rstb_mixed_precision import candidate_gate, verify_comparison  # noqa: E402

INT5_ROOT = Path(__file__).parents[1] / "rstb_int5_confirmation_20260731"
sys.path.insert(0, str(INT5_ROOT))
from decide_rstb_int5_confirmation import fp16_gate  # noqa: E402


FORMAT = "backdropkit-swin2sr-rstb-fallback-confirmation-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-fallback-confirmation-protocol-v1"
MATERIALIZATION_FORMAT = "backdropkit-swin2sr-rstb-fallback-confirmation-materialization-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pinned(contract: dict, label: str) -> Path:
    path = Path(contract["path"])
    if file_sha256(path) != contract["sha256"]:
        raise ValueError(f"sealed fallback source changed: {label}")
    return path


def verify_materialization(
    path: Path,
    protocol_path: Path,
    seed: int,
    arm: str,
    increment: int,
) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != MATERIALIZATION_FORMAT:
        raise ValueError("fallback materialization has the wrong format")
    if int(receipt.get("seed")) != seed or receipt.get("arm") != arm:
        raise ValueError("fallback materialization has the wrong seed or arm")
    if receipt.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("fallback materialization has the wrong protocol")
    if int(receipt["storage"]["incremental_data_bytes"]) != increment:
        raise ValueError("fallback materialization has the wrong storage")
    state = Path(receipt["dense_state"]["path"])
    if file_sha256(state) != receipt["dense_state"]["file_sha256"]:
        raise ValueError("fallback dense state changed")
    return receipt


def choose(results: dict, storage: dict) -> str:
    passing = [name for name, result in results.items() if result["passed"]]
    if not passing:
        return "fp16"
    return min(passing, key=lambda name: (storage[name], name))


def decide(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported RSTB fallback confirmation protocol")
    if not protocol.get("final_heldout_locked") or protocol.get("model_promotion_authorized"):
        raise ValueError("fallback confirmation must remain development-only")
    for name, contract in protocol["source_files"].items():
        verify_pinned(contract, name)

    retention = float(protocol["selection_rule"]["minimum_fp16_effect_retention"])
    by_arm = {}
    for arm in protocol["arms"]:
        seed_results = {}
        for seed_text, contract in protocol["seeds"].items():
            seed = int(seed_text)
            ternary_report = verify_pinned(contract["ternary_development_report"], f"{seed}-ternary-report")
            fp16_report = verify_pinned(contract["fp16_development_report"], f"{seed}-fp16-report")
            fp16_path = verify_pinned(contract["fp16_vs_ternary"], f"{seed}-fp16-comparison")
            fp16 = json.loads(fp16_path.read_text())
            verify_comparison(fp16, ternary_report, fp16_report)
            arm_contract = contract["future_arms"][arm]
            comparison_path = Path(arm_contract["comparison"])
            candidate_report = Path(arm_contract["development_report"])
            comparison = json.loads(comparison_path.read_text())
            verify_comparison(comparison, ternary_report, candidate_report)
            increment = int(protocol["storage"][arm]["incremental_data_bytes"])
            receipt = verify_materialization(
                Path(arm_contract["materialization"]),
                protocol_path,
                seed,
                arm,
                increment,
            )
            fp16_result = fp16_gate(fp16)
            candidate = candidate_gate(comparison, fp16, retention)
            seed_results[seed_text] = {
                "passed": fp16_result["passed"] and candidate["passed"],
                "fp16": fp16_result,
                "candidate": candidate,
                "comparison_sha256": file_sha256(comparison_path),
                "materialization_sha256": file_sha256(Path(arm_contract["materialization"])),
                "dense_state_sha256": receipt["dense_state"]["file_sha256"],
            }
        by_arm[arm] = {
            "passed": all(result["passed"] for result in seed_results.values()),
            "seeds": seed_results,
        }
    storage = {
        arm: int(protocol["storage"][arm]["incremental_data_bytes"])
        for arm in protocol["arms"]
    }
    storage["fp16"] = int(protocol["storage"]["fp16_incremental_data_bytes"])
    selected = choose(by_arm, storage)
    return {
        "format": FORMAT,
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "minimum_fp16_effect_retention": retention,
        "arms": by_arm,
        "selected_rstb_precision": selected,
        "selected_incremental_data_bytes": storage[selected],
        "authorized_effect": (
            "use_as_development_supported_precision_island_in_a_separately_"
            "preregistered_joint_training_study"
            if selected != "fp16" else "retain_fp16_for_further_research"
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
        raise FileExistsError(f"fallback decision already exists: {args.out}")
    result = decide(args.protocol)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_rstb_precision": result["selected_rstb_precision"],
        "selected_incremental_data_bytes": result["selected_incremental_data_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
