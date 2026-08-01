#!/usr/bin/env python3
"""Choose the smallest 5/6/7-bit RSTB arm preserving the FP16 effect."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


MIXED_ROOT = Path(__file__).parents[1] / "rstb_mixed_precision_20260731"
sys.path.insert(0, str(MIXED_ROOT))
from decide_rstb_mixed_precision import (  # noqa: E402
    candidate_gate,
    verify_comparison,
)


FORMAT = "backdropkit-swin2sr-rstb-precision-boundary-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-precision-boundary-protocol-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(results: dict, storage: dict) -> str:
    passing = [name for name, result in results.items() if result["passed"]]
    if not passing:
        return "int8_per_output"
    return min(passing, key=lambda name: (storage[name], name))


def decide(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported RSTB precision-boundary protocol")
    if not protocol.get("final_heldout_locked") or protocol.get("model_promotion_authorized"):
        raise ValueError("precision-boundary study must remain development-only")
    for name, contract in protocol["source_files"].items():
        path = Path(contract["path"])
        if file_sha256(path) != contract["sha256"]:
            raise ValueError(f"sealed precision-boundary source changed: {name}")

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
            "smallest boundary arm preserving at least 90% of every FP16 gain"
            if selected != "int8_per_output"
            else "no smaller boundary arm passed; retain the prior int8 winner"
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
        raise FileExistsError(f"precision-boundary decision already exists: {args.out}")
    result = decide(args.protocol)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_rstb_precision": result["selected_rstb_precision"],
        "projected_incremental_data_bytes": (
            result["selected_projected_incremental_data_bytes"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
