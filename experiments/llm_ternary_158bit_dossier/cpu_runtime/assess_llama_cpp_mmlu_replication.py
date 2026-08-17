#!/usr/bin/env python3
"""Compare exact BF16, direct-PTQ, and QAT MMLU receipts fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    claimed = report.get("report_content_sha256")
    body = {
        key: value for key, value in report.items() if key != "report_content_sha256"
    }
    if not claimed or canonical_digest(body) != claimed:
        raise ValueError(f"report content hash mismatch: {path}")
    if (
        report.get("schema") != "llama_cpp_mmlu_exact_choice_v1"
        or report.get("status") != "completed"
        or report.get("valid") is not True
        or report.get("summary", {}).get("n") != 64
        or len(report.get("records", [])) != 64
    ):
        raise ValueError(f"report is incomplete or invalid: {path}")
    return report


def scientific_contract(report: dict[str, Any]) -> dict[str, Any]:
    excluded = {"model", "model_label", "model_sha256", "server"}
    return {
        key: value
        for key, value in report["contract"].items()
        if key not in excluded
    }


def paired_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["sample_id"]: row for row in left["records"]}
    right_rows = {row["sample_id"]: row for row in right["records"]}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("paired MMLU sample IDs differ")
    keys = sorted(left_rows)
    for key in keys:
        if (
            left_rows[key]["gold"] != right_rows[key]["gold"]
            or left_rows[key]["row_sha256"] != right_rows[key]["row_sha256"]
            or left_rows[key]["dataset_index"] != right_rows[key]["dataset_index"]
        ):
            raise ValueError(f"paired frozen row differs: {key}")
    return {
        "n": len(keys),
        "left_correct": sum(bool(left_rows[key]["correct"]) for key in keys),
        "right_correct": sum(bool(right_rows[key]["correct"]) for key in keys),
        "left_only_correct": sum(
            bool(left_rows[key]["correct"]) and not bool(right_rows[key]["correct"])
            for key in keys
        ),
        "right_only_correct": sum(
            bool(right_rows[key]["correct"]) and not bool(left_rows[key]["correct"])
            for key in keys
        ),
        "prediction_equal": sum(
            left_rows[key]["prediction"] == right_rows[key]["prediction"]
            for key in keys
        ),
        "correctness_equal": sum(
            bool(left_rows[key]["correct"]) == bool(right_rows[key]["correct"])
            for key in keys
        ),
        "differences": [
            {
                "sample_id": key,
                "subject": left_rows[key]["subject"],
                "gold": left_rows[key]["gold"],
                "left_prediction": left_rows[key]["prediction"],
                "left_correct": bool(left_rows[key]["correct"]),
                "right_prediction": right_rows[key]["prediction"],
                "right_correct": bool(right_rows[key]["correct"]),
            }
            for key in keys
            if left_rows[key]["prediction"] != right_rows[key]["prediction"]
        ],
    }


def assess(
    bf16_path: Path,
    ptq_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    bf16 = load_report(bf16_path)
    ptq = load_report(ptq_path)
    candidate = load_report(candidate_path)
    contracts = [scientific_contract(report) for report in (bf16, ptq, candidate)]
    if contracts[1:] != contracts[:-1]:
        raise ValueError("MMLU scientific contracts differ across arms")

    candidate_vs_bf16 = paired_summary(candidate, bf16)
    candidate_vs_ptq = paired_summary(candidate, ptq)
    bf16_correct = int(bf16["summary"]["correct"])
    ptq_correct = int(ptq["summary"]["correct"])
    candidate_correct = int(candidate["summary"]["correct"])
    gates = {
        "candidate_within_three_items_of_bf16": candidate_correct
        >= bf16_correct - 3,
        "candidate_at_least_direct_ptq": candidate_correct >= ptq_correct,
    }
    passed = all(gates.values())
    result = {
        "schema": "llama_cpp_mmlu_replication_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "pass": passed,
        "decision": (
            "authorize_preregistered_two_layer_experiment"
            if passed
            else "keep_depth_frozen"
        ),
        "broad_depth_expansion_authorized": False,
        "maximum_next_scope_if_passed": "one additional preregistered MLP layer on Qwen3.5-4B",
        "claim_boundary": "paired 64-row one-layer replication; not leaderboard-comparable, whole-model, 27B, or a breakthrough",
        "thresholds": {
            "candidate_maximum_loss_vs_bf16_items": 3,
            "candidate_minimum_correct_vs_direct_ptq": "greater_than_or_equal",
        },
        "observed": {
            "bf16_correct": [bf16_correct, 64],
            "direct_ptq_correct": [ptq_correct, 64],
            "candidate_correct": [candidate_correct, 64],
            "candidate_delta_vs_bf16_items": candidate_correct - bf16_correct,
            "candidate_delta_vs_direct_ptq_items": candidate_correct - ptq_correct,
        },
        "gates": gates,
        "candidate_vs_bf16": candidate_vs_bf16,
        "candidate_vs_direct_ptq": candidate_vs_ptq,
        "models": {
            "bf16": bf16["contract"]["model_sha256"],
            "direct_ptq": ptq["contract"]["model_sha256"],
            "candidate": candidate["contract"]["model_sha256"],
        },
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in (
                ("bf16", bf16_path),
                ("direct_ptq", ptq_path),
                ("candidate", candidate_path),
            )
        },
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    result["report_content_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--direct-ptq", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = assess(args.bf16, args.direct_ptq, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
