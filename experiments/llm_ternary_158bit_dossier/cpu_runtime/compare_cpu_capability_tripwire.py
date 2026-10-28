#!/usr/bin/env python3
"""Compare frozen quick capability reports and emit an atomic gate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    claimed = report.get("report_content_sha256")
    body = {key: value for key, value in report.items() if key != "report_content_sha256"}
    if not claimed or canonical_digest(body) != claimed:
        raise RuntimeError(f"report content hash mismatch: {path}")
    return report


def summary(report: dict[str, Any], task: str) -> dict[str, Any]:
    matches = [row for row in report["summaries"] if row["task"] == task]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {task!r} summary, got {len(matches)}")
    return matches[0]


def evaluate_pair(
    task: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    minimum_retention: float,
) -> dict[str, Any]:
    if baseline["manifest_content_sha256"] != candidate["manifest_content_sha256"]:
        raise RuntimeError(f"{task}: baseline/candidate manifests differ")
    base = summary(baseline, task)
    cand = summary(candidate, task)
    if base["n"] != cand["n"] or base["n"] <= 0:
        raise RuntimeError(f"{task}: denominator mismatch or empty task")
    if base["correct"] <= 0:
        return {
            "task": task,
            "status": "invalid_control",
            "pass": False,
            "reason": "BF16 control has no correct items, so the slice cannot measure retention",
            "baseline": base,
            "candidate": cand,
        }
    required_correct = math.ceil(base["correct"] * minimum_retention)
    retained = cand["correct"] / base["correct"]
    passed = cand["correct"] >= required_correct
    return {
        "task": task,
        "status": "pass" if passed else "catastrophic_forgetting",
        "pass": passed,
        "minimum_retention": minimum_retention,
        "required_candidate_correct": required_correct,
        "retained_control_correct_fraction": retained,
        "absolute_accuracy_delta": cand["accuracy"] - base["accuracy"],
        "baseline": base,
        "candidate": cand,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-math", required=True, type=Path)
    parser.add_argument("--candidate-math", required=True, type=Path)
    parser.add_argument("--baseline-cute", required=True, type=Path)
    parser.add_argument("--candidate-cute", required=True, type=Path)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.minimum_retention <= 1.0:
        parser.error("--minimum-retention must be in (0, 1]")

    baseline_math = load_report(args.baseline_math)
    candidate_math = load_report(args.candidate_math)
    baseline_cute = load_report(args.baseline_cute)
    candidate_cute = load_report(args.candidate_cute)
    if baseline_math["model_sha256"] != baseline_cute["model_sha256"]:
        raise RuntimeError("BF16 control model differs across quick gates")
    if candidate_math["model_sha256"] != candidate_cute["model_sha256"]:
        raise RuntimeError("candidate model differs across quick gates")

    gates = [
        evaluate_pair("math500", baseline_math, candidate_math, args.minimum_retention),
        evaluate_pair("cute_quick_20", baseline_cute, candidate_cute, args.minimum_retention),
    ]
    passed = all(gate["pass"] for gate in gates)
    report = {
        "schema": "cpu_capability_tripwire_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_label": args.candidate_label,
        "baseline_model_sha256": baseline_math["model_sha256"],
        "candidate_model_sha256": candidate_math["model_sha256"],
        "policy": {
            "purpose": "cheap catastrophic-forgetting stop gate before longer evaluation or scale-up",
            "minimum_retention": args.minimum_retention,
            "full_promotion_gate": "separate and stricter; this receipt cannot authorize promotion",
        },
        "pass": passed,
        "decision": "continue_to_full_evaluation" if passed else "stop_for_catastrophic_forgetting",
        "gates": gates,
        "inputs": {
            "baseline_math": str(args.baseline_math),
            "candidate_math": str(args.candidate_math),
            "baseline_cute": str(args.baseline_cute),
            "candidate_cute": str(args.candidate_cute),
        },
    }
    report["report_content_sha256"] = canonical_digest(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
