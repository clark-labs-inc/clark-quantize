#!/usr/bin/env python3
"""Build the terminal paired forgetting/recovery receipt for the pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "large_scale_ternary"))

from gate_b_protocol import digest_json, sha256_file  # noqa: E402


def load(path: Path, expected_arm: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("schema") != "qwen38_capability_forgetting_eval_v1":
        raise ValueError(f"unexpected report schema in {path}")
    if report.get("arm") != expected_arm:
        raise ValueError(f"expected arm {expected_arm}, got {report.get('arm')} in {path}")
    return report


def keyed(result: dict) -> dict[str, dict]:
    rows = {row["sample_id"]: row for row in result["samples"]}
    if len(rows) != result["n"]:
        raise ValueError(f"duplicate sample IDs in {result['task']}")
    return rows


def paired(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if set(reference) != set(candidate):
        raise ValueError("paired reports have different sample IDs")
    both_correct = ref_only = candidate_only = both_wrong = 0
    for sample_id in sorted(reference):
        before = bool(reference[sample_id]["correct"])
        after = bool(candidate[sample_id]["correct"])
        if before and after:
            both_correct += 1
        elif before:
            ref_only += 1
        elif after:
            candidate_only += 1
        else:
            both_wrong += 1
    return {
        "both_correct": both_correct,
        "reference_only_correct": ref_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "net_correct_change": candidate_only - ref_only,
    }


def generation_collapse(result: dict) -> dict | None:
    if not any("output" in row for row in result["samples"]):
        return None
    outputs = [str(row.get("output") or "") for row in result["samples"]]
    if not outputs:
        return None
    token_counts = [int(row.get("output_tokens") or 0) for row in result["samples"]]
    normalized = [output.strip() for output in outputs]
    counts = Counter(normalized)
    most_common, frequency = counts.most_common(1)[0]
    return {
        "nonempty": sum(bool(output) for output in normalized),
        "unique_outputs": len(counts),
        "median_output_tokens": sorted(token_counts)[len(token_counts) // 2],
        "maximum_output_tokens": max(token_counts),
        "most_common_output": most_common,
        "most_common_output_count": frequency,
        "most_common_output_fraction": frequency / len(outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--ptq", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reports = {
        "bf16": load(args.bf16, "bf16"),
        "base3_ptq": load(args.ptq, "base3_ptq"),
        "trained": load(args.trained, "qat_patch"),
    }
    manifest_ids = {name: report["manifest"]["content_sha256"] for name, report in reports.items()}
    if len(set(manifest_ids.values())) != 1:
        raise ValueError(f"manifest mismatch: {manifest_ids}")
    by_arm = {
        arm: {result["task"]: result for result in report["results"]}
        for arm, report in reports.items()
    }
    tasks = sorted(by_arm["bf16"])
    if any(set(results) != set(tasks) for results in by_arm.values()):
        raise ValueError("task sets differ across arms")

    task_rows = []
    promotion_failures = []
    for task in tasks:
        bf16 = by_arm["bf16"][task]
        ptq = by_arm["base3_ptq"][task]
        trained = by_arm["trained"][task]
        if len({bf16["n"], ptq["n"], trained["n"]}) != 1:
            raise ValueError(f"sample count mismatch for {task}")
        bf16_rows, ptq_rows, trained_rows = keyed(bf16), keyed(ptq), keyed(trained)
        forgetting_pp = 100.0 * (trained["accuracy"] - bf16["accuracy"])
        recovery_pp = 100.0 * (trained["accuracy"] - ptq["accuracy"])
        if forgetting_pp < -5.0:
            promotion_failures.append(f"{task}: trained accuracy is {abs(forgetting_pp):.3f} pp below BF16")
        task_rows.append({
            "task": task,
            "n": trained["n"],
            "bf16": {"correct": bf16["correct"], "accuracy": bf16["accuracy"]},
            "base3_ptq": {"correct": ptq["correct"], "accuracy": ptq["accuracy"]},
            "trained": {"correct": trained["correct"], "accuracy": trained["accuracy"]},
            "catastrophic_forgetting_vs_bf16_percentage_points": forgetting_pp,
            "recovery_vs_base3_ptq_percentage_points": recovery_pp,
            "paired_vs_bf16": paired(bf16_rows, trained_rows),
            "paired_vs_base3_ptq": paired(ptq_rows, trained_rows),
            "trained_generation_collapse": generation_collapse(trained),
        })

    totals = {
        arm: {
            "correct": sum(result["correct"] for result in by_arm[arm].values()),
            "n": sum(result["n"] for result in by_arm[arm].values()),
        }
        for arm in by_arm
    }
    for value in totals.values():
        value["micro_accuracy"] = value["correct"] / value["n"]
    comparison = {
        "schema": "qwen38_global_pilot_forgetting_comparison_v1",
        "manifest_content_sha256": next(iter(manifest_ids.values())),
        "inputs": {
            "bf16": {"path": str(args.bf16), "sha256": sha256_file(args.bf16), "report_content_sha256": reports["bf16"]["report_content_sha256"]},
            "base3_ptq": {"path": str(args.ptq), "sha256": sha256_file(args.ptq), "report_content_sha256": reports["base3_ptq"]["report_content_sha256"]},
            "trained": {"path": str(args.trained), "sha256": sha256_file(args.trained), "report_content_sha256": reports["trained"]["report_content_sha256"]},
        },
        "trained_transform": reports["trained"]["transform"],
        "tasks": task_rows,
        "aggregate": {
            **totals,
            "catastrophic_forgetting_vs_bf16_percentage_points": 100.0 * (totals["trained"]["micro_accuracy"] - totals["bf16"]["micro_accuracy"]),
            "recovery_vs_base3_ptq_percentage_points": 100.0 * (totals["trained"]["micro_accuracy"] - totals["base3_ptq"]["micro_accuracy"]),
        },
        "promotion": {
            "maximum_allowed_task_forgetting_percentage_points": 5.0,
            "passed": not promotion_failures,
            "failures": promotion_failures,
        },
    }
    comparison["content_sha256"] = digest_json(comparison)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
