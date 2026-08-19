#!/usr/bin/env python3
"""Summarize AB/BA long-process Clang/libomp isolation for Qwen3.8-27B."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


LABELS = (
    "parent_a",
    "clang_default_a",
    "clang_active_a",
    "clang_active_b",
    "clang_default_b",
    "parent_b",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_samples(samples: list[int]) -> dict:
    candidates = []
    for split in range(8, 25):
        left = statistics.median(samples[:split])
        right = statistics.median(samples[split:])
        ratio = max(left / right, right / left)
        candidates.append((ratio, split, left, right))
    ratio, split, left, right = max(candidates)
    return {
        "samples": samples,
        "n": len(samples),
        "mean_ns": statistics.mean(samples),
        "median_ns": statistics.median(samples),
        "minimum_ns": min(samples),
        "maximum_ns": max(samples),
        "first_half_median_ns": statistics.median(samples[:16]),
        "last_half_median_ns": statistics.median(samples[16:]),
        "last_over_first_half": statistics.median(samples[16:])
        / statistics.median(samples[:16]),
        "maximum_changepoint": {
            "split": split,
            "pre_median_ns": left,
            "post_median_ns": right,
            "ratio": ratio,
        },
    }


def mad_fraction(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    runs = {}
    for label in LABELS:
        path = args.state / f"{label}.json"
        samples = [int(value) for value in json.loads(path.read_text())[0]["samples_ns"]]
        runs[label] = {**summarize_samples(samples), "sha256": sha256(path)}

    frequencies = [
        float(line.split()[1])
        for line in (args.state / "frequency.log").read_text().splitlines()
        if len(line.split()) == 2
    ]
    arms = {
        "parent": ["parent_a", "parent_b"],
        "clang_default": ["clang_default_a", "clang_default_b"],
        "clang_active": ["clang_active_a", "clang_active_b"],
    }
    arm_summaries = {}
    for name, labels in arms.items():
        medians = [runs[label]["median_ns"] for label in labels]
        means = [runs[label]["mean_ns"] for label in labels]
        arm_summaries[name] = {
            "run_medians_ns": medians,
            "median_of_run_medians_ns": statistics.median(medians),
            "mean_of_run_means_ns": statistics.mean(means),
            "run_median_mad_fraction": mad_fraction(medians),
            "maximum_changepoint_ratio": max(
                runs[label]["maximum_changepoint"]["ratio"] for label in labels
            ),
            "both_processes_stable": all(
                runs[label]["maximum_changepoint"]["ratio"]
                <= float(prereg["analysis"]["stable_process_maximum_changepoint_ratio"])
                for label in labels
            ),
        }
    parent_median = arm_summaries["parent"]["median_of_run_medians_ns"]
    for name in ("clang_default", "clang_active"):
        arm_summaries[name]["parent_over_arm_median"] = parent_median / arm_summaries[name][
            "median_of_run_medians_ns"
        ]
        arm_summaries[name]["speed_gate_passed"] = arm_summaries[name][
            "parent_over_arm_median"
        ] >= float(prereg["analysis"]["minimum_parent_over_candidate_arm_median"])
        arm_summaries[name]["stability_gate_passed"] = (
            arm_summaries[name]["both_processes_stable"]
            and arm_summaries[name]["run_median_mad_fraction"]
            <= float(prereg["analysis"]["stable_arm_maximum_run_median_mad_fraction"])
        )
    default_phase_reproduced = arm_summaries["clang_default"][
        "maximum_changepoint_ratio"
    ] >= float(prereg["analysis"]["default_phase_reproduction_ratio"])

    if (
        arm_summaries["clang_active"]["stability_gate_passed"]
        and arm_summaries["clang_active"]["speed_gate_passed"]
    ):
        decision = "select_clang_active_policy_for_serving_validation"
    elif (
        arm_summaries["clang_default"]["stability_gate_passed"]
        and arm_summaries["clang_default"]["speed_gate_passed"]
    ):
        decision = "select_clang_default_for_serving_validation"
    else:
        decision = "reject_clang_runtime_composition_retain_q4_representation_kernel_lane"

    frequency_range = max(frequencies) - min(frequencies)
    checks = {
        "six_processes_have_exact_denominator": all(runs[label]["n"] == 32 for label in LABELS),
        "frequency_range_within_limit": frequency_range
        <= float(prereg["validity_gates"]["frequency_range_mhz_maximum"]),
        "all_values_positive": all(
            value > 0 for label in LABELS for value in runs[label]["samples"]
        ),
    }
    result = {
        "schema": "qwen38_27b_clang21_libomp_longrun_result_v34",
        "experiment_id": prereg["experiment_id"],
        "terminal": True,
        "infrastructure_valid": all(checks.values()),
        "passed": decision.startswith("select_"),
        "checks": checks,
        "frequency": {
            "minimum_mhz": min(frequencies),
            "maximum_mhz": max(frequencies),
            "range_mhz": frequency_range,
            "samples": len(frequencies),
        },
        "runs": runs,
        "arms": arm_summaries,
        "default_phase_reproduced_without_perf": default_phase_reproduced,
        "decision": decision,
        "provenance": {
            "preregistration_sha256": sha256(args.preregistration),
            "sealed_inputs_sha256": sha256(args.state / "sealed-inputs.sha256"),
            "frequency_sha256": sha256(args.state / "frequency.log"),
        },
        "claim_boundary": prereg["claim_boundary"],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
