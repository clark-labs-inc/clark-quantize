#!/usr/bin/env python3
"""Summarize the v31 paired Qwen3.8-27B four-row graph gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--parent-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = json.loads(args.preregistration.read_text())
    gate = preregistration["sequential_gates"]["gate_2_paired_real_model_four_row_graph"]
    records = {"parent": [], "candidate": []}
    for pair in range(1, int(gate["paired_processes_per_runtime"]) + 1):
        for label in records:
            path = args.state / "graph" / f"measured-{pair}-{label}.json"
            item = json.loads(path.read_text())[0]
            samples = [int(value) for value in item["samples_ns"]]
            if len(samples) != int(gate["samples_per_process"]):
                raise RuntimeError(f"unexpected denominator in {path}")
            records[label].append(
                {
                    "pair": pair,
                    "median_ns": statistics.median(samples),
                    "mean_ns": statistics.mean(samples),
                    "samples_ns": samples,
                    "sha256": sha256(path),
                }
            )

    ratios = [
        records["parent"][index]["median_ns"] / records["candidate"][index]["median_ns"]
        for index in range(len(records["parent"]))
    ]
    candidate_medians = [record["median_ns"] for record in records["candidate"]]
    candidate_center = statistics.median(candidate_medians)
    candidate_mad_fraction = mad(candidate_medians) / candidate_center
    frequencies = [
        float(line.split()[1])
        for line in (args.state / "graph" / "frequency.log").read_text().splitlines()
        if len(line.split()) == 2
    ]
    frequency_range = max(frequencies) - min(frequencies)
    ratio_median = statistics.median(ratios)
    checks = {
        "paired_graph_speedup_at_least_minimum": ratio_median
        >= float(gate["minimum_parent_over_candidate_latency_ratio"]),
        "candidate_process_median_stability": candidate_mad_fraction
        <= float(gate["maximum_candidate_process_median_mad_fraction"]),
        "frequency_range_within_limit": frequency_range
        <= float(gate["maximum_frequency_range_mhz"]),
    }
    result = {
        "schema": "qwen38_27b_q4_k_avx2_clang21_holdout_graph_result_v31",
        "experiment_id": preregistration["experiment_id"],
        "stage": "four_row_graph",
        "infrastructure_valid": True,
        "passed": all(checks.values()),
        "paired_parent_over_candidate_ratios": ratios,
        "paired_ratio_median": ratio_median,
        "candidate_process_median_mad_fraction": candidate_mad_fraction,
        "frequency": {
            "minimum_mhz": min(frequencies),
            "maximum_mhz": max(frequencies),
            "range_mhz": frequency_range,
            "sample_count": len(frequencies),
        },
        "denominator": {
            "processes_per_runtime": len(records["parent"]),
            "samples_per_process": int(gate["samples_per_process"]),
            "samples_per_runtime": len(records["parent"]) * int(gate["samples_per_process"]),
        },
        "checks": checks,
        "parent": records["parent"],
        "candidate": records["candidate"],
        "provenance": {
            "preregistration_sha256": sha256(args.preregistration),
            "parent_binary_sha256": sha256(args.parent_binary),
            "candidate_binary_sha256": sha256(args.candidate_binary),
        },
        "claim_boundary": "Exact same-weights Qwen3.8-27B four-row Q4_K graph on four Milan cores; not serving, quality, conventional decode, 100 tok/s, or novelty.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
