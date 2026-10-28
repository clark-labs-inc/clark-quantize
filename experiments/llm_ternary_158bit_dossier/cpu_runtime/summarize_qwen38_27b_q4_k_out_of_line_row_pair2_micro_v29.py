#!/usr/bin/env python3
"""Summarize the frozen v29 paired native Q4_K microkernel gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SHAPES = (
    (10240, 5120),
    (6144, 5120),
    (5120, 6144),
    (12288, 5120),
    (1024, 5120),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = read_json(args.preregistration)
    gate = preregistration["sequential_gates"]["gate_1_paired_native_microkernel"]
    expected_pairs = int(gate["measured_pairs_per_shape"])
    all_ratios: list[float] = []
    shape_results = []
    output_hashes_match = True

    for m, k in SHAPES:
        shape_id = f"m{m}-n4-k{k}"
        ratios = []
        parent_us = []
        candidate_us = []
        for pair in range(1, expected_pairs + 1):
            parent = read_json(
                args.state / "micro" / f"{shape_id}-measured-{pair}-parent.json"
            )
            candidate = read_json(
                args.state / "micro" / f"{shape_id}-measured-{pair}-candidate.json"
            )
            parent_sample = float(parent["samples_us"][0])
            candidate_sample = float(candidate["samples_us"][0])
            parent_us.append(parent_sample)
            candidate_us.append(candidate_sample)
            ratios.append(parent_sample / candidate_sample)
            output_hashes_match &= parent["output_fnv1a"] == candidate["output_fnv1a"]
        ratio_median = statistics.median(ratios)
        ratio_mad = median_absolute_deviation(ratios)
        all_ratios.extend(ratios)
        shape_results.append(
            {
                "m": m,
                "n": 4,
                "k": k,
                "pairs": len(ratios),
                "parent_latency_us_median": statistics.median(parent_us),
                "candidate_latency_us_median": statistics.median(candidate_us),
                "paired_parent_over_candidate_ratio_median": ratio_median,
                "paired_ratio_mad": ratio_mad,
                "paired_ratio_mad_fraction": ratio_mad / ratio_median,
            }
        )

    geometric_mean = math.exp(sum(math.log(value) for value in all_ratios) / len(all_ratios))
    minimum_shape_ratio = min(
        result["paired_parent_over_candidate_ratio_median"] for result in shape_results
    )
    maximum_mad_fraction = max(result["paired_ratio_mad_fraction"] for result in shape_results)
    frequency_path = args.state / "micro" / "frequency.log"
    frequencies = [
        float(line.split()[1])
        for line in frequency_path.read_text().splitlines()
        if len(line.split()) == 2
    ]
    checks = {
        "all_expected_pairs_present": len(all_ratios) == expected_pairs * len(SHAPES),
        "outputs_bit_exact": output_hashes_match,
        "geometric_mean_at_least_minimum": geometric_mean
        >= float(gate["minimum_geometric_mean_parent_over_candidate_latency_ratio"]),
        "no_shape_regresses_below_floor": minimum_shape_ratio
        >= float(gate["no_shape_may_regress_below_ratio"]),
        "paired_ratio_stability": maximum_mad_fraction
        <= float(gate["maximum_pairwise_ratio_mad_fraction"]),
    }
    result = {
        "schema": "qwen38_27b_q4_k_avx2_out_of_line_row_pair2_micro_result_v29",
        "experiment_id": preregistration["experiment_id"],
        "passed": all(checks.values()),
        "pairs": len(all_ratios),
        "geometric_mean_parent_over_candidate_latency_ratio": geometric_mean,
        "minimum_shape_ratio_median": minimum_shape_ratio,
        "maximum_pairwise_ratio_mad_fraction": maximum_mad_fraction,
        "frequency_mhz_minimum": min(frequencies) if frequencies else None,
        "frequency_mhz_maximum": max(frequencies) if frequencies else None,
        "checks": checks,
        "shapes": shape_results,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
