#!/usr/bin/env python3
"""Summarize the paired post-Clang Qwen3.8-27B cycle profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


GROUP_PATTERNS = {
    "q4_k_verifier": (
        "ggml_gemm_q4_K_8x8_q8_K",
        "ggml_gemv_q4_K_8x8_q8_K",
        "ggml_gemv_q4_K_8x8_q8_K_pair_local_scaled_pair2_tile",
    ),
    "q8_x_q8_tinyblas": ("tinyBLAS_Q0_AVX<block_q8_0, block_q8_0",),
    "q6_k": ("ggml_vec_dot_q6_K_q8_K",),
    "openmp": ("libgomp.so", "GOMP_", "libomp.so", "__kmp"),
    "load_repack": ("repack_q4_K_to_q4_K_8_bl",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lost_match = re.search(r"Total Lost Samples:\s*([0-9]+)", text)
    sample_match = re.search(
        r"^# Samples:\s*([0-9]+(?:\.[0-9]+)?)([KMG]?)",
        text,
        flags=re.MULTILINE,
    )
    if not lost_match or not sample_match:
        raise RuntimeError(f"missing perf headers in {path}")
    multiplier = {"": 1, "K": 1000, "M": 1_000_000, "G": 1_000_000_000}
    header_samples = int(float(sample_match.group(1)) * multiplier[sample_match.group(2)])
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") or ";" not in line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 4 or not parts[0].endswith("%"):
            continue
        try:
            overhead = float(parts[0][:-1]) / 100.0
            samples = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "overhead": overhead,
                "samples": samples,
                "dso": parts[2],
                "symbol": ";".join(parts[3:]),
            }
        )
    if not rows:
        raise RuntimeError(f"no parseable perf rows in {path}")
    groups = {name: 0.0 for name in GROUP_PATTERNS}
    for row in rows:
        haystack = f"{row['dso']} {row['symbol']}"
        for name, patterns in GROUP_PATTERNS.items():
            if any(pattern in haystack for pattern in patterns):
                groups[name] += row["overhead"]
                break
    return {
        "header_samples": header_samples,
        "lost_samples": int(lost_match.group(1)),
        "parsed_overhead": sum(row["overhead"] for row in rows),
        "groups": groups,
        "top_symbols": sorted(rows, key=lambda row: row["overhead"], reverse=True)[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = json.loads(args.preregistration.read_text())
    amendment = json.loads(args.amendment.read_text())
    validity = preregistration["validity_gates"]
    immutable = amendment["immutable_inputs"]
    immutable_paths = {
        "parent_bench_sha256": args.state / "parent.bench.json",
        "candidate_bench_sha256": args.state / "candidate.bench.json",
        "parent_report_sha256": args.state / "parent.perf-report.csv",
        "candidate_report_sha256": args.state / "candidate.perf-report.csv",
    }
    for key, path in immutable_paths.items():
        observed = sha256(path)
        if observed != immutable[key]:
            raise RuntimeError(f"immutable input mismatch for {path}: {observed}")
    profiles = {}
    frequencies = [
        float(line.split()[1])
        for line in (args.state / "frequency.log").read_text().splitlines()
        if len(line.split()) == 2
    ]
    for label in ("parent", "candidate"):
        report = args.state / f"{label}.perf-report.csv"
        bench = args.state / f"{label}.bench.json"
        profile = parse_report(report)
        bench_payload = json.loads(bench.read_text())[0]
        samples_ns = [int(value) for value in bench_payload["samples_ns"]]
        profile.update(
            {
                "bench_samples": len(samples_ns),
                "bench_median_ns": statistics.median(samples_ns),
                "bench_mean_ns": statistics.mean(samples_ns),
                "bench_sha256": sha256(bench),
                "report_sha256": sha256(report),
            }
        )
        profiles[label] = profile

    q4_speedup = float(preregistration["prior_evidence"]["v31_native_kernel_speedup"])
    parent_groups = profiles["parent"]["groups"]
    q4_share = parent_groups["q4_k_verifier"]
    q8_share = parent_groups["q8_x_q8_tinyblas"]
    denominator_q4_only = 1.0 - q4_share + q4_share / q4_speedup
    denominator_q4_q8_free = 1.0 - q4_share - q8_share + q4_share / q4_speedup
    q4_only_amdahl = 1.0 / denominator_q4_only
    q4_q8_free_amdahl = 1.0 / denominator_q4_q8_free
    q8_allowed_at_target = (1.0 / 1.40) - denominator_q4_q8_free
    required_q8_speedup = q8_share / q8_allowed_at_target if q8_allowed_at_target > 0 else math.inf
    current_graph_ratio = (
        profiles["parent"]["bench_median_ns"] / profiles["candidate"]["bench_median_ns"]
    )
    required_additional_candidate_speedup = 1.40 / current_graph_ratio
    frequency_range = max(frequencies) - min(frequencies)
    checks = {
        "both_profiles_have_minimum_samples": all(
            profile["header_samples"] >= int(validity["samples_per_runtime_minimum"])
            for profile in profiles.values()
        ),
        "zero_lost_samples": all(profile["lost_samples"] == 0 for profile in profiles.values()),
        "bench_denominators_exact": all(
            profile["bench_samples"] == int(validity["llama_bench_samples_per_runtime"])
            for profile in profiles.values()
        ),
        "frequency_range_within_limit": frequency_range
        <= float(validity["frequency_range_mhz_maximum"]),
    }
    result = {
        "schema": "qwen38_27b_q4_k_clang21_full_graph_profile_result_v32",
        "experiment_id": preregistration["experiment_id"],
        "terminal": True,
        "infrastructure_valid": all(checks.values()),
        "passed": all(checks.values()),
        "checks": checks,
        "frequency": {
            "minimum_mhz": min(frequencies),
            "maximum_mhz": max(frequencies),
            "range_mhz": frequency_range,
            "samples": len(frequencies),
        },
        "profiles": profiles,
        "paired_bench_parent_over_candidate": current_graph_ratio,
        "amdahl": {
            "confirmed_q4_native_speedup": q4_speedup,
            "q4_only_parent_relative_ceiling": q4_only_amdahl,
            "q4_plus_infinitely_fast_q8_parent_relative_ceiling": q4_q8_free_amdahl,
            "required_q8_speedup_given_confirmed_q4_for_1_40x": required_q8_speedup,
            "required_additional_candidate_speedup_for_1_40x": required_additional_candidate_speedup,
            "q4_plus_q8_alone_can_reach_1_40x": q4_q8_free_amdahl >= 1.40,
        },
        "decision": (
            "retain_multi_kernel_lane"
            if q4_q8_free_amdahl >= 1.40
            else "reject_two_kernel_only_lane_require_approximation_or_three_plus_groups"
        ),
        "claim_boundary": "Sampling-based bottleneck and Amdahl diagnostic only; not a speed, quality, energy, or novelty result.",
        "provenance": {
            "preregistration_sha256": sha256(args.preregistration),
            "analysis_amendment_sha256": sha256(args.amendment),
            "frequency_sha256": sha256(args.state / "frequency.log"),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
