#!/usr/bin/env python3
"""Summarize the frozen property, microkernel, and Qwen graph gates for v24."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def mad_fraction(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def load_graph(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one llama-bench row in {path}")
    row = payload[0]
    if int(row["n_prompt"]) != 4 or int(row["n_gen"]) != 0:
        raise ValueError(f"{path} is not the frozen p4/n0 graph")
    samples = [int(value) for value in row["samples_ns"]]
    if len(samples) != 7:
        raise ValueError(f"expected seven graph samples in {path}, found {len(samples)}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
    }


def load_pairs(state: Path, comparison: str, left: str) -> dict[str, Any]:
    left_paths = sorted(state.glob(f"{comparison}-pair-*-{left}.json"))
    candidate_paths = sorted(state.glob(f"{comparison}-pair-*-v24.json"))
    if len(left_paths) != 6 or len(candidate_paths) != 6:
        raise ValueError(
            f"{comparison}: expected six {left} and six v24 processes, found "
            f"{len(left_paths)} and {len(candidate_paths)}"
        )
    left_rows = [load_graph(path) for path in left_paths]
    candidate_rows = [load_graph(path) for path in candidate_paths]
    ratios = [
        left_row["median_ns"] / candidate_row["median_ns"]
        for left_row, candidate_row in zip(left_rows, candidate_rows)
    ]
    return {
        "left_label": left,
        "left": left_rows,
        "candidate": candidate_rows,
        "paired_left_over_v24_ratios": ratios,
        "paired_ratio_median": statistics.median(ratios),
    }


def load_frequency(path: Path) -> dict[str, Any]:
    values: list[float] = []
    for line in path.read_text().splitlines():
        try:
            values.append(float(line.rsplit(maxsplit=1)[-1]))
        except (IndexError, ValueError):
            continue
    if not values:
        raise ValueError("frequency monitor produced no parseable samples")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": len(values),
        "minimum_mhz": min(values),
        "maximum_mhz": max(values),
        "range_mhz": max(values) - min(values),
        "median_mhz": statistics.median(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--property-log", type=Path, required=True)
    parser.add_argument("--micro-log", type=Path, required=True)
    parser.add_argument("--backend-log", type=Path, required=True)
    parser.add_argument("--frequency-log", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-patch", type=Path, required=True)
    parser.add_argument("--property-test", type=Path, required=True)
    parser.add_argument("--microbenchmark", type=Path, required=True)
    parser.add_argument("--control-binary", type=Path, required=True)
    parser.add_argument("--parent-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    property_text = args.property_log.read_text()
    backend_text = args.backend_log.read_text()
    micro = json.loads(args.micro_log.read_text())
    if micro.get("schema") != "q8_k_dual_layout_producer_microbenchmark_v24":
        raise ValueError("unexpected microbenchmark schema")
    if micro.get("paired_samples_per_width") != 31:
        raise ValueError("microbenchmark denominator changed")
    widths = [int(row["width"]) for row in micro["widths"]]
    if widths != [2048, 4096, 8192, 16384]:
        raise ValueError(f"microbenchmark widths changed: {widths}")

    parent = load_pairs(args.state, "parent", "v21")
    control = load_pairs(args.state, "control", "control")
    candidate_medians = [
        row["median_ns"]
        for comparison in (parent, control)
        for row in comparison["candidate"]
    ]
    frequency = load_frequency(args.frequency_log)

    checks = {
        "property_source_f32_unchanged": '"source_f32_unchanged":true' in property_text,
        "property_native_matches_parent_bit_exact": (
            '"native_matches_parent_bit_exact":true' in property_text
        ),
        "property_rows_match_parent_bridge_bit_exact": (
            '"row_major_matches_parent_bridge_bit_exact":true' in property_text
        ),
        "property_bsums_match_codes": (
            '"row_major_bsums_match_codes":true' in property_text
        ),
        "property_seeded_blocks_1024": '"seeded_blocks":1024' in property_text,
        "property_special_cases_6": '"special_cases":6' in property_text,
        "backend_reference_3_of_3": (
            "3/3 tests passed" in backend_text
            and "1/1 backends passed" in backend_text
            and "FAILED" not in backend_text
        ),
        "micro_parent_over_dual_geomean_at_least_1_10": (
            float(micro["parent_over_dual_geomean"]) >= 1.10
        ),
        "micro_each_width_has_31_paired_samples": all(
            len(row["parent_ns_per_call"]) == 31
            and len(row["dual_ns_per_call"]) == 31
            for row in micro["widths"]
        ),
        "v21_over_v24_graph_at_least_1_05": parent["paired_ratio_median"] >= 1.05,
        "control_over_v24_graph_at_least_1_18": control["paired_ratio_median"] >= 1.18,
        "candidate_process_median_mad_at_most_0_03": (
            mad_fraction(candidate_medians) <= 0.03
        ),
        "frequency_range_at_most_100_mhz": frequency["range_mhz"] <= 100.0,
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_q4_k_dual_layout_q8_producer_graph_result_v24",
        "stage": "property_microkernel_and_four_row_graph",
        "terminal": not passed,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "host_scope": "four physical AMD EPYC Milan cores",
        "denominator": {
            "property_seeded_blocks": 1024,
            "property_special_cases": 6,
            "micro_widths": widths,
            "micro_paired_samples_per_width": 31,
            "graph_processes_per_runtime_per_comparison": 6,
            "graph_samples_per_process": 7,
            "candidate_graph_processes_total": 12,
        },
        "microbenchmark": micro,
        "parent_v21_comparison": parent,
        "promoted_control_comparison": control,
        "candidate_process_median_mad_fraction": mad_fraction(candidate_medians),
        "frequency": frequency,
        "checks": checks,
        "passed": passed,
        "decision": "advance_to_system_gate" if passed else "reject_pre_system_gate",
        "next_stage": "v21-v24 target and MTP3 plus quick capability" if passed else None,
        "provenance": {
            "property_log_sha256": sha256_file(args.property_log),
            "micro_log_sha256": sha256_file(args.micro_log),
            "backend_log_sha256": sha256_file(args.backend_log),
            "preregistration_sha256": sha256_file(args.preregistration),
            "candidate_patch_sha256": sha256_file(args.candidate_patch),
            "property_test_sha256": sha256_file(args.property_test),
            "microbenchmark_source_sha256": sha256_file(args.microbenchmark),
            "control_binary_sha256": sha256_file(args.control_binary),
            "parent_binary_sha256": sha256_file(args.parent_binary),
            "candidate_binary_sha256": sha256_file(args.candidate_binary),
        },
        "claim_boundary": (
            "This stage measures a byte-exact Q8 producer and one four-row Q4_K "
            "Qwen graph. It is not target serving, a quality result, conventional "
            "decode, 100 tok/s, trained approximation, novelty, or breakthrough evidence."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "v24_graph_terminal", "passed": passed}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
