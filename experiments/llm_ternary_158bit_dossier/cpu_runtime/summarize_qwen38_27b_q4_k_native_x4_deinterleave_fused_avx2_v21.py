#!/usr/bin/env python3
"""Summarize frozen stages of the exact native-Q8_Kx4 four-row experiment."""

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


def load_bench(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one llama-bench row in {path}")
    row = payload[0]
    if int(row["n_prompt"]) != 4 or int(row["n_gen"]) != 0:
        raise ValueError(f"{path} is not the frozen p4/n0 graph")
    samples = [int(value) for value in row["samples_ns"]]
    if len(samples) != 7:
        raise ValueError(f"expected seven samples in {path}, found {len(samples)}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
    }


def frequency_summary(path: Path) -> dict[str, Any]:
    values: list[float] = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                values.append(float(line.rsplit(maxsplit=1)[-1]))
            except (ValueError, IndexError):
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


def summarize_graph(args: argparse.Namespace) -> int:
    control_paths = sorted(args.state.glob("pair-*-control.json"))
    candidate_paths = sorted(args.state.glob("pair-*-candidate.json"))
    if len(control_paths) != 6 or len(candidate_paths) != 6:
        raise ValueError(
            f"expected six processes per runtime, found {len(control_paths)} control "
            f"and {len(candidate_paths)} candidate"
        )
    control = [load_bench(path) for path in control_paths]
    candidate = [load_bench(path) for path in candidate_paths]
    ratios = [
        control_row["median_ns"] / candidate_row["median_ns"]
        for control_row, candidate_row in zip(control, candidate)
    ]
    candidate_medians = [row["median_ns"] for row in candidate]
    frequency = frequency_summary(args.frequency_log)

    property_text = args.property_log.read_text()
    backend_text = args.backend_log.read_text()
    checks = {
        "layout_property_bit_exact": all(
            marker in property_text
            for marker in (
                '"seeded_native_blocks":1024',
                '"source_unchanged":true',
                '"mapping_bit_exact":true',
                '"bsums_match_codes":true',
                '"inverse_roundtrip_bit_exact":true',
            )
        ),
        "backend_reference_3_of_3": (
            "3/3 tests passed" in backend_text
            and "1/1 backends passed" in backend_text
            and "FAILED" not in backend_text
        ),
        "paired_graph_speedup_at_least_1_10": statistics.median(ratios) >= 1.10,
        "candidate_process_median_mad_at_most_0_03": mad_fraction(candidate_medians)
        <= 0.03,
        "frequency_range_at_most_100_mhz": frequency["range_mhz"] <= 100.0,
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_graph_result_v21",
        "stage": "four_row_graph",
        "terminal": not passed,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "host_scope": "four physical AMD EPYC Milan cores",
        "denominator": {
            "processes_per_runtime": 6,
            "samples_per_process": 7,
            "samples_per_runtime": 42,
            "paired_process_ratios": 6,
        },
        "control": control,
        "candidate": candidate,
        "paired_control_over_candidate_ratios": ratios,
        "paired_ratio_median": statistics.median(ratios),
        "candidate_process_median_mad_fraction": mad_fraction(candidate_medians),
        "frequency": frequency,
        "checks": checks,
        "passed": passed,
        "provenance": {
            "property_log_sha256": sha256_file(args.property_log),
            "backend_log_sha256": sha256_file(args.backend_log),
            "preregistration_sha256": sha256_file(args.preregistration),
            "layout_amendment_sha256": sha256_file(args.layout_amendment),
            "candidate_patch_sha256": sha256_file(args.candidate_patch),
            "layout_test_sha256": sha256_file(args.layout_test),
            "control_binary_sha256": sha256_file(args.control_binary),
            "candidate_binary_sha256": sha256_file(args.candidate_binary),
        },
        "next_stage": "target-only and MTP3 system gates" if passed else None,
        "decision": "advance_to_system_gate" if passed else "reject_graph_gate",
        "claim_boundary": (
            "This stage tests one exact-four-row same-weights Q4_K graph on four "
            "Milan cores. It is not a whole-model serving, quality, conventional "
            "decode, 100 tok/s, novelty, or breakthrough result."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "graph_terminal" if not passed else "graph_passed", "passed": passed}))
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("graph",), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--property-log", type=Path, required=True)
    parser.add_argument("--backend-log", type=Path, required=True)
    parser.add_argument("--frequency-log", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--layout-amendment", type=Path, required=True)
    parser.add_argument("--candidate-patch", type=Path, required=True)
    parser.add_argument("--layout-test", type=Path, required=True)
    parser.add_argument("--control-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "graph":
        return summarize_graph(args)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
