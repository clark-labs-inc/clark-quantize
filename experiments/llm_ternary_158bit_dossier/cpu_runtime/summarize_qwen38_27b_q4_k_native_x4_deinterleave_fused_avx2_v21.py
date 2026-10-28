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


def load_mtp_summary(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("terminal") is not True:
        raise ValueError(f"non-terminal MTP receipt at {root}")
    generation = summary["generation"]
    if generation["prompt_count"] != 4 or generation["tokens_per_prompt"] != 96:
        raise ValueError(f"non-frozen MTP denominator at {root}")
    return summary


def mtp_configuration(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return next(row for row in summary["configurations"] if row["configuration"] == label)


def token_rows(root: Path, label: str) -> list[list[int]]:
    return [
        json.loads((root / label / f"prompt-{index:02d}.json").read_text())["response"][
            "tokens"
        ]
        for index in range(4)
    ]


def token_digests(rows: list[list[int]]) -> list[str]:
    return [
        hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest()
        for row in rows
    ]


def first_mismatch_indices(
    control: list[list[int]], candidate: list[list[int]]
) -> list[int | None]:
    result: list[int | None] = []
    for control_row, candidate_row in zip(control, candidate):
        result.append(
            next(
                (
                    index
                    for index, (left, right) in enumerate(zip(control_row, candidate_row))
                    if left != right
                ),
                None,
            )
        )
    return result


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


def summarize_system(args: argparse.Namespace) -> int:
    graph = json.loads(args.graph_decision.read_text())
    if graph.get("passed") is not True or graph.get("terminal") is not False:
        raise ValueError("system stage requires a passed, non-terminal graph receipt")

    control_summary = load_mtp_summary(args.control_mtp)
    candidate_summary = load_mtp_summary(args.candidate_mtp)
    control_target = mtp_configuration(control_summary, "target")
    candidate_target = mtp_configuration(candidate_summary, "target")
    control_mtp3 = mtp_configuration(control_summary, "mtp3")
    candidate_mtp3 = mtp_configuration(candidate_summary, "mtp3")

    control_target_tokens = token_rows(args.control_mtp, "target")
    candidate_target_tokens = token_rows(args.candidate_mtp, "target")
    control_mtp3_tokens = token_rows(args.control_mtp, "mtp3")
    candidate_mtp3_tokens = token_rows(args.candidate_mtp, "mtp3")

    target_ratio = (
        candidate_target["accepted_tokens_per_second_weighted"]
        / control_target["accepted_tokens_per_second_weighted"]
    )
    mtp3_ratio = (
        candidate_mtp3["accepted_tokens_per_second_weighted"]
        / control_mtp3["accepted_tokens_per_second_weighted"]
    )
    target_exact = control_target_tokens == candidate_target_tokens
    mtp3_exact = control_mtp3_tokens == candidate_mtp3_tokens
    candidate_target_zero_fraction = (
        sum(token == 0 for row in candidate_target_tokens for token in row)
        / sum(len(row) for row in candidate_target_tokens)
    )
    candidate_mtp3_zero_fraction = (
        sum(token == 0 for row in candidate_mtp3_tokens for token in row)
        / sum(len(row) for row in candidate_mtp3_tokens)
    )

    checks = {
        "graph_gate_passed": True,
        "target_only_cross_runtime_tokens_exact": target_exact,
        "target_only_zero_fraction_below_0_01": candidate_target_zero_fraction < 0.01,
        "target_only_regression_at_most_0_02": target_ratio >= 0.98,
        "mtp3_candidate_over_control_at_least_1_05": mtp3_ratio >= 1.05,
        "candidate_target_denominator_is_4x96": candidate_target[
            "output_tokens_including_prompt_eval_token"
        ]
        == 384,
        "candidate_mtp3_denominator_is_4x96": candidate_mtp3[
            "output_tokens_including_prompt_eval_token"
        ]
        == 384,
        "candidate_mtp3_zero_fraction_below_0_01": candidate_mtp3_zero_fraction < 0.01,
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_system_result_v21",
        "stage": "target_and_mtp3_system",
        "terminal": not passed,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "host_scope": "four physical AMD EPYC Milan cores",
        "denominator": {"prompts": 4, "tokens_per_prompt": 96},
        "graph_gate": {
            "paired_control_over_candidate_ratio_median": graph["paired_ratio_median"],
            "receipt_sha256": sha256_file(args.graph_decision),
        },
        "target_only": {
            "control_accepted_tokens_per_second": control_target[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_accepted_tokens_per_second": candidate_target[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_over_control": target_ratio,
            "cross_runtime_tokens_exact": target_exact,
            "candidate_zero_token_fraction": candidate_target_zero_fraction,
            "first_mismatch_index_by_prompt": first_mismatch_indices(
                control_target_tokens, candidate_target_tokens
            ),
        },
        "mtp3": {
            "control_accepted_tokens_per_second": control_mtp3[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_accepted_tokens_per_second": candidate_mtp3[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_over_control": mtp3_ratio,
            "cross_runtime_tokens_exact": mtp3_exact,
            "first_mismatch_index_by_prompt": first_mismatch_indices(
                control_mtp3_tokens, candidate_mtp3_tokens
            ),
            "control_draft_n": control_mtp3["draft_n"],
            "candidate_draft_n": candidate_mtp3["draft_n"],
            "control_draft_n_accepted": control_mtp3["draft_n_accepted"],
            "candidate_draft_n_accepted": candidate_mtp3["draft_n_accepted"],
            "control_mean_committed_tokens_per_target_call": control_mtp3[
                "mean_committed_tokens_per_target_call"
            ],
            "candidate_mean_committed_tokens_per_target_call": candidate_mtp3[
                "mean_committed_tokens_per_target_call"
            ],
            "candidate_zero_token_fraction": candidate_mtp3_zero_fraction,
        },
        "checks": checks,
        "passed": passed,
        "decision": "advance_to_quality_gate" if passed else "reject_system_gate",
        "next_stage": "CUTE-20 and four-chunk WikiText" if passed else None,
        "provenance": {
            "control_server_sha256": sha256_file(args.control_server),
            "candidate_server_sha256": sha256_file(args.candidate_server),
            "control_summary_sha256": sha256_file(args.control_mtp / "summary.json"),
            "candidate_summary_sha256": sha256_file(args.candidate_mtp / "summary.json"),
            "control_target_token_sha256": token_digests(control_target_tokens),
            "candidate_target_token_sha256": token_digests(candidate_target_tokens),
            "control_mtp3_token_sha256": token_digests(control_mtp3_tokens),
            "candidate_mtp3_token_sha256": token_digests(candidate_mtp3_tokens),
        },
        "claim_boundary": (
            "A pass admits the same-weights kernel to frozen CUTE and WikiText. "
            "It is not a conventional-decode breakthrough, trained approximation, "
            "final-node, energy, 100 tok/s, novelty, or scientific-breakthrough result."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "system_terminal" if not passed else "system_passed", "passed": passed}))
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("graph", "system"), required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--property-log", type=Path)
    parser.add_argument("--backend-log", type=Path)
    parser.add_argument("--frequency-log", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--layout-amendment", type=Path)
    parser.add_argument("--candidate-patch", type=Path)
    parser.add_argument("--layout-test", type=Path)
    parser.add_argument("--control-binary", type=Path)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--graph-decision", type=Path)
    parser.add_argument("--control-mtp", type=Path)
    parser.add_argument("--candidate-mtp", type=Path)
    parser.add_argument("--control-server", type=Path)
    parser.add_argument("--candidate-server", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "graph":
        return summarize_graph(args)
    if args.stage == "system":
        return summarize_system(args)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
