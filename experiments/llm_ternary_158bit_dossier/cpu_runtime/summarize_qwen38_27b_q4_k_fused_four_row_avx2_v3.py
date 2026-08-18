#!/usr/bin/env python3
"""Summarize the paired Qwen3.8-27B four-row Q4_K AVX2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
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


def parse_bench(paths: list[Path]) -> dict[str, Any]:
    samples: list[int] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"expected one benchmark row in {path}")
        row = payload[0]
        if int(row["n_prompt"]) != 4 or int(row["n_gen"]) != 0:
            raise ValueError(f"{path} is not the frozen four-row prompt graph")
        samples.extend(int(value) for value in row["samples_ns"])
    if len(samples) != 14:
        raise ValueError(f"expected 14 paired graph samples, got {len(samples)}")
    return {
        "samples_ns": samples,
        "sample_count": len(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
    }


def load_summary(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("terminal") is not True:
        raise ValueError(f"non-terminal MTP summary: {root}")
    generation = summary["generation"]
    if generation["prompt_count"] != 4 or generation["tokens_per_prompt"] != 96:
        raise ValueError(f"non-frozen 4 x 96 denominator: {root}")
    return summary


def configuration(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return next(
        row for row in summary["configurations"] if row["configuration"] == label
    )


def token_rows(root: Path, label: str) -> list[list[int]]:
    return [
        json.loads((root / label / f"prompt-{index:02d}.json").read_text())["response"][
            "tokens"
        ]
        for index in range(4)
    ]


def token_digests(rows: list[list[int]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(row, separators=(",", ":")).encode()
        ).hexdigest()
        for row in rows
    ]


def prompt_receipt_digests(root: Path, label: str) -> list[str]:
    return [sha256_file(root / label / f"prompt-{index:02d}.json") for index in range(4)]


def first_mismatch_indices(
    control: list[list[int]], candidate: list[list[int]]
) -> list[int | None]:
    result: list[int | None] = []
    for control_row, candidate_row in zip(control, candidate):
        result.append(
            next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(control_row, candidate_row)
                    )
                    if left != right
                ),
                None,
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-bench", type=Path, action="append", required=True)
    parser.add_argument("--candidate-bench", type=Path, action="append", required=True)
    parser.add_argument("--backend-log", type=Path, required=True)
    parser.add_argument("--control-mtp", type=Path, required=True)
    parser.add_argument("--candidate-mtp", type=Path, required=True)
    parser.add_argument("--control-server", type=Path, required=True)
    parser.add_argument("--candidate-server", type=Path, required=True)
    parser.add_argument("--base-patch", type=Path, required=True)
    parser.add_argument("--fused-patch", type=Path, required=True)
    parser.add_argument("--four-row-patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_graph = parse_bench(args.control_bench)
    candidate_graph = parse_bench(args.candidate_bench)
    graph_speedup = control_graph["median_ns"] / candidate_graph["median_ns"]

    backend_text = args.backend_log.read_text()
    backend_reference_passed = (
        "3/3 tests passed" in backend_text
        and "1/1 backends passed" in backend_text
        and "FAILED" not in backend_text
    )

    control_summary = load_summary(args.control_mtp)
    candidate_summary = load_summary(args.candidate_mtp)
    control_target = configuration(control_summary, "target")
    candidate_target = configuration(candidate_summary, "target")
    control_mtp3 = configuration(control_summary, "mtp3")
    candidate_mtp3 = configuration(candidate_summary, "mtp3")

    target_ratio = (
        candidate_target["accepted_tokens_per_second_weighted"]
        / control_target["accepted_tokens_per_second_weighted"]
    )
    mtp3_ratio = (
        candidate_mtp3["accepted_tokens_per_second_weighted"]
        / control_mtp3["accepted_tokens_per_second_weighted"]
    )
    control_target_tokens = token_rows(args.control_mtp, "target")
    candidate_target_tokens = token_rows(args.candidate_mtp, "target")
    control_mtp3_tokens = token_rows(args.control_mtp, "mtp3")
    candidate_mtp3_tokens = token_rows(args.candidate_mtp, "mtp3")
    target_exact = control_target_tokens == candidate_target_tokens
    mtp3_exact = control_mtp3_tokens == candidate_mtp3_tokens
    parity_pattern_equal = (
        control_mtp3["prompt_token_parity"] == candidate_mtp3["prompt_token_parity"]
    )
    draft_behavior_equal = (
        control_mtp3["draft_n"] == candidate_mtp3["draft_n"]
        and control_mtp3["draft_n_accepted"] == candidate_mtp3["draft_n_accepted"]
    )

    checks = {
        "backend_reference_cases_pass": backend_reference_passed,
        "four_row_graph_speedup_at_least_1_10": graph_speedup >= 1.10,
        "target_only_regression_at_most_0_02": target_ratio >= 0.98,
        "whole_model_mtp3_speedup_at_least_1_05": mtp3_ratio >= 1.05,
        "cross_runtime_target_exact_token_parity": target_exact,
        "cross_runtime_mtp3_exact_token_parity": mtp3_exact,
        "internal_parity_pattern_unchanged": parity_pattern_equal,
        "draft_behavior_unchanged": draft_behavior_equal,
    }
    payload = {
        "schema": "qwen38_27b_q4_k_fused_four_row_avx2_result_v3",
        "terminal": True,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "host_scope": "four physical AMD EPYC Milan cores",
        "shape": {"mtp_draft_width": 3, "target_verification_rows": 4},
        "provenance": {
            "control_server_sha256": sha256_file(args.control_server),
            "candidate_server_sha256": sha256_file(args.candidate_server),
            "base_patch_sha256": sha256_file(args.base_patch),
            "fused_patch_sha256": sha256_file(args.fused_patch),
            "four_row_patch_sha256": sha256_file(args.four_row_patch),
            "backend_log_sha256": sha256_file(args.backend_log),
            "control_bench_sha256": [sha256_file(path) for path in args.control_bench],
            "candidate_bench_sha256": [
                sha256_file(path) for path in args.candidate_bench
            ],
            "control_summary_sha256": sha256_file(args.control_mtp / "summary.json"),
            "candidate_summary_sha256": sha256_file(
                args.candidate_mtp / "summary.json"
            ),
            "control_target_prompt_receipt_sha256": prompt_receipt_digests(
                args.control_mtp, "target"
            ),
            "candidate_target_prompt_receipt_sha256": prompt_receipt_digests(
                args.candidate_mtp, "target"
            ),
            "control_mtp3_prompt_receipt_sha256": prompt_receipt_digests(
                args.control_mtp, "mtp3"
            ),
            "candidate_mtp3_prompt_receipt_sha256": prompt_receipt_digests(
                args.candidate_mtp, "mtp3"
            ),
        },
        "real_model_four_row_graph": {
            "control": control_graph,
            "candidate": candidate_graph,
            "speedup": graph_speedup,
        },
        "whole_model": {
            "prompt_count": 4,
            "tokens_per_prompt": 96,
            "control_target_accepted_tokens_per_second": control_target[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_target_accepted_tokens_per_second": candidate_target[
                "accepted_tokens_per_second_weighted"
            ],
            "target_only_candidate_over_control": target_ratio,
            "control_mtp3_accepted_tokens_per_second": control_mtp3[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_mtp3_accepted_tokens_per_second": candidate_mtp3[
                "accepted_tokens_per_second_weighted"
            ],
            "mtp3_candidate_over_control": mtp3_ratio,
            "cross_runtime_target_exact_token_parity": target_exact,
            "cross_runtime_mtp3_exact_token_parity": mtp3_exact,
            "control_internal_target_vs_mtp3_prompt_parity": control_mtp3[
                "prompt_token_parity"
            ],
            "candidate_internal_target_vs_mtp3_prompt_parity": candidate_mtp3[
                "prompt_token_parity"
            ],
            "control_draft_n": control_mtp3["draft_n"],
            "candidate_draft_n": candidate_mtp3["draft_n"],
            "control_draft_n_accepted": control_mtp3["draft_n_accepted"],
            "candidate_draft_n_accepted": candidate_mtp3["draft_n_accepted"],
        },
        "token_diagnostics": {
            "control_target_token_sha256": token_digests(control_target_tokens),
            "candidate_target_token_sha256": token_digests(candidate_target_tokens),
            "control_mtp3_token_sha256": token_digests(control_mtp3_tokens),
            "candidate_mtp3_token_sha256": token_digests(candidate_mtp3_tokens),
            "target_first_mismatch_index_by_prompt": first_mismatch_indices(
                control_target_tokens, candidate_target_tokens
            ),
            "mtp3_first_mismatch_index_by_prompt": first_mismatch_indices(
                control_mtp3_tokens, candidate_mtp3_tokens
            ),
            "candidate_target_unique_token_ids": sorted(
                {token for row in candidate_target_tokens for token in row}
            ),
            "candidate_target_zero_token_fraction": (
                sum(token == 0 for row in candidate_target_tokens for token in row)
                / sum(len(row) for row in candidate_target_tokens)
            ),
        },
        "checks": checks,
        "kernel_promoted": all(checks.values()),
        "cute_required": all(checks.values()),
        "passed": all(checks.values()),
        "claim_boundary": (
            "Paired Qwen3.8-27B Q4_K_M four-row kernel result on four Milan "
            "cores only; broad quality, trained approximation, final-node, "
            "100 accepted tok/s, and novelty gates remain separate."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "terminal", "kernel_promoted": payload["passed"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
