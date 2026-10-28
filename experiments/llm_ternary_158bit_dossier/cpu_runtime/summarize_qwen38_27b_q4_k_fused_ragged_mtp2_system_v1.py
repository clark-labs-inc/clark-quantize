#!/usr/bin/env python3
"""Summarize the shape-matched Qwen3.8-27B MTP2 fused-kernel A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-mtp2", type=Path, required=True)
    parser.add_argument("--candidate-mtp2", type=Path, required=True)
    parser.add_argument("--control-mtp3", type=Path, required=True)
    parser.add_argument("--control-server", type=Path, required=True)
    parser.add_argument("--candidate-server", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_summary = load_summary(args.control_mtp2)
    candidate_summary = load_summary(args.candidate_mtp2)
    control_mtp3_summary = load_summary(args.control_mtp3)

    control_target = configuration(control_summary, "target")
    candidate_target = configuration(candidate_summary, "target")
    control_mtp2 = configuration(control_summary, "mtp2")
    candidate_mtp2 = configuration(candidate_summary, "mtp2")
    control_mtp3 = configuration(control_mtp3_summary, "mtp3")

    target_ratio = (
        candidate_target["accepted_tokens_per_second_weighted"]
        / control_target["accepted_tokens_per_second_weighted"]
    )
    paired_mtp2_ratio = (
        candidate_mtp2["accepted_tokens_per_second_weighted"]
        / control_mtp2["accepted_tokens_per_second_weighted"]
    )
    candidate_mtp2_over_control_mtp3 = (
        candidate_mtp2["accepted_tokens_per_second_weighted"]
        / control_mtp3["accepted_tokens_per_second_weighted"]
    )
    target_exact = token_rows(args.control_mtp2, "target") == token_rows(
        args.candidate_mtp2, "target"
    )
    mtp2_exact = token_rows(args.control_mtp2, "mtp2") == token_rows(
        args.candidate_mtp2, "mtp2"
    )
    parity_pattern_equal = (
        control_mtp2["prompt_token_parity"] == candidate_mtp2["prompt_token_parity"]
    )

    kernel_checks = {
        "paired_mtp2_speedup_at_least_1_05": paired_mtp2_ratio >= 1.05,
        "target_only_regression_at_most_0_02": target_ratio >= 0.98,
        "cross_runtime_target_exact_token_parity": target_exact,
        "cross_runtime_mtp2_exact_token_parity": mtp2_exact,
        "internal_parity_pattern_unchanged": parity_pattern_equal,
    }
    deployment_winner = candidate_mtp2_over_control_mtp3 > 1.0
    payload = {
        "schema": "qwen38_27b_q4_k_fused_ragged_mtp2_system_result_v1",
        "terminal": True,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "shape": {
            "mtp_draft_width": 2,
            "target_verification_rows": 3,
        },
        "provenance": {
            "control_server_sha256": sha256_file(args.control_server),
            "candidate_server_sha256": sha256_file(args.candidate_server),
        },
        "throughput": {
            "control_target_accepted_tokens_per_second": control_target[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_target_accepted_tokens_per_second": candidate_target[
                "accepted_tokens_per_second_weighted"
            ],
            "target_candidate_over_control": target_ratio,
            "control_mtp2_accepted_tokens_per_second": control_mtp2[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_mtp2_accepted_tokens_per_second": candidate_mtp2[
                "accepted_tokens_per_second_weighted"
            ],
            "paired_mtp2_candidate_over_control": paired_mtp2_ratio,
            "control_mtp3_accepted_tokens_per_second": control_mtp3[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_mtp2_over_control_mtp3": candidate_mtp2_over_control_mtp3,
        },
        "correctness": {
            "cross_runtime_target_exact_token_parity": target_exact,
            "cross_runtime_mtp2_exact_token_parity": mtp2_exact,
            "control_internal_target_vs_mtp2_prompt_parity": control_mtp2[
                "prompt_token_parity"
            ],
            "candidate_internal_target_vs_mtp2_prompt_parity": candidate_mtp2[
                "prompt_token_parity"
            ],
        },
        "kernel_checks": kernel_checks,
        "kernel_promoted": all(kernel_checks.values()),
        "deployment_winner_check": {
            "candidate_mtp2_exceeds_control_mtp3": deployment_winner,
        },
        "deployment_winner": deployment_winner and all(kernel_checks.values()),
        "passed": all(kernel_checks.values()),
        "claim_boundary": (
            "Paired four-core Milan MTP2 result only. Kernel promotion and the "
            "serving-configuration winner are intentionally separate decisions."
        ),
    }
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "event": "terminal",
                "kernel_promoted": payload["kernel_promoted"],
                "deployment_winner": payload["deployment_winner"],
            }
        )
    )
    return 0 if payload["kernel_promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
