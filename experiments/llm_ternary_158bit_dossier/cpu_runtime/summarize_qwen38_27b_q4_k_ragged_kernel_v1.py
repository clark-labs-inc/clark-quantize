#!/usr/bin/env python3
"""Summarize the paired Qwen3.8-27B ragged Q4_K kernel experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any


PERF_ROW = re.compile(
    r"MUL_MAT\(type_a=q4_K,type_b=f32,m=4096,n=(\d+),k=14336,.*?"
    r"(\d+) runs -\s+([0-9.]+) us/run"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_perf(paths: list[Path]) -> dict[int, dict[str, Any]]:
    samples: dict[int, list[float]] = {}
    adaptive_runs: dict[int, list[int]] = {}
    for path in paths:
        text = path.read_text()
        matches = PERF_ROW.findall(text)
        if not matches:
            raise ValueError(f"no Q4_K performance rows in {path}")
        for width, runs, usec in matches:
            samples.setdefault(int(width), []).append(float(usec))
            adaptive_runs.setdefault(int(width), []).append(int(runs))
    required = {1, 2, 3, 4, 5, 8}
    if set(samples) != required:
        raise ValueError(f"expected widths {sorted(required)}, got {sorted(samples)}")
    return {
        width: {
            "invocation_microseconds": values,
            "median_microseconds": statistics.median(values),
            "adaptive_runs_per_invocation": adaptive_runs[width],
        }
        for width, values in samples.items()
    }


def load_mtp(root: Path) -> tuple[dict[str, Any], list[list[int]]]:
    summary = json.loads((root / "summary.json").read_text())
    tokens = [
        json.loads((root / "target" / f"prompt-{index:02d}.json").read_text())[
            "response"
        ]["tokens"]
        for index in range(4)
    ]
    return summary, tokens


def configuration(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return next(
        row for row in summary["configurations"] if row["configuration"] == label
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-perf", type=Path, action="append", required=True)
    parser.add_argument("--candidate-perf", type=Path, action="append", required=True)
    parser.add_argument("--control-mtp", type=Path, required=True)
    parser.add_argument("--candidate-mtp", type=Path, required=True)
    parser.add_argument("--control-server", type=Path, required=True)
    parser.add_argument("--candidate-server", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_perf = parse_perf(args.control_perf)
    candidate_perf = parse_perf(args.candidate_perf)
    micro = {}
    for width in sorted(control_perf):
        control_us = control_perf[width]["median_microseconds"]
        candidate_us = candidate_perf[width]["median_microseconds"]
        micro[str(width)] = {
            "control": control_perf[width],
            "candidate": candidate_perf[width],
            "speedup": control_us / candidate_us,
        }

    control_mtp, control_tokens = load_mtp(args.control_mtp)
    candidate_mtp, candidate_tokens = load_mtp(args.candidate_mtp)
    control_target = configuration(control_mtp, "target")
    control_width3 = configuration(control_mtp, "mtp3")
    candidate_target = configuration(candidate_mtp, "target")
    candidate_width3 = configuration(candidate_mtp, "mtp3")
    target_ratio = (
        candidate_target["accepted_tokens_per_second_weighted"]
        / control_target["accepted_tokens_per_second_weighted"]
    )
    width3_ratio = (
        candidate_width3["accepted_tokens_per_second_weighted"]
        / control_width3["accepted_tokens_per_second_weighted"]
    )
    cross_runtime_target_parity = control_tokens == candidate_tokens

    checks = {
        "micro_width_2_speedup_at_least_1_25": micro["2"]["speedup"] >= 1.25,
        "micro_width_3_speedup_at_least_1_50": micro["3"]["speedup"] >= 1.50,
        "whole_model_width_3_speedup_at_least_1_15": width3_ratio >= 1.15,
        "target_only_regression_at_most_0_02": target_ratio >= 0.98,
        "control_internal_exact_parity": control_width3["exact_token_parity"],
        "candidate_internal_exact_parity": candidate_width3["exact_token_parity"],
        "cross_runtime_target_exact_parity": cross_runtime_target_parity,
    }
    payload = {
        "schema": "qwen38_27b_q4_k_ragged_rows_kernel_result_v1",
        "terminal": True,
        "control_server_sha256": sha256_file(args.control_server),
        "candidate_server_sha256": sha256_file(args.candidate_server),
        "microbenchmark": micro,
        "whole_model": {
            "control_target_accepted_tokens_per_second": control_target[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_target_accepted_tokens_per_second": candidate_target[
                "accepted_tokens_per_second_weighted"
            ],
            "target_only_candidate_over_control": target_ratio,
            "control_mtp3_accepted_tokens_per_second": control_width3[
                "accepted_tokens_per_second_weighted"
            ],
            "candidate_mtp3_accepted_tokens_per_second": candidate_width3[
                "accepted_tokens_per_second_weighted"
            ],
            "mtp3_candidate_over_control": width3_ratio,
            "cross_runtime_target_exact_token_parity": cross_runtime_target_parity,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": "Paired four-core Milan kernel result only; not final-node or 100 accepted tok/s evidence.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "terminal", "passed": payload["passed"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
