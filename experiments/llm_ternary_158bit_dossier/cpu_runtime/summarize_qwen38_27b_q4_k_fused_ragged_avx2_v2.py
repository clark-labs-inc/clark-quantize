#!/usr/bin/env python3
"""Summarize the paired real-model Q4_K fused-ragged AVX2 experiment."""

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


def parse_bench(paths: list[Path]) -> dict[int, dict[str, Any]]:
    samples: dict[int, list[int]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON array in {path}")
        for row in payload:
            width = int(row["n_prompt"])
            if int(row["n_gen"]) != 0:
                raise ValueError(f"{path} is not a prompt-only graph benchmark")
            samples.setdefault(width, []).extend(int(value) for value in row["samples_ns"])
    if set(samples) != {1, 2, 3}:
        raise ValueError(f"expected prompt widths 1, 2, 3; got {sorted(samples)}")
    return {
        width: {
            "samples_ns": values,
            "sample_count": len(values),
            "median_ns": statistics.median(values),
            "mean_ns": statistics.mean(values),
        }
        for width, values in samples.items()
    }


def configuration(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return next(
        row for row in summary["configurations"] if row["configuration"] == label
    )


def token_rows(root: Path, label: str) -> list[list[int]]:
    summary = json.loads((root / "summary.json").read_text())
    prompt_count = int(summary["generation"]["prompt_count"])
    return [
        json.loads((root / label / f"prompt-{index:02d}.json").read_text())["response"][
            "tokens"
        ]
        for index in range(prompt_count)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-bench", type=Path, action="append", required=True)
    parser.add_argument("--candidate-bench", type=Path, action="append", required=True)
    parser.add_argument("--control-mtp", type=Path, required=True)
    parser.add_argument("--candidate-mtp", type=Path, required=True)
    parser.add_argument("--control-server", type=Path, required=True)
    parser.add_argument("--candidate-server", type=Path, required=True)
    parser.add_argument("--base-patch", type=Path, required=True)
    parser.add_argument("--fused-patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_bench = parse_bench(args.control_bench)
    candidate_bench = parse_bench(args.candidate_bench)
    graph = {
        str(width): {
            "control": control_bench[width],
            "candidate": candidate_bench[width],
            "speedup": control_bench[width]["median_ns"]
            / candidate_bench[width]["median_ns"],
        }
        for width in (1, 2, 3)
    }

    control_summary = json.loads((args.control_mtp / "summary.json").read_text())
    candidate_summary = json.loads((args.candidate_mtp / "summary.json").read_text())
    for label, summary in (("control", control_summary), ("candidate", candidate_summary)):
        if summary.get("terminal") is not True:
            raise ValueError(f"{label} MTP summary is not terminal")
        generation = summary["generation"]
        if generation["prompt_count"] != 4 or generation["tokens_per_prompt"] != 96:
            raise ValueError(f"{label} MTP denominator is not frozen at 4 x 96")

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
    target_exact = token_rows(args.control_mtp, "target") == token_rows(
        args.candidate_mtp, "target"
    )
    mtp3_exact = token_rows(args.control_mtp, "mtp3") == token_rows(
        args.candidate_mtp, "mtp3"
    )
    parity_pattern_equal = (
        control_mtp3["prompt_token_parity"] == candidate_mtp3["prompt_token_parity"]
    )

    checks = {
        "graph_width_1_regression_at_most_0_02": graph["1"]["speedup"] >= 1 / 1.02,
        "graph_width_2_speedup_at_least_1_08": graph["2"]["speedup"] >= 1.08,
        "graph_width_3_speedup_at_least_1_10": graph["3"]["speedup"] >= 1.10,
        "target_only_regression_at_most_0_02": target_ratio >= 0.98,
        "whole_model_mtp3_speedup_at_least_1_05": mtp3_ratio >= 1.05,
        "cross_runtime_target_exact_token_parity": target_exact,
        "cross_runtime_mtp3_exact_token_parity": mtp3_exact,
        "internal_parity_pattern_unchanged": parity_pattern_equal,
    }
    payload = {
        "schema": "qwen38_27b_q4_k_fused_ragged_avx2_result_v2",
        "terminal": True,
        "infrastructure_valid": True,
        "model": "Qwen/Qwen3.8-27B",
        "host_scope": "four physical AMD EPYC Milan cores",
        "provenance": {
            "control_server_sha256": sha256_file(args.control_server),
            "candidate_server_sha256": sha256_file(args.candidate_server),
            "base_patch_sha256": sha256_file(args.base_patch),
            "fused_patch_sha256": sha256_file(args.fused_patch),
        },
        "real_model_prompt_graph": graph,
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
        },
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "Paired Qwen3.8-27B Q4_K_M result on four Milan cores only; not a "
            "final-node, capability, quality, or 100 accepted tok/s result."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "terminal", "passed": payload["passed"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
