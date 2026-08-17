#!/usr/bin/env python3
"""Validate and summarize a paired Qwen3.8-27B llama-bench matrix."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--llama-bench-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--claim-boundary", required=True)
    parser.add_argument("--minimum-every-row", type=float, default=1.0)
    parser.add_argument("--minimum-geometric-mean", type=float, default=1.05)
    args = parser.parse_args()

    models = {
        "baseline": {"path": str(args.baseline.resolve()), "sha256": args.baseline_sha256},
        "candidate": {"path": str(args.candidate.resolve()), "sha256": args.candidate_sha256},
    }
    rows: dict[str, dict[tuple, dict]] = {"baseline": {}, "candidate": {}}
    serialized = []
    for label in ("baseline", "candidate"):
        for threads in (1, 2, 4):
            for mode in ("prefill", "decode"):
                path = args.root / f"{label}-{mode}-t{threads}.json"
                payload = json.loads(path.read_text())
                expected_rows = 3 if mode == "prefill" else 1
                if not isinstance(payload, list) or len(payload) != expected_rows:
                    raise ValueError(f"unexpected row count in {path}")
                for raw in payload:
                    if raw["model_filename"] != models[label]["path"]:
                        raise ValueError(f"model mismatch in {path}")
                    if int(raw["n_threads"]) != threads or raw["cpu_strict"] is not True:
                        raise ValueError(f"CPU contract mismatch in {path}")
                    samples = [float(value) for value in raw["samples_ts"]]
                    if len(samples) != 5 or not all(math.isfinite(v) and v > 0 for v in samples):
                        raise ValueError(f"invalid timing samples in {path}")
                    prompt, generated = int(raw["n_prompt"]), int(raw["n_gen"])
                    if mode == "prefill" and not (prompt in {128, 512, 2048} and generated == 0):
                        raise ValueError(f"prefill shape mismatch in {path}")
                    if mode == "decode" and not (prompt == 0 and generated == 128):
                        raise ValueError(f"decode shape mismatch in {path}")
                    key = (mode, threads, prompt, generated)
                    row = {
                        "model": label,
                        "mode": mode,
                        "threads": threads,
                        "prompt_tokens": prompt,
                        "generated_tokens": generated,
                        "mean_tokens_per_second": float(raw["avg_ts"]),
                        "stddev_tokens_per_second": float(raw["stddev_ts"]),
                        "median_tokens_per_second": statistics.median(samples),
                        "minimum_tokens_per_second": min(samples),
                        "maximum_tokens_per_second": max(samples),
                        "samples_tokens_per_second": samples,
                    }
                    if key in rows[label]:
                        raise ValueError(f"duplicate paired row key in {path}: {key}")
                    rows[label][key] = row
                    serialized.append(row)
    if len(rows["baseline"]) != 12 or set(rows["baseline"]) != set(rows["candidate"]):
        raise ValueError("paired row keys are incomplete or differ")

    comparisons = []
    for key in sorted(rows["baseline"]):
        baseline_row, candidate_row = rows["baseline"][key], rows["candidate"][key]
        speedup = candidate_row["median_tokens_per_second"] / baseline_row["median_tokens_per_second"]
        comparisons.append(
            {
                "mode": key[0],
                "threads": key[1],
                "prompt_tokens": key[2],
                "generated_tokens": key[3],
                "baseline_median_tokens_per_second": baseline_row["median_tokens_per_second"],
                "candidate_median_tokens_per_second": candidate_row["median_tokens_per_second"],
                "candidate_over_baseline_speedup": speedup,
            }
        )
    speedups = [row["candidate_over_baseline_speedup"] for row in comparisons]
    geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    checks = {
        "row_count": len(comparisons) == 12,
        "no_row_below_frozen_minimum": min(speedups) >= args.minimum_every_row,
        "geometric_mean_meets_frozen_minimum": geometric_mean >= args.minimum_geometric_mean,
    }
    summary = {
        "schema": args.schema,
        "models": models,
        "llama_bench": {
            "path": str(args.llama_bench.resolve()),
            "sha256": args.llama_bench_sha256,
        },
        "rows": serialized,
        "comparisons": comparisons,
        "minimum_speedup": min(speedups),
        "geometric_mean_speedup": geometric_mean,
        "thresholds": {
            "minimum_every_row": args.minimum_every_row,
            "minimum_geometric_mean": args.minimum_geometric_mean,
        },
        "checks": checks,
        "valid": True,
        "speed_promotion_pass": all(checks.values()),
        "claim_boundary": args.claim_boundary,
    }
    temporary = args.root / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.root / "summary.json")


if __name__ == "__main__":
    main()
