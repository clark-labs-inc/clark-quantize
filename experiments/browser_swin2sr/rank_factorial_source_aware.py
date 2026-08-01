#!/usr/bin/env python3
"""Rerank a completed Swin2SR race by its weakest held-out source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_recovery_factorial import paired_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def rank_key(row: dict) -> tuple[float, float, float, float]:
    paired = row["paired_q4"]
    means = paired["mean_deltas"]
    return (
        paired["worst_source_psnr_delta"],
        means["psnr_vs_ground_truth"],
        means["edge_psnr_vs_ground_truth"],
        means["psnr_vs_teacher"],
    )


def rerank(stage: Path, q4_reference: dict) -> dict:
    rows = []
    for result_path in sorted(stage.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        heldout = result.get("heldout") or result.get("metrics")
        if result.get("status") not in {"complete", "already_complete"} or not heldout:
            continue
        rows.append({
            "name": result["name"],
            "result": str(result_path),
            "paired_q4": paired_scores(heldout, q4_reference),
        })
    ranked = sorted(rows, key=rank_key, reverse=True)
    return {
        "format": "backdropkit-swin2sr-source-aware-ranking-v1",
        "stage": str(stage),
        "completed": len(ranked),
        "ranking": [
            {"rank": index + 1, **row}
            for index, row in enumerate(ranked)
        ],
    }


def main() -> None:
    args = parse_args()
    report = rerank(
        args.stage,
        json.loads(args.q4_reference.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
