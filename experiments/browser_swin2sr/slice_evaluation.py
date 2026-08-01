#!/usr/bin/env python3
"""Create a reproducible prefix slice of a full Swin2SR evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_stratified import grouped_summaries, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def slice_report(report: dict, samples: int) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    if samples > len(report["samples"]):
        raise ValueError(
            f"requested {samples} samples from report containing {len(report['samples'])}"
        )
    selected = report["samples"][:samples]
    result = {
        key: value
        for key, value in report.items()
        if key not in {
            "overall",
            "by_source",
            "by_split",
            "by_script",
            "by_language",
            "samples",
        }
    }
    result.update({
        "slice": {
            "kind": "prefix",
            "samples": samples,
            "source_samples": len(report["samples"]),
        },
        "overall": summarize(selected),
        "by_source": grouped_summaries(selected, "source"),
        "by_split": grouped_summaries(selected, "split"),
        "by_script": grouped_summaries(selected, "script"),
        "by_language": grouped_summaries(selected, "language"),
        "samples": selected,
    })
    return result


def main() -> None:
    args = parse_args()
    result = slice_report(json.loads(args.report.read_text()), args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
