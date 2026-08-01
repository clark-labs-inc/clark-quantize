#!/usr/bin/env python3
"""Paired comparison of two stratified Swin2SR evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
    "psnr_gain_over_bicubic",
    "student_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def indexed_samples(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads(path.read_text())
    samples = {
        sample["sha256"]: sample
        for sample in report["samples"]
    }
    if None in samples:
        raise ValueError(f"{path} contains a sample without sha256")
    if len(samples) != len(report["samples"]):
        raise ValueError(f"{path} contains duplicate sample hashes")
    return report, samples


def sample_value(sample: dict, metric: str) -> float:
    if metric == "psnr_gain_over_bicubic":
        return sample["psnr_vs_ground_truth"] - sample["bicubic_psnr_vs_ground_truth"]
    return sample[metric]


def paired_summary(
    reference: dict[str, dict],
    candidate: dict[str, dict],
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    shared = sorted(reference.keys() & candidate.keys())
    if not shared:
        raise ValueError("evaluation reports have no shared samples")
    deltas = np.asarray([
        sample_value(candidate[key], metric) - sample_value(reference[key], metric)
        for key in shared
    ])
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    direction = -1 if metric == "student_seconds" else 1
    return {
        "samples": len(shared),
        "candidate_minus_reference_mean": float(deltas.mean()),
        "candidate_minus_reference_median": float(np.median(deltas)),
        "paired_bootstrap_95ci": [float(lower), float(upper)],
        "candidate_win_fraction": float(np.mean(direction * deltas > 0)),
    }


def main() -> None:
    args = parse_args()
    reference_report, reference = indexed_samples(args.reference)
    candidate_report, candidate = indexed_samples(args.candidate)
    if reference.keys() != candidate.keys():
        missing_candidate = sorted(reference.keys() - candidate.keys())
        missing_reference = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "reports must contain identical samples; "
            f"missing candidate={len(missing_candidate)}, "
            f"missing reference={len(missing_reference)}"
        )
    rng = np.random.default_rng(args.seed)
    comparison = {
        "reference": {
            "path": str(args.reference),
            "state": reference_report["state"],
            "state_sha256": reference_report["state_sha256"],
        },
        "candidate": {
            "path": str(args.candidate),
            "state": candidate_report["state"],
            "state_sha256": candidate_report["state_sha256"],
        },
        "metrics": {
            metric: paired_summary(
                reference,
                candidate,
                metric,
                args.bootstrap_samples,
                rng,
            )
            for metric in METRICS
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison["metrics"], indent=2))


if __name__ == "__main__":
    main()
