#!/usr/bin/env python3
"""Select a development optimizer from paired Swin2SR replications.

The primary gate is paired ground-truth PSNR from
``aggregate_optimizer_replicates.py``. An arm may trade quality for lower
training state only when it is non-inferior to the empirical best on both
ground-truth and teacher PSNR. This decision is development-only and cannot
promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aggregate_optimizer_replicates import confidence_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-margin-db", type=float, default=0.10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_summary(path: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads(path.read_text())
    if summary.get("format") != (
        "backdropkit-swin2sr-optimizer-tournament-summary-v1"
    ):
        raise RuntimeError(f"unsupported summary format: {path}")
    if summary.get("stage") != "optimizer-replicate":
        raise RuntimeError(f"summary is not an optimizer replicate: {path}")
    if summary.get("arms_rejected") or not summary.get("ranking"):
        raise RuntimeError(f"summary has rejected or missing arms: {path}")
    indexed = {arm["name"]: arm for arm in summary["ranking"]}
    if len(indexed) != len(summary["ranking"]):
        raise RuntimeError(f"summary contains duplicate arms: {path}")
    return summary, indexed


def paired_teacher_gate(
    best: str,
    candidate: str,
    arms_by_seed: dict[int, dict[str, dict]],
    margin: float,
) -> dict:
    deltas = [
        float(arms[best]["psnr_vs_teacher"])
        - float(arms[candidate]["psnr_vs_teacher"])
        for _, arms in sorted(arms_by_seed.items())
    ]
    interval = confidence_interval(deltas)
    return {
        "best": best,
        "candidate": candidate,
        "delta_definition": "best_minus_candidate",
        **interval,
        "noninferiority_margin_db": margin,
        "candidate_noninferior_within_margin": interval["ci95_high"] <= margin,
        "paired_deltas": deltas,
    }


def decide(
    aggregate_path: Path,
    summary_paths: list[Path],
    teacher_margin: float,
) -> dict:
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("format") != (
        "backdropkit-swin2sr-optimizer-replicates-v1"
    ):
        raise RuntimeError("unsupported aggregate format")
    if not aggregate.get("diagnostic_only"):
        raise RuntimeError("optimizer aggregate must be development-only")
    ranking = aggregate.get("ranking") or []
    if not ranking:
        raise RuntimeError("optimizer aggregate has no ranking")
    best = ranking[0]["name"]
    ground_truth_gates = {
        comparison["candidate"]: comparison
        for comparison in aggregate.get("paired_against_best", [])
    }
    expected_runs = {
        (
            run["manifest_sha256"],
            run["results_sha256"],
        ): int(run["seed"])
        for run in aggregate["runs"]
    }
    if len(expected_runs) != len(aggregate["runs"]):
        raise RuntimeError("aggregate has duplicate run identities")

    summaries = []
    arms_by_seed: dict[int, dict[str, dict]] = {}
    expected_arms = {item["name"] for item in ranking}
    for path in summary_paths:
        summary, arms = indexed_summary(path)
        identity = (
            summary["manifest_sha256"],
            summary["results_sha256"],
        )
        if identity not in expected_runs:
            raise RuntimeError(f"summary is not part of aggregate: {path}")
        seed = expected_runs[identity]
        if seed in arms_by_seed:
            raise RuntimeError(f"duplicate summary seed: {seed}")
        if set(arms) != expected_arms:
            raise RuntimeError(f"summary arm mismatch: {path}")
        arms_by_seed[seed] = arms
        summaries.append({
            "path": str(path),
            "sha256": sha256(path),
            "seed": seed,
        })
    if len(arms_by_seed) != len(expected_runs):
        raise RuntimeError(
            f"expected {len(expected_runs)} summaries, got {len(arms_by_seed)}"
        )

    teacher_gates = {
        candidate: paired_teacher_gate(
            best,
            candidate,
            arms_by_seed,
            teacher_margin,
        )
        for candidate in sorted(expected_arms - {best})
    }
    candidates = []
    mean_by_name = {item["name"]: float(item["mean"]) for item in ranking}
    for name in sorted(expected_arms):
        gt_noninferior = (
            name == best
            or ground_truth_gates[name]["candidate_noninferior_within_margin"]
        )
        teacher_noninferior = (
            name == best
            or teacher_gates[name]["candidate_noninferior_within_margin"]
        )
        state_values = [
            int(arms[name]["optimizer_state_bytes"])
            for arms in arms_by_seed.values()
        ]
        teacher_values = [
            float(arms[name]["psnr_vs_teacher"])
            for arms in arms_by_seed.values()
        ]
        candidates.append({
            "name": name,
            "optimizer": next(iter(arms_by_seed.values()))[name]["optimizer"],
            "ground_truth_mean_db": mean_by_name[name],
            "teacher_mean_db": sum(teacher_values) / len(teacher_values),
            "optimizer_state_bytes_by_seed": state_values,
            "optimizer_state_bytes_max": max(state_values),
            "ground_truth_noninferior": gt_noninferior,
            "teacher_noninferior": teacher_noninferior,
            "eligible": gt_noninferior and teacher_noninferior,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("empirical best was unexpectedly ineligible")
    eligible.sort(key=lambda candidate: (
        candidate["optimizer_state_bytes_max"],
        -candidate["ground_truth_mean_db"],
        -candidate["teacher_mean_db"],
        candidate["name"],
    ))
    selected = eligible[0]
    return {
        "format": "backdropkit-swin2sr-optimizer-decision-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "promotion_is_not_final_acceptance": True,
        "rule": (
            "Among arms paired-noninferior to the empirical GT-PSNR best "
            "and teacher-PSNR best, minimize maximum optimizer-state bytes; "
            "then maximize GT PSNR, teacher PSNR, and use name as a final "
            "deterministic tie-break."
        ),
        "aggregate": str(aggregate_path),
        "aggregate_sha256": sha256(aggregate_path),
        "ground_truth_margin_db": aggregate["noninferiority_margin_db"],
        "teacher_margin_db": teacher_margin,
        "summaries": sorted(summaries, key=lambda item: item["seed"]),
        "empirical_quality_best": best,
        "ground_truth_gates": ground_truth_gates,
        "teacher_gates": teacher_gates,
        "candidates": candidates,
        "eligible_low_state_set": [item["name"] for item in eligible],
        "selected": selected,
    }


def main() -> None:
    args = parse_args()
    receipt = decide(
        args.aggregate,
        args.summaries,
        args.teacher_margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "empirical_quality_best": receipt["empirical_quality_best"],
        "selected": receipt["selected"]["name"],
        "eligible_low_state_set": receipt["eligible_low_state_set"],
    }, indent=2))


if __name__ == "__main__":
    main()
