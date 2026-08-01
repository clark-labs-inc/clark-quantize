#!/usr/bin/env python3
"""Produce a paired, development-only regression diagnostic across model states."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-paired-regression-diagnostic-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)
STRATA = ("source", "script", "language")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worst-samples", type=int, default=25)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"report has no sample evidence: {path}")
    return report


def index_samples(report: dict) -> dict[str, dict]:
    indexed = {}
    for sample in report["samples"]:
        sha256 = sample.get("sha256")
        if not sha256 or sha256 in indexed:
            raise RuntimeError("sample identities are missing or duplicated")
        indexed[sha256] = sample
    return indexed


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def delta_summary(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "win_fraction": sum(value > 0 for value in values) / len(values),
        "within_0_1_db_fraction": (
            sum(value >= -0.1 for value in values) / len(values)
        ),
    }


def paired_deltas(
    left: dict[str, dict],
    right: dict[str, dict],
    metric: str,
) -> dict[str, float]:
    return {
        sha256: float(right[sha256][metric]) - float(left[sha256][metric])
        for sha256 in left
    }


def stratified(
    sample_index: dict[str, dict],
    deltas: dict[str, float],
    key: str,
) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sha256, value in deltas.items():
        stratum = sample_index[sha256].get(key)
        if stratum is not None:
            grouped[str(stratum)].append(value)
    return {
        stratum: delta_summary(values)
        for stratum, values in sorted(grouped.items())
    }


def ranked_strata(groups: dict[str, dict], minimum_samples: int) -> list[dict]:
    eligible = [
        {"stratum": key, **summary}
        for key, summary in groups.items()
        if summary["samples"] >= minimum_samples
    ]
    return sorted(eligible, key=lambda item: (item["mean"], item["stratum"]))


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def bucketed_effect(
    ordering: dict[str, float],
    effect: dict[str, float],
    buckets: int = 10,
) -> list[dict]:
    ordered = sorted(ordering, key=lambda key: (ordering[key], key))
    bucketed = []
    for index, keys in enumerate(np.array_split(ordered, buckets)):
        members = [str(key) for key in keys]
        values = [effect[key] for key in members]
        positions = [ordering[key] for key in members]
        bucketed.append({
            "bucket": index + 1,
            "ordering_min": min(positions),
            "ordering_max": max(positions),
            "effect": delta_summary(values),
        })
    return bucketed


def comparison(
    left: dict[str, dict],
    right: dict[str, dict],
) -> dict:
    result = {"metrics": {}, "strata": {}}
    for metric in METRICS:
        deltas = paired_deltas(left, right, metric)
        result["metrics"][metric] = delta_summary(list(deltas.values()))
        result["strata"][metric] = {
            key: stratified(left, deltas, key)
            for key in STRATA
        }
    return result


def worst_samples(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    reference: dict[str, dict],
    limit: int,
) -> list[dict]:
    rows = []
    for sha256 in baseline:
        metadata = baseline[sha256]
        row = {
            "sha256": sha256,
            "source": metadata.get("source"),
            "split": metadata.get("split"),
            "language": metadata.get("language"),
            "script": metadata.get("script"),
            "candidate_minus_baseline": {},
            "baseline_minus_q4": {},
        }
        for metric in METRICS:
            row["candidate_minus_baseline"][metric] = (
                float(candidate[sha256][metric])
                - float(baseline[sha256][metric])
            )
            row["baseline_minus_q4"][metric] = (
                float(baseline[sha256][metric])
                - float(reference[sha256][metric])
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            item["candidate_minus_baseline"]["psnr_vs_ground_truth"],
            item["sha256"],
        ),
    )[:limit]


def diagnose(
    baseline_report: dict,
    candidate_report: dict,
    reference_report: dict,
    worst_limit: int,
) -> dict:
    baseline = index_samples(baseline_report)
    candidate = index_samples(candidate_report)
    reference = index_samples(reference_report)
    if baseline.keys() != candidate.keys() or baseline.keys() != reference.keys():
        raise RuntimeError("reports do not contain the exact same sample identities")

    candidate_vs_baseline = comparison(baseline, candidate)
    baseline_vs_reference = comparison(reference, baseline)
    candidate_vs_reference = comparison(reference, candidate)
    baseline_gap = paired_deltas(
        reference,
        baseline,
        "psnr_vs_ground_truth",
    )
    continuation_delta = paired_deltas(
        baseline,
        candidate,
        "psnr_vs_ground_truth",
    )
    baseline_psnr = {
        key: float(sample["psnr_vs_ground_truth"])
        for key, sample in baseline.items()
    }
    bicubic_psnr = {
        key: float(sample["bicubic_psnr_vs_ground_truth"])
        for key, sample in baseline.items()
    }
    teacher_psnr = {
        key: float(sample["teacher_psnr_vs_ground_truth"])
        for key, sample in baseline.items()
    }
    baseline_gain_over_bicubic = {
        key: baseline_psnr[key] - bicubic_psnr[key]
        for key in baseline
    }
    ordered = sorted(baseline)
    source_groups = candidate_vs_baseline["strata"][
        "psnr_vs_ground_truth"
    ]["source"]
    script_groups = candidate_vs_baseline["strata"][
        "edge_psnr_vs_ground_truth"
    ]["script"]
    language_groups = candidate_vs_baseline["strata"][
        "psnr_vs_ground_truth"
    ]["language"]
    baseline_q4_sources = baseline_vs_reference["strata"][
        "psnr_vs_ground_truth"
    ]["source"]
    baseline_q4_scripts = baseline_vs_reference["strata"][
        "edge_psnr_vs_ground_truth"
    ]["script"]
    baseline_q4_languages = baseline_vs_reference["strata"][
        "psnr_vs_ground_truth"
    ]["language"]
    return {
        "format": FORMAT,
        "selection_source": "development",
        "final_heldout_locked": True,
        "model_promotion_authorized": False,
        "sample_identity_count": len(ordered),
        "candidate_vs_baseline": candidate_vs_baseline,
        "baseline_vs_q4": baseline_vs_reference,
        "candidate_vs_q4": candidate_vs_reference,
        "diagnostics": {
            "correlation_continuation_delta_with_existing_q4_gap": correlation(
                [baseline_gap[key] for key in ordered],
                [continuation_delta[key] for key in ordered],
            ),
            "correlation_continuation_delta_with_baseline_psnr": correlation(
                [baseline_psnr[key] for key in ordered],
                [continuation_delta[key] for key in ordered],
            ),
            "correlation_continuation_delta_with_bicubic_psnr": correlation(
                [bicubic_psnr[key] for key in ordered],
                [continuation_delta[key] for key in ordered],
            ),
            "correlation_continuation_delta_with_teacher_psnr": correlation(
                [teacher_psnr[key] for key in ordered],
                [continuation_delta[key] for key in ordered],
            ),
            "correlation_continuation_delta_with_baseline_gain_over_bicubic": (
                correlation(
                    [baseline_gain_over_bicubic[key] for key in ordered],
                    [continuation_delta[key] for key in ordered],
                )
            ),
            "continuation_effect_by_existing_q4_gap_decile": bucketed_effect(
                baseline_gap,
                continuation_delta,
            ),
            "continuation_effect_by_baseline_psnr_decile": bucketed_effect(
                baseline_psnr,
                continuation_delta,
            ),
            "worst_sources_candidate_vs_baseline": ranked_strata(
                source_groups,
                minimum_samples=1,
            ),
            "worst_scripts_candidate_vs_baseline": ranked_strata(
                script_groups,
                minimum_samples=10,
            ),
            "worst_languages_candidate_vs_baseline": ranked_strata(
                language_groups,
                minimum_samples=10,
            ),
            "worst_sources_baseline_vs_q4": ranked_strata(
                baseline_q4_sources,
                minimum_samples=1,
            ),
            "worst_scripts_baseline_vs_q4": ranked_strata(
                baseline_q4_scripts,
                minimum_samples=10,
            ),
            "worst_languages_baseline_vs_q4": ranked_strata(
                baseline_q4_languages,
                minimum_samples=10,
            ),
            "worst_samples_candidate_vs_baseline": worst_samples(
                baseline,
                candidate,
                reference,
                worst_limit,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    reports = {
        "baseline": load_report(args.baseline),
        "candidate": load_report(args.candidate),
        "reference": load_report(args.reference),
    }
    result = diagnose(
        reports["baseline"],
        reports["candidate"],
        reports["reference"],
        args.worst_samples,
    )
    result["inputs"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "state_sha256": reports[name].get("state_sha256"),
        }
        for name, path in (
            ("baseline", args.baseline),
            ("candidate", args.candidate),
            ("reference", args.reference),
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "candidate_minus_baseline_db": result["candidate_vs_baseline"][
            "metrics"
        ]["psnr_vs_ground_truth"]["mean"],
        "candidate_minus_q4_db": result["candidate_vs_q4"]["metrics"][
            "psnr_vs_ground_truth"
        ]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
