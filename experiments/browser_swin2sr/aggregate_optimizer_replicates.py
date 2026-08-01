#!/usr/bin/env python3
"""Aggregate matched-seed optimizer replications with paired uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", default="psnr_vs_ground_truth")
    parser.add_argument("--noninferiority-margin-db", type=float, default=0.10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_interval(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.mean(values)
    if count < 2:
        return {
            "count": count,
            "mean": mean,
            "sample_stddev": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    standard_deviation = statistics.stdev(values)
    critical = T_975.get(count - 1, 1.96)
    radius = critical * standard_deviation / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "sample_stddev": standard_deviation,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def comparable_protocol(protocol: dict) -> dict:
    return {key: value for key, value in protocol.items() if key != "seed"}


def load_runs(paths: list[Path], metric: str) -> tuple[list[dict], dict]:
    loaded = []
    reference_protocol = None
    reference_sources = None
    reference_arms = None
    for path in paths:
        manifest_path = path / "manifest.json"
        results_path = path / "results.json"
        manifest = json.loads(manifest_path.read_text())
        results = json.loads(results_path.read_text())
        protocol = comparable_protocol(manifest["protocol"])
        source_hashes = manifest["source_sha256"]
        arm_names = sorted(arm["name"] for arm in manifest["arms"])
        if reference_protocol is None:
            reference_protocol = protocol
            reference_sources = source_hashes
            reference_arms = arm_names
        if protocol != reference_protocol:
            raise RuntimeError(f"protocol mismatch in {path}")
        if source_hashes != reference_sources:
            raise RuntimeError(f"source hash mismatch in {path}")
        if arm_names != reference_arms:
            raise RuntimeError(f"arm mismatch in {path}")
        by_name = {result["name"]: result for result in results["results"]}
        if sorted(by_name) != arm_names:
            raise RuntimeError(f"manifest/result mismatch in {path}")
        for result in by_name.values():
            validation = result.get("validation")
            if result.get("return_code") != 0 or not validation:
                raise RuntimeError(f"ineligible arm {result['name']} in {path}")
            value = float(validation[metric])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {metric} for {result['name']} in {path}")
        loaded.append({
            "path": str(path),
            "seed": int(manifest["protocol"]["seed"]),
            "manifest_sha256": sha256(manifest_path),
            "results_sha256": sha256(results_path),
            "results": by_name,
        })
    seeds = [run["seed"] for run in loaded]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"duplicate seeds: {seeds}")
    return loaded, reference_protocol or {}


def aggregate(
    paths: list[Path],
    metric: str,
    noninferiority_margin: float,
) -> dict:
    runs, protocol = load_runs(paths, metric)
    values_by_arm: dict[str, list[tuple[int, float]]] = defaultdict(list)
    metadata = {}
    for run in runs:
        for name, result in run["results"].items():
            value = float(result["validation"][metric])
            values_by_arm[name].append((run["seed"], value))
            metadata[name] = {
                "optimizer": result["optimizer"],
                "learning_rate": result["learning_rate"],
                "learning_rate_multiplier": result["learning_rate_multiplier"],
            }
    summaries = []
    for name, seeded_values in values_by_arm.items():
        seeded_values.sort()
        summaries.append({
            "name": name,
            **metadata[name],
            **confidence_interval([value for _, value in seeded_values]),
            "seed_values": [
                {"seed": seed, metric: value}
                for seed, value in seeded_values
            ],
        })
    summaries.sort(key=lambda item: -float(item["mean"]))
    best = summaries[0]
    best_values = {
        item["seed"]: float(item[metric])
        for item in best["seed_values"]
    }
    paired = []
    for candidate in summaries[1:]:
        candidate_values = {
            item["seed"]: float(item[metric])
            for item in candidate["seed_values"]
        }
        deltas = [
            best_values[seed] - candidate_values[seed]
            for seed in sorted(best_values)
        ]
        interval = confidence_interval(deltas)
        paired.append({
            "best": best["name"],
            "candidate": candidate["name"],
            "delta_definition": "best_minus_candidate",
            **interval,
            "best_statistically_superior": interval["ci95_low"] > 0,
            "candidate_noninferior_within_margin": (
                interval["ci95_high"] <= noninferiority_margin
            ),
            "paired_deltas": deltas,
        })
    return {
        "format": "backdropkit-swin2sr-optimizer-replicates-v1",
        "diagnostic_only": True,
        "metric": metric,
        "noninferiority_margin_db": noninferiority_margin,
        "protocol_without_seed": protocol,
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key != "results"
            }
            for run in runs
        ],
        "ranking": summaries,
        "paired_against_best": paired,
    }


def main() -> None:
    args = parse_args()
    receipt = aggregate(
        args.runs,
        args.metric,
        args.noninferiority_margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "ranking": [
            {"name": item["name"], "mean": item["mean"]}
            for item in receipt["ranking"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
