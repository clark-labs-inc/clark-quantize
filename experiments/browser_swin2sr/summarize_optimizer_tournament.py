#!/usr/bin/env python3
"""Validate and summarize a completed optimizer tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--promote-count", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def arm_summary(run: Path, result: dict) -> dict:
    validation = result.get("validation")
    if result.get("return_code") != 0 or not validation:
        return {
            "name": result["name"],
            "optimizer": result["optimizer"],
            "learning_rate": result["learning_rate"],
            "eligible": False,
            "reason": f"return_code={result.get('return_code')} validation={bool(validation)}",
        }
    metric_names = ("psnr_vs_ground_truth", "psnr_vs_teacher", "loss")
    metrics = {name: float(validation[name]) for name in metric_names}
    if not all(math.isfinite(value) for value in metrics.values()):
        return {
            "name": result["name"],
            "optimizer": result["optimizer"],
            "learning_rate": result["learning_rate"],
            "eligible": False,
            "reason": "non-finite metric",
            **metrics,
        }
    checkpoint = run / result["name"] / "checkpoint-final.pt"
    payload = torch.load(checkpoint, map_location="cpu")
    optimizer_state_bytes = tensor_bytes(payload["optimizer"])
    return {
        "name": result["name"],
        "optimizer": result["optimizer"],
        "learning_rate": result["learning_rate"],
        "learning_rate_multiplier": result["learning_rate_multiplier"],
        "eligible": True,
        **metrics,
        "training_seconds": float(
            json.loads((run / result["name"] / "summary.json").read_text())[
                "elapsed_seconds"
            ]
        ),
        "optimizer_state_bytes": optimizer_state_bytes,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
    }


def summarize(run: Path, promote_count: int) -> dict:
    manifest_path = run / "manifest.json"
    results_path = run / "results.json"
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    expected_names = [arm["name"] for arm in manifest["arms"]]
    result_names = [result["name"] for result in results["results"]]
    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("manifest contains duplicate arm names")
    if set(expected_names) != set(result_names):
        raise RuntimeError("manifest/result arm mismatch")
    arms = [arm_summary(run, result) for result in results["results"]]
    eligible = sorted(
        (arm for arm in arms if arm["eligible"]),
        key=lambda arm: (
            -arm["psnr_vs_ground_truth"],
            -arm["psnr_vs_teacher"],
            arm["optimizer_state_bytes"],
        ),
    )
    best_by_optimizer = {}
    for arm in eligible:
        best_by_optimizer.setdefault(arm["optimizer"], arm)
    optimizer_ranking = sorted(
        best_by_optimizer.values(),
        key=lambda arm: (
            -arm["psnr_vs_ground_truth"],
            -arm["psnr_vs_teacher"],
            arm["optimizer_state_bytes"],
        ),
    )
    promoted = optimizer_ranking[: max(0, promote_count)]
    return {
        "format": "backdropkit-swin2sr-optimizer-tournament-summary-v1",
        "diagnostic_only": True,
        "promotion_is_not_final_acceptance": True,
        "stage": results["stage"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "results": str(results_path),
        "results_sha256": sha256(results_path),
        "arms_expected": len(expected_names),
        "arms_eligible": len(eligible),
        "arms_rejected": len(arms) - len(eligible),
        "ranking": eligible,
        "best_by_optimizer": optimizer_ranking,
        "promoted": promoted,
    }


def main() -> None:
    args = parse_args()
    receipt = summarize(args.run, args.promote_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "arms_eligible": receipt["arms_eligible"],
        "promoted": [arm["name"] for arm in receipt["promoted"]],
    }, indent=2))


if __name__ == "__main__":
    main()
