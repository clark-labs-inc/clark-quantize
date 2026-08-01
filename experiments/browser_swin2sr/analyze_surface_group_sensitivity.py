#!/usr/bin/env python3
"""Screen one-at-a-time scale-group refinement across ternary surfaces."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from analyze_surface_restore_sensitivity import (
    METRICS,
    load_sample,
    metric_delta,
    state_metrics,
    stratified_records,
)
from evaluate_hard_started_checkpoints import load_checkpoint_student
from evaluate_stratified import (
    file_sha256,
    load_bundle_records,
    product_pixels,
    reconstruct,
)
from probe_sensitive_io_group_sizes import storage_receipt
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-surface-group-sensitivity-v1"
METHODS = ("current", "optimal_mse", "absmean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-group-size", type=int, default=2)
    parser.add_argument("--method", choices=METHODS, default="current")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--surface", action="append")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def paired_metric_evidence(
    candidate_rows: list[dict[str, float]],
    baseline_rows: list[dict[str, float]],
    bootstrap_indices: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate and baseline rows must be paired")
    if not candidate_rows:
        raise ValueError("paired evidence requires at least one sample")
    evidence = {}
    for metric in METRICS:
        deltas = np.asarray([
            candidate[metric] - baseline[metric]
            for candidate, baseline in zip(candidate_rows, baseline_rows)
        ], dtype=np.float64)
        bootstrap_means = deltas[bootstrap_indices].mean(axis=1)
        evidence[metric] = {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "lower_95": float(np.quantile(bootstrap_means, 0.025)),
            "upper_95": float(np.quantile(bootstrap_means, 0.975)),
            "wins": int(np.count_nonzero(deltas > 0)),
            "ties": int(np.count_nonzero(deltas == 0)),
            "samples": int(deltas.size),
        }
    return evidence


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"group-sensitivity output exists: {args.out}")
    if (
        args.target_group_size < 1
        or args.torch_threads < 1
        or args.bootstrap_resamples < 1
    ):
        raise ValueError("group size, threads, and resamples must be positive")
    torch.set_num_threads(args.torch_threads)
    bundle, records = load_bundle_records(args.bundle)
    selected_records = stratified_records(records, args.max_samples)
    samples = [load_sample(record, args.device) for record in selected_records]
    bootstrap_indices = np.random.default_rng(args.bootstrap_seed).integers(
        0,
        len(samples),
        size=(args.bootstrap_resamples, len(samples)),
    )
    student, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        args.device,
    )
    if len(surfaces) != 204:
        raise RuntimeError("checkpoint does not cover all 204 surfaces")
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    available = {surface.name: surface for surface in surfaces}
    requested = args.surface or sorted(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown surfaces requested: {unknown}")
    total_values = sum(surface.module.weight.numel() for surface in surfaces)
    teacher = pretrained(args.device).eval()
    with torch.inference_mode():
        teacher_outputs = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in samples
        ]
        baseline = state_metrics(
            student,
            samples,
            teacher_outputs,
            args.device,
        )
        started = time.perf_counter()
        results = []
        for index, name in enumerate(requested, start=1):
            surface = available[name]
            original_group_size = surface.quantizer.group_size
            original_method = surface.quantizer.method
            candidate_method = (
                original_method if args.method == "current" else args.method
            )
            surface.quantizer.group_size = args.target_group_size
            surface.quantizer.method = candidate_method
            candidate = state_metrics(
                student,
                samples,
                teacher_outputs,
                args.device,
            )
            surface.quantizer.group_size = original_group_size
            surface.quantizer.method = original_method
            storage = storage_receipt(
                surface.module.weight.numel(),
                args.target_group_size,
                total_values,
                original_group_size,
            )
            delta = metric_delta(
                candidate["summary"],
                baseline["summary"],
            )
            scale_bytes_delta = (
                storage["scale_count"]
                - (surface.module.weight.numel() + original_group_size - 1)
                // original_group_size
            ) * 2
            results.append({
                "name": name,
                "category": surface.category,
                "values": surface.module.weight.numel(),
                "reference_method": original_method,
                "reference_group_size": original_group_size,
                "candidate_method": candidate_method,
                "candidate_group_size": args.target_group_size,
                "metrics": candidate["summary"],
                "delta_vs_hard": delta,
                "paired_evidence": paired_metric_evidence(
                    candidate["rows"],
                    baseline["rows"],
                    bootstrap_indices,
                ),
                "storage": {
                    **storage,
                    "scale_bytes_delta": scale_bytes_delta,
                },
                "ground_truth_db_per_added_kib": (
                    delta["psnr_vs_ground_truth"]
                    / (scale_bytes_delta / 1024)
                    if scale_bytes_delta > 0
                    else None
                ),
            })
            if index % 10 == 0:
                print(json.dumps({
                    "evaluated_surfaces": index,
                    "total_surfaces": len(requested),
                }), flush=True)
    results.sort(
        key=lambda item: (
            -item["delta_vs_hard"]["psnr_vs_ground_truth"],
            item["storage"]["scale_bytes_delta"],
            item["name"],
        )
    )
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "development_screen",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "device": args.device,
        "torch_threads": args.torch_threads,
        "bootstrap": {
            "method": "paired_nonparametric_mean",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
        "sample_count": len(samples),
        "samples": [
            {
                key: sample.get(key)
                for key in ("sha256", "source", "script", "language")
            }
            for sample in samples
        ],
        "target_group_size": args.target_group_size,
        "candidate_method": args.method,
        "surface_count": len(requested),
        "total_quantized_values": total_values,
        "hard": baseline["summary"],
        "surfaces": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "sample_count": report["sample_count"],
        "surface_count": report["surface_count"],
        "top_surfaces": [
            {
                "name": surface["name"],
                "delta_vs_hard": surface["delta_vs_hard"],
                "scale_bytes_delta": surface["storage"]["scale_bytes_delta"],
            }
            for surface in results[:15]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
