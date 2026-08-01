#!/usr/bin/env python3
"""Screen ternary group-size changes on development data excluding prior use."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from analyze_surface_group_sensitivity import (
    METHODS,
    paired_metric_evidence,
)
from analyze_surface_restore_sensitivity import (
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


FORMAT = "backdropkit-swin2sr-disjoint-surface-group-sensitivity-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, action="append", default=[])
    parser.add_argument("--target-group-size", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, default="current")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--surface", action="append")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict]]:
    identities = set()
    receipts = []
    for path in paths:
        report = json.loads(path.read_text())
        samples = report.get("samples")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(f"exclusion report has no samples: {path}")
        report_identities = {
            str(sample["sha256"])
            for sample in samples
        }
        identities.update(report_identities)
        receipts.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "sample_count": len(samples),
            "unique_sample_count": len(report_identities),
        })
    return identities, receipts


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"group-sensitivity output exists: {args.out}")
    if min(
        args.target_group_size,
        args.max_samples,
        args.bootstrap_resamples,
        args.torch_threads,
    ) < 1:
        raise ValueError("group size, samples, resamples, and threads must be positive")
    torch.set_num_threads(args.torch_threads)
    bundle, records = load_bundle_records(args.bundle)
    excluded, exclusion_receipts = load_exclusions(args.exclude_report)
    available_records = [
        record for record in records
        if record["sha256"] not in excluded
    ]
    selected_records = stratified_records(available_records, args.max_samples)
    if len(selected_records) != args.max_samples:
        raise RuntimeError("insufficient disjoint development samples")
    if any(record["sha256"] in excluded for record in selected_records):
        raise RuntimeError("disjoint selection reused an excluded sample")
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
        baseline = state_metrics(student, samples, teacher_outputs, args.device)
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
                "delta_vs_hard": metric_delta(
                    candidate["summary"],
                    baseline["summary"],
                ),
                "paired_evidence": paired_metric_evidence(
                    candidate["rows"],
                    baseline["rows"],
                    bootstrap_indices,
                ),
                "storage": {
                    **storage,
                    "scale_bytes_delta": scale_bytes_delta,
                },
            })
            if index % 10 == 0:
                print(json.dumps({
                    "evaluated_surfaces": index,
                    "total_surfaces": len(requested),
                }), flush=True)
    results.sort(key=lambda item: (
        -item["delta_vs_hard"]["psnr_vs_ground_truth"],
        item["storage"]["scale_bytes_delta"],
        item["name"],
    ))
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "disjoint_development_screen",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "exclusion_receipts": exclusion_receipts,
        "excluded_unique_sample_count": len(excluded),
        "device": args.device,
        "torch_threads": args.torch_threads,
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
        "bootstrap": {
            "method": "paired_nonparametric_mean",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
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
