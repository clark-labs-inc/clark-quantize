#!/usr/bin/env python3
"""Estimate layer group-size sensitivity from task-loss gradients."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_surface_restore_sensitivity import (
    load_sample,
    stratified_records,
)
from evaluate_hard_started_checkpoints import load_checkpoint_student
from evaluate_stratified import file_sha256, load_bundle_records
from probe_sensitive_io_group_sizes import storage_receipt
from qat_objective import distillation_losses
from recovery_controls import recovery_loss_weights
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-gradient-group-sensitivity-v1"
OBJECTIVES = ("charbonnier", "psnr_aligned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-group-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--empirical", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_empirical(
    results: list[dict],
    empirical_path: Path | None,
) -> dict | None:
    if empirical_path is None:
        return None
    empirical = json.loads(empirical_path.read_text())
    by_name = {row["name"]: row for row in empirical["surfaces"]}
    matched = [row for row in results if row["name"] in by_name]
    actual = [
        by_name[row["name"]]["delta_vs_hard"]["psnr_vs_ground_truth"]
        for row in matched
    ]
    return {
        "path": str(empirical_path),
        "sha256": file_sha256(empirical_path),
        "matched_surfaces": len(matched),
        "correlation": {
            objective: correlation(
                [
                    row["predicted_improvement"][objective]
                    for row in matched
                ],
                actual,
            )
            for objective in OBJECTIVES
        },
    }


def projection_delta(surface, target_group_size: int) -> torch.Tensor:
    quantizer = surface.quantizer
    master = surface.module.parametrizations.weight.original
    original_group_size = quantizer.group_size
    with torch.no_grad():
        current = quantizer.project(master).detach()
        quantizer.group_size = target_group_size
        try:
            candidate = quantizer.project(master).detach()
        finally:
            quantizer.group_size = original_group_size
    return candidate - current


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"gradient sensitivity exists: {args.out}")
    if args.target_group_size < 1 or args.torch_threads < 1:
        raise ValueError("group size and torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    bundle, records = load_bundle_records(args.bundle)
    selected_records = stratified_records(records, args.max_samples)
    samples = [load_sample(record, args.device) for record in selected_records]
    student, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        args.device,
    )
    if len(surfaces) != 204:
        raise RuntimeError("checkpoint does not cover all 204 surfaces")
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    teacher = pretrained(args.device).eval()
    student.eval()
    weights = recovery_loss_weights("balanced")
    total_values = sum(surface.module.weight.numel() for surface in surfaces)
    deltas = {
        surface.name: projection_delta(surface, args.target_group_size)
        for surface in surfaces
    }
    accumulators = {
        objective: defaultdict(float)
        for objective in OBJECTIVES
    }
    loss_totals = defaultdict(float)
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        high = torch.from_numpy(sample["truth"]).permute(2, 0, 1)
        high = high.unsqueeze(0).to(args.device)
        with torch.no_grad():
            teacher_output = teacher(
                sample["low"],
                output_hidden_states=True,
            )
        student_output = student(
            sample["low"],
            output_hidden_states=True,
        )
        reconstruction = student_output.reconstruction.clamp(0, 1)
        teacher_reconstruction = teacher_output.reconstruction.clamp(0, 1)
        losses = {}
        for objective in OBJECTIVES:
            components = distillation_losses(
                reconstruction,
                teacher_reconstruction,
                high,
                (
                    student_output.hidden_states,
                    teacher_output.hidden_states,
                ),
                pixel_objective=objective,
                ground_truth_mse_floor=3.2e-4,
                teacher_mse_floor=2e-5,
            )
            losses[objective] = sum(
                weights[name] * components[name]
                for name in weights
            )
            loss_totals[objective] += float(losses[objective].detach())
        for objective_index, objective in enumerate(OBJECTIVES):
            student.zero_grad(set_to_none=True)
            losses[objective].backward(
                retain_graph=objective_index < len(OBJECTIVES) - 1
            )
            for surface in surfaces:
                master = surface.module.parametrizations.weight.original
                if master.grad is None:
                    raise RuntimeError(
                        f"surface has no task gradient: {surface.name}"
                    )
                change = torch.sum(
                    master.grad.detach().float()
                    * deltas[surface.name].float()
                ).item()
                accumulators[objective][surface.name] += change
        print(json.dumps({
            "gradient_samples": index,
            "total_samples": len(samples),
        }), flush=True)

    results = []
    for surface in surfaces:
        original_group_size = surface.quantizer.group_size
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
        candidate_minus_current = {
            objective: (
                accumulators[objective][surface.name] / len(samples)
            )
            for objective in OBJECTIVES
        }
        results.append({
            "name": surface.name,
            "category": surface.category,
            "values": surface.module.weight.numel(),
            "method": surface.quantizer.method,
            "reference_group_size": original_group_size,
            "candidate_group_size": args.target_group_size,
            "predicted_candidate_minus_current_loss": (
                candidate_minus_current
            ),
            "predicted_improvement": {
                objective: -value
                for objective, value in candidate_minus_current.items()
            },
            "scale_bytes_delta": scale_bytes_delta,
        })
    results.sort(
        key=lambda item: (
            -item["predicted_improvement"]["psnr_aligned"],
            item["scale_bytes_delta"],
            item["name"],
        )
    )
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "development_screen",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "method": (
            "First-order task-loss change from the current hard projection "
            "to a finer ternary scale group, evaluated at the selected g2 "
            "checkpoint. Negative candidate-minus-current loss is favorable."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "device": args.device,
        "torch_threads": args.torch_threads,
        "target_group_size": args.target_group_size,
        "sample_count": len(samples),
        "samples": [
            {
                key: sample.get(key)
                for key in ("sha256", "source", "script", "language")
            }
            for sample in samples
        ],
        "mean_objective_loss": {
            objective: loss_totals[objective] / len(samples)
            for objective in OBJECTIVES
        },
        "surface_count": len(results),
        "surfaces": results,
        "empirical_comparison": compare_empirical(
            results,
            args.empirical,
        ),
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
        "top_surfaces": results[:15],
        "empirical_comparison": report["empirical_comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
