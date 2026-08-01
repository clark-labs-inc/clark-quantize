#!/usr/bin/env python3
"""Rank endpoint scale-codebook approximation error by exact Swin2SR surface."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from scale_codebook_state import load_scale_codebook_student


FORMAT = "backdropkit-swin2sr-trained-scale-codebook-analysis-v1"


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError("scale-codebook tensors have different shapes")
    reference = reference.detach().float().reshape(-1)
    candidate = candidate.detach().float().reshape(-1)
    delta = candidate - reference
    error_squared = float(torch.dot(delta, delta).item())
    signal_squared = float(torch.dot(reference, reference).item())
    return {
        "values": reference.numel(),
        "error_squared": error_squared,
        "signal_squared": signal_squared,
        "relative_rmse": math.sqrt(error_squared / max(signal_squared, 1e-30)),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "max_abs_delta": float(delta.abs().max().item()),
    }


def analyze(checkpoint: Path) -> dict:
    student, surfaces, codebooks, payload, _parent = load_scale_codebook_student(
        checkpoint,
        "cpu",
    )
    by_name = {item.name: item for item in codebooks}
    rows = []
    category_totals = defaultdict(lambda: {
        "surfaces": 0,
        "values": 0,
        "error_squared": 0.0,
        "signal_squared": 0.0,
    })
    for surface in surfaces:
        codebook = by_name[surface.name]
        surface.quantizer.amount.fill_(1.0)
        codebook.projection.amount.fill_(0.0)
        without_codebook = surface.module.weight.detach().clone()
        codebook.projection.amount.fill_(1.0)
        with_codebook = surface.module.weight.detach().clone()
        stats = error_stats(without_codebook, with_codebook)
        centers = codebook.projection.stored_centers.detach().float()
        row = {
            "name": surface.name,
            "category": surface.category,
            "target_bits": codebook.target_bits,
            "group_size": surface.quantizer.group_size,
            "center_count": centers.numel(),
            "center_min": float(centers.min().item()),
            "center_max": float(centers.max().item()),
            **stats,
        }
        rows.append(row)
        total = category_totals[surface.category]
        total["surfaces"] += 1
        total["values"] += stats["values"]
        total["error_squared"] += stats["error_squared"]
        total["signal_squared"] += stats["signal_squared"]
    total_error = sum(row["error_squared"] for row in rows)
    for row in rows:
        row["error_share"] = row["error_squared"] / max(total_error, 1e-30)
    categories = {}
    for category, total in sorted(category_totals.items()):
        categories[category] = {
            **total,
            "relative_rmse": math.sqrt(
                total["error_squared"] / max(total["signal_squared"], 1e-30)
            ),
            "error_share": total["error_squared"] / max(total_error, 1e-30),
        }
    rows.sort(key=lambda row: (-row["error_squared"], row["name"]))
    return {
        "format": FORMAT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "scale_step": int(payload["scale_step"]),
        "interpretation": (
            "Errors compare the trained hard ternary surface immediately before "
            "and after its trained hard scale codebook. They isolate endpoint "
            "scale approximation, not model-output sensitivity."
        ),
        "surface_count": len(rows),
        "total_error_squared": total_error,
        "categories": categories,
        "surfaces": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "categories": report["categories"],
        "top_surfaces": report["surfaces"][:10],
    }, indent=2))


if __name__ == "__main__":
    main()
