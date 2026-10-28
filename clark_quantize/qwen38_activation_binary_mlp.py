"""Activation-calibrated binary residuals for a complete Qwen3.8-27B MLP.

This module supplies the bounded numerical pieces for the second rung after a
full real-weight binary-low-rank screen.  The exported runtime form remains a
row-normalized fixed-support ternary matrix plus signed INT4 left factors and
binary-right factors.  Only the left coefficients are recalibrated on real
layer inputs; no dense correction is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from .qwen38_binary_low_rank_residual import shared_binary_low_rank_storage_projection
from .qwen38_hash_residual import (
    fixed_support_ternary,
    quantize_hash_coefficients_int4,
)
from .qwen38_row_scale_codebook import fit_row_normalized_log_codebook


@dataclass(frozen=True)
class ActivationLeftFit:
    codes: torch.Tensor
    scales: torch.Tensor
    coefficients: torch.Tensor
    calibration_squared_error: float


@dataclass(frozen=True)
class ExportedTernaryMatrix:
    weights: torch.Tensor
    scale_record: dict


def exported_int4_left(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if (
        codes.ndim != 2
        or scales.ndim != 1
        or codes.shape[0] != scales.numel()
        or codes.dtype != torch.int8
        or bool((codes < -7).any())
        or bool((codes > 7).any())
        or not scales.is_floating_point()
        or not bool(torch.isfinite(scales).all())
        or bool((scales <= 0).any())
    ):
        raise ValueError("invalid exported INT4 left factor")
    return (
        codes.to(torch.float32)
        * scales.to(torch.float16).to(torch.float32)[:, None]
    ).contiguous()


@torch.no_grad()
def fit_activation_left_int4(
    features: torch.Tensor,
    target_outputs: torch.Tensor,
    *,
    relative_ridge: float = 1e-4,
) -> ActivationLeftFit:
    """Fit exported per-output INT4 coefficients for frozen rank features."""
    if (
        features.ndim != 2
        or target_outputs.ndim != 2
        or features.shape[0] != target_outputs.shape[0]
        or features.shape[0] == 0
        or features.shape[1] == 0
        or target_outputs.shape[1] == 0
        or not features.is_floating_point()
        or not target_outputs.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(target_outputs).all())
        or relative_ridge <= 0
    ):
        raise ValueError("invalid activation-left fitting problem")
    x = features.to(torch.float32)
    y = target_outputs.to(torch.float32)
    gram = x.T @ x
    ridge = relative_ridge * max(float(torch.diagonal(gram).mean().item()), 1e-20)
    regularized = gram + torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    continuous = torch.linalg.solve(regularized, x.T @ y).T.contiguous()
    codes, scales, exported = quantize_hash_coefficients_int4(continuous)
    predicted = x @ exported.T
    error = float(
        (y - predicted).square().sum(dtype=torch.float64).item()
    )
    return ActivationLeftFit(
        codes=codes,
        scales=scales,
        coefficients=exported,
        calibration_squared_error=error,
    )


@torch.no_grad()
def exported_row_normalized_ternary(
    source: torch.Tensor,
    *,
    group: int = 128,
    nonzero_per_group: int = 64,
) -> ExportedTernaryMatrix:
    """Apply the exact stored row-anchor/ratio-codebook scale contract."""
    if (
        source.ndim != 2
        or not source.is_floating_point()
        or source.numel() == 0
        or source.shape[1] % group
        or not bool(torch.isfinite(source).all())
    ):
        raise ValueError("invalid row-normalized ternary source")
    codes, scales, _ = fixed_support_ternary(
        source, group=group, nonzero_per_group=nonzero_per_group
    )
    fitted = fit_row_normalized_log_codebook(
        scales.cpu().numpy(), entries=16, iterations=50, chunk_rows=256
    )
    frozen_scales = torch.from_numpy(fitted.reconstructed).to(torch.float32)
    weights = (
        codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * frozen_scales[:, :, None]
    ).reshape_as(source).contiguous()
    scale_error = scales.to(torch.float32) - frozen_scales
    source_squared = float(scales.square().sum(dtype=torch.float64).item())
    error_squared = float(scale_error.square().sum(dtype=torch.float64).item())
    return ExportedTernaryMatrix(
        weights=weights,
        scale_record={
            "groups": int(scales.numel()),
            "row_anchor_bytes": int(fitted.anchors.nbytes),
            "ratio_codebook_bytes": int(fitted.values.nbytes),
            "scale_index_bytes": math.ceil(int(fitted.indices.size) / 2),
            "scale_normalized_rmse": math.sqrt(
                error_squared / max(source_squared, 1e-30)
            ),
        },
    )


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if (
        reference.shape != candidate.shape
        or reference.numel() == 0
        or not reference.is_floating_point()
        or not candidate.is_floating_point()
        or not bool(torch.isfinite(reference).all())
        or not bool(torch.isfinite(candidate).all())
    ):
        raise ValueError("invalid activation error comparison")
    source = reference.to(torch.float32)
    residual = source - candidate.to(torch.float32)
    source_squared = float(source.square().sum(dtype=torch.float64).item())
    squared_error = float(residual.square().sum(dtype=torch.float64).item())
    return {
        "squared_error": squared_error,
        "relative_nmse": squared_error / max(source_squared, 1e-30),
        "relative_rmse": math.sqrt(squared_error / max(source_squared, 1e-30)),
        "maximum_absolute_error": float(residual.abs().max().item()),
    }


def complete_mlp_storage_projection(
    *,
    hidden: int,
    intermediate: int,
    rank: int,
    group: int = 128,
    nonzero_per_group: int = 64,
) -> dict:
    """Count gate/up shared-right plus one independent down residual factor."""
    gate_up = shared_binary_low_rank_storage_projection(
        [(intermediate, hidden), (intermediate, hidden)],
        rank=rank,
        group=group,
        nonzero_per_group=nonzero_per_group,
    )
    down = shared_binary_low_rank_storage_projection(
        [(hidden, intermediate)],
        rank=rank,
        group=group,
        nonzero_per_group=nonzero_per_group,
    )
    parameters = int(gate_up["parameters"] + down["parameters"])
    bytes_total = int(gate_up["bytes"] + down["bytes"])
    return {
        "gate_up": gate_up,
        "down": down,
        "parameters": parameters,
        "bytes": bytes_total,
        "bits_per_weight": bytes_total * 8 / parameters,
    }

