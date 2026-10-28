"""Activation-weighted Clark-Hash residual fitting for Qwen3.8-27B.

The data-free hash screen fits ``W ~= T + A R`` in weight space.  This module
keeps the frozen ternary matrix ``T`` and sparse signed projection ``R`` but
refits the small coefficient matrix ``A`` against real layer inputs.  Only the
coefficient matrix is trained, so the method remains bounded and row-shardable
on a CPU host.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from .qwen38_hash_residual import (
    SparseJLProjection,
    quantize_hash_coefficients_int4,
    sketch_rows,
)


@dataclass(frozen=True)
class Int4ActivationFit:
    coefficients: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    calibration_history: tuple[dict[str, float | int], ...]
    best_step: int


def _validate_problem(
    residual: torch.Tensor,
    inputs: torch.Tensor,
    projection: SparseJLProjection,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.ndim != 2
        or inputs.ndim != 2
        or not residual.is_floating_point()
        or not inputs.is_floating_point()
        or residual.shape[1] != projection.input_dim
        or inputs.shape[1] != projection.input_dim
        or inputs.shape[0] == 0
    ):
        raise ValueError("activation-hash residual and input dimensions disagree")
    source = residual.detach().to(torch.float32).contiguous()
    activation = inputs.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(activation).all()):
        raise ValueError("activation-hash fitting requires finite tensors")
    return source, activation


def output_error_stats(
    residual: torch.Tensor,
    coefficients: torch.Tensor,
    inputs: torch.Tensor,
    projection: SparseJLProjection,
) -> dict[str, float]:
    """Measure ``A(Rx)`` against the exact residual-layer output."""
    source, activation = _validate_problem(residual, inputs, projection)
    if coefficients.shape != (source.shape[0], projection.sketch_dim):
        raise ValueError("activation-hash coefficient dimensions disagree")
    target = activation @ source.T
    candidate = sketch_rows(activation, projection) @ coefficients.float().T
    difference = candidate - target
    squared_error = float(difference.square().sum(dtype=torch.float64).item())
    target_energy = float(target.square().sum(dtype=torch.float64).item())
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


@torch.no_grad()
def activation_weighted_hash_coefficients(
    residual: torch.Tensor,
    inputs: torch.Tensor,
    projection: SparseJLProjection,
    *,
    reference_coefficients: torch.Tensor | None = None,
    relative_ridge: float = 1e-3,
) -> torch.Tensor:
    """Solve the activation-weighted coefficient regression exactly.

    The objective is ``||X residual.T - (X R.T) A.T||_F^2``.  When a reference
    is supplied, ridge regularization is centered on that data-free solution,
    which keeps underdetermined activation screens deterministic and bounded.
    """
    if relative_ridge <= 0:
        raise ValueError("activation-hash relative ridge must be positive")
    source, activation = _validate_problem(residual, inputs, projection)
    if reference_coefficients is None:
        reference = torch.zeros(
            source.shape[0], projection.sketch_dim, dtype=torch.float32
        )
    else:
        if reference_coefficients.shape != (source.shape[0], projection.sketch_dim):
            raise ValueError("activation-hash reference dimensions disagree")
        reference = reference_coefficients.detach().to(torch.float32).contiguous()
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("activation-hash reference must be finite")

    sketched = sketch_rows(activation, projection)
    target = activation @ source.T
    gram = sketched.T @ sketched
    diagonal_mean = float(torch.diagonal(gram).mean().item())
    ridge = relative_ridge * max(diagonal_mean, 1e-12)
    regularized = gram + torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    right = sketched.T @ target + reference.T * ridge
    fitted = torch.linalg.solve(regularized, right).T.contiguous()
    if not bool(torch.isfinite(fitted).all()):
        raise FloatingPointError("activation-weighted hash solve produced non-finite values")
    return fitted


def _int4_ste(coefficients: torch.Tensor) -> torch.Tensor:
    scales = (coefficients.detach().abs().amax(dim=1) / 7.0).clamp_min(1e-12)
    hard = (
        torch.round(coefficients / scales[:, None])
        .clamp(-7, 7)
        .to(torch.int8)
        .to(torch.float32)
        * scales.to(torch.float16).to(torch.float32)[:, None]
    )
    return coefficients + (hard - coefficients).detach()


def _normalized_mse(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.mse_loss(candidate.float(), target.float()) / target.float().square().mean().clamp_min(1e-12)


def train_int4_activation_hash_coefficients(
    residual: torch.Tensor,
    inputs: torch.Tensor,
    projection: SparseJLProjection,
    initial_coefficients: torch.Tensor,
    *,
    reference_coefficients: torch.Tensor | None = None,
    steps: int = 12,
    learning_rate: float = 3e-4,
    anchor_weight: float = 1e-3,
    gradient_clip_norm: float = 1.0,
) -> Int4ActivationFit:
    """Run a short hard-INT4 coefficient fit and retain its best train step."""
    if (
        steps <= 0
        or learning_rate <= 0
        or anchor_weight < 0
        or gradient_clip_norm <= 0
    ):
        raise ValueError("invalid activation-hash INT4 training configuration")
    source, activation = _validate_problem(residual, inputs, projection)
    expected = (source.shape[0], projection.sketch_dim)
    if initial_coefficients.shape != expected:
        raise ValueError("activation-hash initial coefficient dimensions disagree")
    reference = (
        initial_coefficients.detach().float()
        if reference_coefficients is None
        else reference_coefficients.detach().float()
    )
    if reference.shape != expected or not bool(torch.isfinite(reference).all()):
        raise ValueError("activation-hash training reference is invalid")

    sketched = sketch_rows(activation, projection)
    target = activation @ source.T
    master = torch.nn.Parameter(initial_coefficients.detach().float().contiguous())
    optimizer = torch.optim.AdamW([master], lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []

    with torch.no_grad():
        _, initial_scales, initial_export = quantize_hash_coefficients_int4(master)
        best_coefficients = initial_export.clone()
        best_loss = float(_normalized_mse(sketched @ initial_export.T, target).item())
        best_step = 0
        best_scales = initial_scales.clone()

    reference_energy = reference.square().mean().clamp_min(1e-12)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        hard = _int4_ste(master)
        output_loss = _normalized_mse(sketched @ hard.T, target)
        anchor_loss = (hard - reference).square().mean() / reference_energy
        loss = output_loss + anchor_weight * anchor_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("activation-hash INT4 loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([master], gradient_clip_norm)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("activation-hash INT4 gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            codes, scales, exported = quantize_hash_coefficients_int4(master)
            exported_loss = float(_normalized_mse(sketched @ exported.T, target).item())
            history.append(
                {
                    "step": step,
                    "ste_loss": float(loss.detach().item()),
                    "ste_output_nmse": float(output_loss.detach().item()),
                    "ste_anchor_nmse": float(anchor_loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm.detach().item()),
                    "exported_output_nmse": exported_loss,
                }
            )
            if exported_loss < best_loss:
                best_loss = exported_loss
                best_step = step
                best_coefficients = exported.clone()
                best_scales = scales.clone()

    best_codes = torch.round(best_coefficients / best_scales.float()[:, None]).clamp(-7, 7).to(torch.int8)
    checked = best_codes.float() * best_scales.float()[:, None]
    if not torch.equal(checked, best_coefficients):
        raise AssertionError("activation-hash INT4 export is not canonical")
    return Int4ActivationFit(
        coefficients=best_coefficients.contiguous(),
        codes=best_codes.contiguous(),
        scales=best_scales.contiguous(),
        calibration_history=tuple(history),
        best_step=best_step,
    )


def relative_error_reduction(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    if candidate["target_squared_sum"] != baseline["target_squared_sum"]:
        raise ValueError("activation-hash error targets changed")
    return 1.0 - candidate["squared_error"] / max(baseline["squared_error"], 1e-30)
