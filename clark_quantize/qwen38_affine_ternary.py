"""CPU-compatible affine ternary fitting for Qwen3.8-27B matrices.

The representation is

    W[r, g, i] ~= center[r] + scale[r, g] * code[r, g, i],
    code in {-1, 0, +1}.

Only one center is stored per output row.  During GEMV its contribution is
``center[r] * sum(x)``.  Clark's ternary CPU path already forms activation
sums while centering stored digits, so the affine correction needs one extra
row FMA rather than a dense residual matrix.

For fixed codes, ``center`` and every group scale are the exact weighted least
squares solution after analytically eliminating the scales.  For fixed
parameters, nearest-level assignment is exact.  Alternating these two steps
therefore cannot increase the diagonal activation-weighted reconstruction
objective (up to floating-point roundoff).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ternary import ternary_codes_scales


@dataclass(frozen=True)
class AffineTernaryFit:
    codes: torch.Tensor
    scales: torch.Tensor
    centers: torch.Tensor
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool


def _validate_inputs(
    weight: torch.Tensor,
    activation_weights: torch.Tensor | None,
    group: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("affine ternary fitting expects a floating-point matrix")
    if weight.numel() == 0 or group <= 0:
        raise ValueError("weight must be nonempty and group must be positive")
    source = weight.detach().to(torch.float32)
    columns = source.shape[1]
    if activation_weights is None:
        diagonal = torch.ones(columns, device=source.device, dtype=torch.float32)
    else:
        diagonal = activation_weights.detach().to(
            device=source.device, dtype=torch.float32
        )
        if diagonal.ndim != 1 or diagonal.numel() != columns:
            raise ValueError("activation weights must have one value per input column")
        if not torch.isfinite(diagonal).all() or torch.any(diagonal < 0):
            raise ValueError("activation weights must be finite and nonnegative")
        if not torch.any(diagonal > 0):
            raise ValueError("at least one activation weight must be positive")
    if not torch.isfinite(source).all():
        raise ValueError("weight matrix must be finite")
    pad = (-columns) % group
    if pad:
        source = torch.nn.functional.pad(source, (0, pad))
        diagonal = torch.nn.functional.pad(diagonal, (0, pad))
    return source, diagonal, columns


def _weighted_objective(
    source: torch.Tensor,
    diagonal: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    reconstructed = centers[:, None, None] + scales[:, :, None] * codes
    residual = source - reconstructed
    return (diagonal * residual.square()).sum(dtype=torch.float64)


def _solve_affine_parameters(
    source: torch.Tensor,
    diagonal: torch.Tensor,
    codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve every row center and group scale exactly for fixed codes."""
    weighted = diagonal * source
    h_sum = diagonal.sum(dim=-1)
    c_sum = weighted.sum(dim=-1)
    q = codes
    a = (diagonal * q.square()).sum(dim=-1)
    b = (diagonal * q).sum(dim=-1)
    d = (weighted * q).sum(dim=-1)
    active = a > torch.finfo(torch.float32).eps
    safe_a = torch.where(active, a, torch.ones_like(a))
    denominator_terms = torch.where(active, h_sum - b.square() / safe_a, h_sum)
    numerator_terms = torch.where(active, c_sum - b * d / safe_a, c_sum)
    denominator = denominator_terms.sum(dim=-1)
    numerator = numerator_terms.sum(dim=-1)
    identifiable = denominator.abs() > torch.finfo(torch.float32).eps
    centers = torch.where(
        identifiable,
        numerator / torch.where(identifiable, denominator, torch.ones_like(denominator)),
        torch.zeros_like(numerator),
    )
    scales = torch.where(
        active,
        (d - centers[:, None] * b) / safe_a,
        torch.zeros_like(a),
    )
    negative = scales < 0
    canonical_codes = torch.where(negative[:, :, None], -q, q)
    scales = scales.abs()
    reconstructed = centers[:, None, None] + scales[:, :, None] * canonical_codes
    return centers, scales, reconstructed


def _weighted_row_median(source: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    ordered, permutation = torch.sort(source, dim=1)
    ordered_weights = diagonal.expand_as(source).gather(1, permutation)
    cutoff = 0.5 * ordered_weights.sum(dim=1, keepdim=True)
    index = (ordered_weights.cumsum(dim=1) >= cutoff).to(torch.int64).argmax(dim=1)
    return ordered.gather(1, index[:, None]).squeeze(1)


@torch.no_grad()
def affine_ternary_codes_scales(
    weight: torch.Tensor,
    *,
    group: int = 128,
    activation_weights: torch.Tensor | None = None,
    max_iterations: int = 12,
    relative_tolerance: float = 1e-7,
) -> AffineTernaryFit:
    """Fit a row-centered, group-scaled ternary matrix by coordinate descent.

    ``activation_weights`` is the diagonal of an input second-moment matrix.
    With calibration activations ``X``, use ``mean(X**2, axis=0)``.  Uniform
    weights recover ordinary Frobenius fitting.  The returned tensors exclude
    padding and store centers/scales as FP32; export policy chooses their final
    storage precision separately.
    """
    if max_iterations <= 0 or relative_tolerance < 0:
        raise ValueError("iterations must be positive and tolerance nonnegative")
    source, diagonal, original_columns = _validate_inputs(
        weight, activation_weights, group
    )
    rows, padded_columns = source.shape
    groups = padded_columns // group
    grouped_source = source.reshape(rows, groups, group)
    grouped_diagonal = diagonal.reshape(1, groups, group)

    initial_codes, initial_scales = ternary_codes_scales(
        source, group=group, method="optimal"
    )
    symmetric_codes = initial_codes.reshape(rows, groups, group).to(torch.float32)
    codes = symmetric_codes
    scales = initial_scales.to(torch.float32)
    centers = torch.zeros(rows, device=source.device, dtype=torch.float32)
    history = [
        float(
            _weighted_objective(
                grouped_source, grouped_diagonal, codes, scales, centers
            ).item()
        )
    ]

    # A translated three-level line can be far from the origin.  Use a small,
    # deterministic multistart set, but retain the symmetric fit as a candidate
    # and as the frozen no-regression objective.
    weighted_mean = (source * diagonal).sum(dim=1) / diagonal.sum()
    weighted_median = _weighted_row_median(source, diagonal)
    candidate_codes = [symmetric_codes]
    for proposed_center in (weighted_mean, weighted_median):
        centered_codes, _ = ternary_codes_scales(
            source - proposed_center[:, None], group=group, method="optimal"
        )
        candidate_codes.append(
            centered_codes.reshape(rows, groups, group).to(torch.float32)
        )
    best_objective = float("inf")
    for candidate in candidate_codes:
        candidate_centers, candidate_scales, _ = _solve_affine_parameters(
            grouped_source, grouped_diagonal, candidate
        )
        candidate_objective = float(
            _weighted_objective(
                grouped_source,
                grouped_diagonal,
                candidate,
                candidate_scales,
                candidate_centers,
            ).item()
        )
        if candidate_objective < best_objective:
            best_objective = candidate_objective
            codes = candidate
    converged = False
    completed = 0

    for iteration in range(1, max_iterations + 1):
        centers, scales, _ = _solve_affine_parameters(
            grouped_source, grouped_diagonal, codes
        )
        centered = grouped_source - centers[:, None, None]
        thresholds = 0.5 * scales[:, :, None]
        reassigned = torch.sign(centered) * (
            centered.abs() >= thresholds
        ).to(centered.dtype)
        reassigned = torch.where(grouped_diagonal > 0, reassigned, torch.zeros_like(reassigned))
        unchanged = torch.equal(reassigned, codes)
        codes = reassigned
        centers, scales, _ = _solve_affine_parameters(
            grouped_source, grouped_diagonal, codes
        )
        objective = float(
            _weighted_objective(
                grouped_source, grouped_diagonal, codes, scales, centers
            ).item()
        )
        previous = history[-1]
        numerical_slack = max(1e-10, abs(previous) * 5e-7)
        if objective > previous + numerical_slack:
            raise RuntimeError("affine ternary coordinate descent increased its objective")
        history.append(objective)
        completed = iteration
        relative_gain = (previous - objective) / max(abs(previous), 1e-30)
        if unchanged or relative_gain <= relative_tolerance:
            converged = True
            break

    codes_i8 = codes.reshape(rows, padded_columns)[:, :original_columns].to(torch.int8)
    return AffineTernaryFit(
        codes=codes_i8.contiguous(),
        scales=scales.contiguous(),
        centers=centers.contiguous(),
        objective_history=tuple(history),
        iterations=completed,
        converged=converged,
    )


@torch.no_grad()
def reconstruct_affine_ternary(
    fit: AffineTernaryFit,
    *,
    group: int = 128,
) -> torch.Tensor:
    """Reconstruct the represented matrix exactly in FP32 arithmetic."""
    if fit.codes.ndim != 2 or fit.scales.ndim != 2 or fit.centers.ndim != 1:
        raise ValueError("invalid affine ternary fit dimensions")
    rows, columns = fit.codes.shape
    if fit.centers.numel() != rows or fit.scales.shape[0] != rows:
        raise ValueError("affine ternary row dimensions disagree")
    expected_groups = (columns + group - 1) // group
    if fit.scales.shape[1] != expected_groups:
        raise ValueError("affine ternary scale dimensions disagree")
    expanded_scales = fit.scales.repeat_interleave(group, dim=1)[:, :columns]
    return (
        fit.centers.to(torch.float32)[:, None]
        + expanded_scales.to(torch.float32) * fit.codes.to(torch.float32)
    ).contiguous()
