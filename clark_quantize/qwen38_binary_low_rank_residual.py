"""Rate-constrained binary-right low-rank residuals for Qwen3.8-27B.

The deployable matrix is

    W_hat = T + A C,

where ``T`` is the fixed-support ternary backbone, ``A`` is signed per-row
INT4, and every row of ``C`` is an FP16 scale times a binary sign vector.
Gate and up projections may share ``C`` because they consume the same input.
This is a deliberately stronger, still CPU-cheap successor to a one-sparse
Clark-Hash route; it is not by itself a model-quality or novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .qwen38_fixed_support_codec import fixed_support_width_bits
from .qwen38_hash_residual import quantize_hash_coefficients_int4


@dataclass(frozen=True)
class BinaryRightFactor:
    signs: torch.Tensor
    scales: torch.Tensor


@dataclass(frozen=True)
class BinaryLowRankFit:
    right: BinaryRightFactor
    left_codes: torch.Tensor
    left_scales: torch.Tensor
    left_coefficients: torch.Tensor
    initial_squared_error: float
    objective_history: tuple[float, ...]
    best_iteration: int


def _validate_right(factor: BinaryRightFactor) -> None:
    if (
        factor.signs.ndim != 2
        or factor.scales.ndim != 1
        or factor.signs.shape[0] != factor.scales.numel()
        or factor.signs.numel() == 0
        or factor.signs.dtype != torch.int8
        or not bool(((factor.signs == -1) | (factor.signs == 1)).all())
        or not factor.scales.is_floating_point()
        or not bool(torch.isfinite(factor.scales).all())
        or bool((factor.scales <= 0).any())
    ):
        raise ValueError("binary right factor is invalid")


def binary_right_values(factor: BinaryRightFactor) -> torch.Tensor:
    _validate_right(factor)
    return (
        factor.signs.to(torch.float32)
        * factor.scales.to(torch.float32)[:, None]
    ).contiguous()


def quantize_binary_rows(values: torch.Tensor) -> BinaryRightFactor:
    if (
        values.ndim != 2
        or values.numel() == 0
        or not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("binary row quantization requires a finite matrix")
    signs = torch.where(
        values >= 0,
        torch.ones_like(values, dtype=torch.int8),
        -torch.ones_like(values, dtype=torch.int8),
    )
    scales = values.abs().mean(dim=1).clamp_min(1e-12).to(torch.float16)
    if bool((scales == 0).any()) or not bool(torch.isfinite(scales).all()):
        raise ValueError("binary right scale is not representable in FP16")
    return BinaryRightFactor(signs=signs.contiguous(), scales=scales.contiguous())


def expand_binary_low_rank(
    left_coefficients: torch.Tensor,
    right: BinaryRightFactor,
) -> torch.Tensor:
    values = binary_right_values(right)
    if (
        left_coefficients.ndim != 2
        or not left_coefficients.is_floating_point()
        or left_coefficients.shape[1] != values.shape[0]
    ):
        raise ValueError("binary low-rank dimensions disagree")
    return (left_coefficients.to(torch.float32) @ values).contiguous()


def binary_low_rank_outputs(
    inputs: torch.Tensor,
    left_coefficients: torch.Tensor,
    right: BinaryRightFactor,
) -> torch.Tensor:
    values = binary_right_values(right)
    if (
        inputs.ndim != 2
        or inputs.shape[1] != values.shape[1]
        or left_coefficients.ndim != 2
        or left_coefficients.shape[1] != values.shape[0]
    ):
        raise ValueError("binary low-rank output dimensions disagree")
    return (
        (inputs.to(torch.float32) @ values.T)
        @ left_coefficients.to(torch.float32).T
    ).contiguous()


def _ridge_solve_left(
    target: torch.Tensor,
    right: torch.Tensor,
    relative_ridge: float,
) -> torch.Tensor:
    gram = right @ right.T
    ridge = relative_ridge * max(float(torch.diagonal(gram).mean().item()), 1e-20)
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * ridge
    return torch.linalg.solve(regularized, (target @ right.T).T).T.contiguous()


def _ridge_solve_right(
    target: torch.Tensor,
    left: torch.Tensor,
    relative_ridge: float,
) -> torch.Tensor:
    gram = left.T @ left
    ridge = relative_ridge * max(float(torch.diagonal(gram).mean().item()), 1e-20)
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * ridge
    return torch.linalg.solve(regularized, left.T @ target).contiguous()


@torch.no_grad()
def fit_binary_low_rank_residual(
    residual: torch.Tensor,
    *,
    rank: int,
    iterations: int = 4,
    pca_iterations: int = 2,
    seed: int = 0xB17A4E,
    relative_ridge: float = 1e-5,
) -> BinaryLowRankFit:
    """Fit an exported INT4-by-binary correction to a dense residual.

    Randomized PCA supplies a strong right-subspace initialization. Alternating
    ridge solves then optimize against the *exported* binary right and INT4
    left factors. The best exported iterate is retained, so additional fitting
    iterations cannot worsen the returned weight-space objective.
    """
    if (
        residual.ndim != 2
        or not residual.is_floating_point()
        or residual.numel() == 0
        or rank <= 0
        or rank > min(residual.shape)
        or iterations <= 0
        or pca_iterations < 0
        or relative_ridge <= 0
    ):
        raise ValueError("invalid binary low-rank fitting problem")
    source = residual.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(source).all()):
        raise ValueError("binary low-rank fitting requires finite values")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, right_vectors = torch.pca_lowrank(
            source, q=rank, center=False, niter=pca_iterations
        )
    right = quantize_binary_rows(right_vectors.T.contiguous())
    source_error = float(source.square().sum(dtype=torch.float64).item())
    history: list[float] = []
    best_error = math.inf
    best_iteration = -1
    best_right = right
    best_codes = best_scales = best_left = None

    for iteration in range(iterations):
        right_values = binary_right_values(right)
        continuous_left = _ridge_solve_left(source, right_values, relative_ridge)
        left_codes, left_scales, exported_left = quantize_hash_coefficients_int4(
            continuous_left
        )
        continuous_right = _ridge_solve_right(
            source, exported_left, relative_ridge
        )
        right = quantize_binary_rows(continuous_right)
        right_values = binary_right_values(right)
        continuous_left = _ridge_solve_left(source, right_values, relative_ridge)
        left_codes, left_scales, exported_left = quantize_hash_coefficients_int4(
            continuous_left
        )
        error = float(
            (source - exported_left @ right_values)
            .square()
            .sum(dtype=torch.float64)
            .item()
        )
        history.append(error)
        if error < best_error:
            best_error = error
            best_iteration = iteration + 1
            best_right = BinaryRightFactor(
                signs=right.signs.clone(), scales=right.scales.clone()
            )
            best_codes = left_codes.clone()
            best_scales = left_scales.clone()
            best_left = exported_left.clone()

    if best_codes is None or best_scales is None or best_left is None:
        raise AssertionError("binary low-rank fitting produced no export")
    if best_error > source_error + max(1e-8, source_error * 1e-6):
        raise RuntimeError("binary low-rank correction is worse than zero")
    return BinaryLowRankFit(
        right=best_right,
        left_codes=best_codes.contiguous(),
        left_scales=best_scales.contiguous(),
        left_coefficients=best_left.contiguous(),
        initial_squared_error=source_error,
        objective_history=tuple(history),
        best_iteration=best_iteration,
    )


def shared_binary_low_rank_storage_projection(
    matrices: Sequence[tuple[int, int]],
    *,
    rank: int,
    group: int = 128,
    nonzero_per_group: int = 64,
    scale_index_bits: int = 4,
    header_bytes: int = 64,
) -> dict:
    """Count complete matrix bodies plus one shared binary right factor."""
    if (
        not matrices
        or rank <= 0
        or group <= 0
        or not 0 < nonzero_per_group <= group
        or scale_index_bits != 4
        or header_bytes < 0
        or any(rows <= 0 or columns <= 0 or columns % group for rows, columns in matrices)
        or len({columns for _, columns in matrices}) != 1
    ):
        raise ValueError("invalid shared binary low-rank storage problem")
    columns = matrices[0][1]
    group_width = fixed_support_width_bits(group, nonzero_per_group)
    matrix_records = []
    total_bytes = 0
    total_parameters = 0
    for rows, matrix_columns in matrices:
        groups = rows * (matrix_columns // group)
        components = {
            "header_bytes": header_bytes,
            "ternary_bytes": math.ceil(groups * group_width / 8),
            "row_anchor_bytes": rows * 2,
            "scale_index_bytes": math.ceil(groups * scale_index_bits / 8),
            "ratio_codebook_bytes": 16 * 2,
            "left_INT4_bytes": rows * math.ceil(rank * 4 / 8),
            "left_scale_bytes": rows * 2,
        }
        body = sum(components.values())
        parameters = rows * matrix_columns
        total_bytes += body
        total_parameters += parameters
        matrix_records.append(
            {
                "rows": rows,
                "columns": matrix_columns,
                "parameters": parameters,
                "components": components,
                "body_bytes_excluding_shared_right": body,
            }
        )
    shared = {
        "header_bytes": header_bytes,
        "binary_sign_bytes": math.ceil(rank * columns / 8),
        "FP16_rank_scale_bytes": rank * 2,
    }
    total_bytes += sum(shared.values())
    return {
        "matrices": matrix_records,
        "shared_right": shared,
        "rank": rank,
        "group": group,
        "nonzero_per_group": nonzero_per_group,
        "group_width_bits": group_width,
        "parameters": total_parameters,
        "bytes": total_bytes,
        "bits_per_weight": total_bytes * 8 / total_parameters,
    }
