"""Clark-Hash-assisted sparse ternary residual research for Qwen3.8-27B.

This module studies the structured layer approximation

    W ~= T + A R,

where R is the deterministic sparse signed projection used by Clark Hash, A is
a small learned coefficient matrix, and T is a rate-constrained groupwise
ternary residual.  It is an approximation/output codec, not lossless recovery
of an arbitrary dense matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_REPETITION = 0xD1B54A32D192ED03


def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


@dataclass(frozen=True)
class SparseJLProjection:
    input_dim: int
    sketch_dim: int
    hashes_per_input: int
    seed: int
    buckets: torch.Tensor
    signs: torch.Tensor


@dataclass(frozen=True)
class HashResidualFit:
    ternary_codes: torch.Tensor
    ternary_scales: torch.Tensor
    hash_coefficients: torch.Tensor
    projection: SparseJLProjection
    fp32_objective_history: tuple[float, ...]
    exported_squared_error: float


def sparse_jl_projection(
    input_dim: int,
    sketch_dim: int,
    *,
    hashes_per_input: int = 4,
    seed: int = 0x5EEDCAFE12345678,
    device: torch.device | str = "cpu",
) -> SparseJLProjection:
    """Reproduce Clark Hash bucket/sign generation exactly."""
    if input_dim <= 0 or sketch_dim <= 0 or hashes_per_input <= 0 or seed < 0:
        raise ValueError("sparse JL dimensions, repetitions, and seed must be positive")
    buckets = torch.empty(hashes_per_input, input_dim, dtype=torch.int64)
    signs = torch.empty(hashes_per_input, input_dim, dtype=torch.float32)
    for repetition in range(hashes_per_input):
        repetition_key = (repetition * _REPETITION) & _MASK64
        for dimension in range(input_dim):
            key = seed ^ ((dimension * _GOLDEN) & _MASK64) ^ repetition_key
            hashed = _splitmix64(key)
            buckets[repetition, dimension] = hashed % sketch_dim
            signs[repetition, dimension] = 1.0 if hashed >> 63 == 0 else -1.0
    return SparseJLProjection(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        hashes_per_input=hashes_per_input,
        seed=seed,
        buckets=buckets.to(device),
        signs=signs.to(device),
    )


def sketch_rows(values: torch.Tensor, projection: SparseJLProjection) -> torch.Tensor:
    """Return ``values @ R.T`` for Clark Hash's implicit sparse R."""
    if values.ndim != 2 or values.shape[1] != projection.input_dim:
        raise ValueError("row matrix and sparse JL input dimensions disagree")
    output = torch.zeros(
        values.shape[0],
        projection.sketch_dim,
        device=values.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(projection.hashes_per_input)
    source = values.to(torch.float32)
    for repetition in range(projection.hashes_per_input):
        indices = projection.buckets[repetition].expand(values.shape[0], -1)
        signed = source * projection.signs[repetition] * scale
        output.scatter_add_(1, indices, signed)
    return output.contiguous()


def expand_sketch_rows(
    coefficients: torch.Tensor, projection: SparseJLProjection
) -> torch.Tensor:
    """Return ``coefficients @ R`` without materializing R."""
    if coefficients.ndim != 2 or coefficients.shape[1] != projection.sketch_dim:
        raise ValueError("coefficient matrix and sparse JL sketch dimensions disagree")
    scale = 1.0 / math.sqrt(projection.hashes_per_input)
    output = torch.zeros(
        coefficients.shape[0],
        projection.input_dim,
        device=coefficients.device,
        dtype=torch.float32,
    )
    source = coefficients.to(torch.float32)
    for repetition in range(projection.hashes_per_input):
        gathered = source[:, projection.buckets[repetition]]
        output += gathered * projection.signs[repetition] * scale
    return output.contiguous()


def projection_gram(projection: SparseJLProjection) -> torch.Tensor:
    """Construct the exact ``R R.T`` Gram matrix in O(d s^2)."""
    size = projection.sketch_dim
    flattened = torch.zeros(size * size, device=projection.buckets.device)
    scale = 1.0 / projection.hashes_per_input
    for left in range(projection.hashes_per_input):
        for right in range(projection.hashes_per_input):
            indices = (
                projection.buckets[left] * size + projection.buckets[right]
            )
            values = projection.signs[left] * projection.signs[right] * scale
            flattened.scatter_add_(0, indices, values)
    return flattened.reshape(size, size).contiguous()


def least_squares_hash_coefficients(
    target: torch.Tensor,
    projection: SparseJLProjection,
    *,
    relative_ridge: float = 1e-6,
) -> torch.Tensor:
    """Solve ``argmin_A ||target - A R||_F^2`` with a tiny frozen ridge."""
    if relative_ridge < 0:
        raise ValueError("relative ridge must be nonnegative")
    right = sketch_rows(target, projection)
    gram = projection_gram(projection)
    ridge = relative_ridge * float(torch.diagonal(gram).mean().item())
    regularized = gram + torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    ) * ridge
    return torch.linalg.solve(regularized, right.T).T.contiguous()


def fixed_support_ternary(
    values: torch.Tensor,
    *,
    group: int = 128,
    nonzero_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact best groupwise ternary fit at a fixed support size."""
    if values.ndim != 2 or group <= 0 or not 0 <= nonzero_per_group <= group:
        raise ValueError("invalid fixed-support ternary dimensions")
    rows, columns = values.shape
    pad = (-columns) % group
    source = values.to(torch.float32)
    if pad:
        source = torch.nn.functional.pad(source, (0, pad))
    grouped = source.reshape(rows, -1, group)
    if nonzero_per_group == 0:
        codes = torch.zeros_like(grouped, dtype=torch.int8)
        scales = torch.zeros(rows, grouped.shape[1], device=values.device)
    else:
        _, indices = torch.topk(
            grouped.abs(), nonzero_per_group, dim=-1, largest=True, sorted=False
        )
        codes = torch.zeros_like(grouped, dtype=torch.int8)
        selected = grouped.gather(-1, indices)
        codes.scatter_(-1, indices, torch.sign(selected).to(torch.int8))
        scales = selected.abs().mean(dim=-1)
    reconstructed = codes.to(torch.float32) * scales[:, :, None]
    return (
        codes.reshape(rows, -1)[:, :columns].contiguous(),
        scales.contiguous(),
        reconstructed.reshape(rows, -1)[:, :columns].contiguous(),
    )


def quantize_hash_coefficients_int4(
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply signed per-row int4 quantization with levels -7 through +7."""
    if coefficients.ndim != 2 or not coefficients.is_floating_point():
        raise ValueError("hash coefficients must be a floating-point matrix")
    scales = (coefficients.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
    codes = torch.round(coefficients / scales[:, None]).clamp(-7, 7).to(torch.int8)
    reconstructed = codes.to(torch.float32) * scales.to(torch.float16).to(torch.float32)[:, None]
    return codes.contiguous(), scales.to(torch.float16).contiguous(), reconstructed.contiguous()


@torch.no_grad()
def fit_hash_ternary_residual(
    weight: torch.Tensor,
    *,
    sketch_dim: int,
    hashes_per_input: int = 4,
    seed: int = 0x5EEDCAFE12345678,
    group: int = 128,
    nonzero_per_group: int,
    iterations: int = 4,
    relative_ridge: float = 1e-6,
) -> HashResidualFit:
    """Alternately fit a sparse ternary residual and fixed-hash correction."""
    if weight.ndim != 2 or not weight.is_floating_point() or iterations <= 0:
        raise ValueError("hash-residual fitting expects a floating matrix and iterations")
    source = weight.detach().to(torch.float32)
    if source.shape[1] % group:
        raise ValueError("hash-residual fitting requires groups to divide input width")
    projection = sparse_jl_projection(
        source.shape[1],
        sketch_dim,
        hashes_per_input=hashes_per_input,
        seed=seed,
        device=source.device,
    )
    coefficients = torch.zeros(source.shape[0], sketch_dim, device=source.device)
    history = []
    codes = scales = ternary = None
    for _ in range(iterations):
        residual = source - expand_sketch_rows(coefficients, projection)
        codes, scales, ternary = fixed_support_ternary(
            residual, group=group, nonzero_per_group=nonzero_per_group
        )
        coefficients = least_squares_hash_coefficients(
            source - ternary,
            projection,
            relative_ridge=relative_ridge,
        )
        reconstructed = ternary + expand_sketch_rows(coefficients, projection)
        objective = float((source - reconstructed).square().sum(dtype=torch.float64).item())
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-5):
            raise RuntimeError("hash-residual coordinate fit increased its objective")
        history.append(objective)
    if codes is None or scales is None or ternary is None:
        raise AssertionError("hash-residual fit did not execute")
    _, _, exported_coefficients = quantize_hash_coefficients_int4(coefficients)
    exported_scales = scales.to(torch.float16).to(torch.float32)
    exported_ternary = (
        codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * exported_scales[:, :, None]
    ).reshape_as(source)
    exported = exported_ternary + expand_sketch_rows(exported_coefficients, projection)
    exported_error = float((source - exported).square().sum(dtype=torch.float64).item())
    return HashResidualFit(
        ternary_codes=codes,
        ternary_scales=scales,
        hash_coefficients=coefficients,
        projection=projection,
        fp32_objective_history=tuple(history),
        exported_squared_error=exported_error,
    )


def symmetric_ternary_entropy_bits(zero_fraction: float) -> float:
    if not 0 <= zero_fraction <= 1:
        raise ValueError("zero fraction must lie in [0, 1]")
    side = (1.0 - zero_fraction) / 2.0
    return -sum(value * math.log2(value) for value in (side, zero_fraction, side) if value)


def projected_hash_residual_bits_per_weight(
    *,
    input_dim: int,
    sketch_dim: int,
    nonzero_per_group: int,
    group: int = 128,
    hash_bits: int = 4,
    scale_index_bits: int = 4,
) -> float:
    """Project codes, hash coefficients, one row scale, and scale indices."""
    if input_dim <= 0 or sketch_dim <= 0 or not 0 <= nonzero_per_group <= group:
        raise ValueError("invalid hash-residual storage dimensions")
    zero_fraction = 1.0 - nonzero_per_group / group
    return (
        symmetric_ternary_entropy_bits(zero_fraction)
        + hash_bits * sketch_dim / input_dim
        + scale_index_bits / group
        + 16 / input_dim
        + 16 / input_dim
    )
