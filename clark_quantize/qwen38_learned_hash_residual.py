"""Learned signed-hash residual correction for Qwen3.8-27B.

Clark Hash uses a stateless random sparse projection.  For a model layer the
projection is shared by thousands of output rows, so storing one bucket and
one sign per input coordinate costs almost nothing per original weight.  This
module studies the stronger representation

    W ~= T + A R,

where ``T`` is fixed-support ternary and the one-sparse routing matrix ``R`` is
learned by signed vector quantization of the residual columns.  The mapping is
still cheap at inference: ``R x`` is one signed accumulation per input value.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .qwen38_fixed_support_codec import fixed_support_width_bits
from .qwen38_hash_residual import (
    fixed_support_ternary,
    quantize_hash_coefficients_int4,
)


@dataclass(frozen=True)
class LearnedHashProjection:
    input_dim: int
    sketch_dim: int
    buckets: torch.Tensor
    signs: torch.Tensor


@dataclass(frozen=True)
class LearnedRoutingResult:
    projection: LearnedHashProjection
    coefficients: torch.Tensor
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class LearnedHashResidualFit:
    ternary_codes: torch.Tensor
    ternary_scales: torch.Tensor
    hash_coefficients: torch.Tensor
    projection: LearnedHashProjection
    fp32_objective_history: tuple[float, ...]
    exported_squared_error: float


def _validate_projection(projection: LearnedHashProjection) -> None:
    if (
        projection.input_dim <= 0
        or projection.sketch_dim <= 0
        or projection.buckets.shape != (projection.input_dim,)
        or projection.signs.shape != (projection.input_dim,)
        or projection.buckets.dtype != torch.int64
        or projection.signs.dtype != torch.int8
        or bool((projection.buckets < 0).any())
        or bool((projection.buckets >= projection.sketch_dim).any())
        or not bool(((projection.signs == -1) | (projection.signs == 1)).all())
    ):
        raise ValueError("learned hash projection is invalid")


def learned_sketch_rows(
    values: torch.Tensor,
    projection: LearnedHashProjection,
) -> torch.Tensor:
    """Return ``values @ R.T`` for a one-sparse learned signed routing."""
    _validate_projection(projection)
    if values.ndim != 2 or values.shape[1] != projection.input_dim:
        raise ValueError("learned hash input dimensions disagree")
    source = values.to(torch.float32)
    output = torch.zeros(
        source.shape[0], projection.sketch_dim,
        dtype=torch.float32, device=source.device,
    )
    indices = projection.buckets.to(source.device).expand(source.shape[0], -1)
    signed = source * projection.signs.to(source.device, torch.float32)
    output.scatter_add_(1, indices, signed)
    return output.contiguous()


def expand_learned_hash(
    coefficients: torch.Tensor,
    projection: LearnedHashProjection,
) -> torch.Tensor:
    """Return ``A R`` without materializing the routing matrix."""
    _validate_projection(projection)
    if coefficients.ndim != 2 or coefficients.shape[1] != projection.sketch_dim:
        raise ValueError("learned hash coefficient dimensions disagree")
    buckets = projection.buckets.to(coefficients.device)
    signs = projection.signs.to(coefficients.device, torch.float32)
    return (coefficients.to(torch.float32)[:, buckets] * signs).contiguous()


def fit_learned_hash_coefficients(
    target: torch.Tensor,
    projection: LearnedHashProjection,
) -> torch.Tensor:
    """Exactly fit ``A`` for a frozen one-sparse routing under Frobenius loss."""
    _validate_projection(projection)
    if (
        target.ndim != 2
        or not target.is_floating_point()
        or target.shape[1] != projection.input_dim
    ):
        raise ValueError("learned hash target dimensions disagree")
    source = target.to(torch.float32)
    buckets = projection.buckets.to(source.device)
    signs = projection.signs.to(source.device, torch.float32)
    coefficients = torch.zeros(
        source.shape[0], projection.sketch_dim,
        dtype=torch.float32, device=source.device,
    )
    coefficients.index_add_(1, buckets, source * signs)
    counts = torch.bincount(buckets, minlength=projection.sketch_dim).clamp_min(1)
    coefficients /= counts.to(source.device, torch.float32)[None, :]
    return coefficients.contiguous()


def _assign_signed_clusters(
    points: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dots = points @ prototypes.T
    distances = (
        points.square().sum(dim=1, keepdim=True)
        + prototypes.square().sum(dim=1)[None, :]
        - 2.0 * dots.abs()
    ).clamp_min_(0.0)
    buckets = distances.argmin(dim=1)
    selected_dots = dots.gather(1, buckets[:, None]).squeeze(1)
    signs = torch.where(
        selected_dots >= 0,
        torch.ones_like(buckets, dtype=torch.int8),
        -torch.ones_like(buckets, dtype=torch.int8),
    )
    selected_error = distances.gather(1, buckets[:, None]).squeeze(1)
    return buckets, signs, selected_error


def _farthest_signed_initialization(
    points: torch.Tensor,
    clusters: int,
) -> torch.Tensor:
    if clusters > points.shape[0]:
        raise ValueError("learned hash sketch cannot exceed the input dimension")
    norms = points.square().sum(dim=1)
    selected = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    first = int(norms.argmax().item())
    prototypes = [points[first].clone()]
    selected[first] = True
    nearest = (
        norms + prototypes[0].square().sum() - 2.0 * (points @ prototypes[0]).abs()
    ).clamp_min_(0.0)
    for _ in range(1, clusters):
        candidates = nearest.masked_fill(selected, -1.0)
        index = int(candidates.argmax().item())
        prototypes.append(points[index].clone())
        selected[index] = True
        current = (
            norms + prototypes[-1].square().sum()
            - 2.0 * (points @ prototypes[-1]).abs()
        ).clamp_min_(0.0)
        nearest = torch.minimum(nearest, current)
    return torch.stack(prototypes)


@torch.no_grad()
def learn_signed_hash_routing(
    target: torch.Tensor,
    *,
    sketch_dim: int,
    iterations: int = 6,
    initial_projection: LearnedHashProjection | None = None,
    initial_coefficients: torch.Tensor | None = None,
) -> LearnedRoutingResult:
    """Fit one bucket/sign per residual column with deterministic signed Lloyd."""
    if (
        target.ndim != 2
        or not target.is_floating_point()
        or target.shape[0] == 0
        or target.shape[1] == 0
        or sketch_dim <= 0
        or iterations <= 0
        or sketch_dim > target.shape[1]
    ):
        raise ValueError("invalid learned signed-hash routing problem")
    source = target.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(source).all()):
        raise ValueError("learned signed-hash routing requires finite values")
    points = source.T.contiguous()
    if initial_projection is None or initial_coefficients is None:
        prototypes = _farthest_signed_initialization(points, sketch_dim)
    else:
        _validate_projection(initial_projection)
        if (
            initial_projection.input_dim != source.shape[1]
            or initial_projection.sketch_dim != sketch_dim
            or initial_coefficients.shape != (source.shape[0], sketch_dim)
        ):
            raise ValueError("learned signed-hash initialization dimensions disagree")
        prototypes = initial_coefficients.detach().to(torch.float32).T.contiguous()

    history: list[float] = []
    buckets = signs = None
    for _ in range(iterations):
        buckets, signs, _ = _assign_signed_clusters(points, prototypes)
        signed_points = points * signs.to(torch.float32)[:, None]
        sums = torch.zeros_like(prototypes)
        sums.index_add_(0, buckets, signed_points)
        counts = torch.bincount(buckets, minlength=sketch_dim)
        occupied = counts > 0
        prototypes[occupied] = sums[occupied] / counts[occupied, None].to(torch.float32)
        reconstructed = prototypes[buckets] * signs.to(torch.float32)[:, None]
        objective = float(
            (points - reconstructed).square().sum(dtype=torch.float64).item()
        )
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-6):
            raise RuntimeError("learned signed-hash Lloyd objective increased")
        history.append(objective)
    if buckets is None or signs is None:
        raise AssertionError("learned signed-hash routing did not execute")
    projection = LearnedHashProjection(
        input_dim=source.shape[1],
        sketch_dim=sketch_dim,
        buckets=buckets.to(torch.int64).contiguous(),
        signs=signs.to(torch.int8).contiguous(),
    )
    coefficients = prototypes.T.contiguous()
    return LearnedRoutingResult(
        projection=projection,
        coefficients=coefficients,
        objective_history=tuple(history),
    )


@torch.no_grad()
def learn_hash_ternary_residual(
    weight: torch.Tensor,
    *,
    sketch_dim: int,
    group: int = 128,
    nonzero_per_group: int,
    coordinate_iterations: int = 3,
    routing_iterations: int = 6,
) -> LearnedHashResidualFit:
    """Alternately learn ternary weights and signed residual-column routing."""
    if (
        weight.ndim != 2
        or not weight.is_floating_point()
        or weight.shape[1] % group
        or coordinate_iterations <= 0
    ):
        raise ValueError("invalid learned hash-ternary fitting problem")
    source = weight.detach().to(torch.float32).contiguous()
    coefficients = torch.zeros(source.shape[0], sketch_dim, dtype=torch.float32)
    projection = None
    history: list[float] = []
    codes = scales = ternary = None
    for _ in range(coordinate_iterations):
        correction = (
            torch.zeros_like(source)
            if projection is None
            else expand_learned_hash(coefficients, projection)
        )
        codes, scales, ternary = fixed_support_ternary(
            source - correction,
            group=group,
            nonzero_per_group=nonzero_per_group,
        )
        routing = learn_signed_hash_routing(
            source - ternary,
            sketch_dim=sketch_dim,
            iterations=routing_iterations,
            initial_projection=projection,
            initial_coefficients=coefficients if projection is not None else None,
        )
        projection = routing.projection
        coefficients = routing.coefficients
        objective = float(
            (source - ternary - expand_learned_hash(coefficients, projection))
            .square().sum(dtype=torch.float64).item()
        )
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-5):
            raise RuntimeError("learned hash-ternary coordinate objective increased")
        history.append(objective)
    if projection is None or codes is None or scales is None or ternary is None:
        raise AssertionError("learned hash-ternary fit did not execute")
    _, _, exported_coefficients = quantize_hash_coefficients_int4(coefficients)
    exported_scales = scales.to(torch.float16).to(torch.float32)
    exported_ternary = (
        codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * exported_scales[:, :, None]
    ).reshape_as(source)
    exported = exported_ternary + expand_learned_hash(
        exported_coefficients, projection
    )
    exported_error = float(
        (source - exported).square().sum(dtype=torch.float64).item()
    )
    return LearnedHashResidualFit(
        ternary_codes=codes,
        ternary_scales=scales,
        hash_coefficients=coefficients,
        projection=projection,
        fp32_objective_history=tuple(history),
        exported_squared_error=exported_error,
    )


@torch.no_grad()
def fit_fixed_learned_hash_ternary_residual(
    weight: torch.Tensor,
    projection: LearnedHashProjection,
    *,
    group: int = 128,
    nonzero_per_group: int,
    iterations: int = 3,
) -> LearnedHashResidualFit:
    """Fit row-local ``T`` and ``A`` while keeping learned routing frozen."""
    _validate_projection(projection)
    if (
        weight.ndim != 2
        or not weight.is_floating_point()
        or weight.shape[1] != projection.input_dim
        or weight.shape[1] % group
        or iterations <= 0
    ):
        raise ValueError("invalid fixed learned-hash fitting problem")
    source = weight.detach().to(torch.float32).contiguous()
    coefficients = torch.zeros(
        source.shape[0], projection.sketch_dim, dtype=torch.float32
    )
    history: list[float] = []
    codes = scales = ternary = None
    for _ in range(iterations):
        codes, scales, ternary = fixed_support_ternary(
            source - expand_learned_hash(coefficients, projection),
            group=group,
            nonzero_per_group=nonzero_per_group,
        )
        coefficients = fit_learned_hash_coefficients(source - ternary, projection)
        objective = float(
            (source - ternary - expand_learned_hash(coefficients, projection))
            .square().sum(dtype=torch.float64).item()
        )
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-5):
            raise RuntimeError("fixed learned-hash coordinate objective increased")
        history.append(objective)
    if codes is None or scales is None or ternary is None:
        raise AssertionError("fixed learned-hash fit did not execute")
    _, _, exported_coefficients = quantize_hash_coefficients_int4(coefficients)
    exported_scales = scales.to(torch.float16).to(torch.float32)
    exported_ternary = (
        codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * exported_scales[:, :, None]
    ).reshape_as(source)
    exported_error = float(
        (
            source
            - exported_ternary
            - expand_learned_hash(exported_coefficients, projection)
        ).square().sum(dtype=torch.float64).item()
    )
    return LearnedHashResidualFit(
        ternary_codes=codes,
        ternary_scales=scales,
        hash_coefficients=coefficients,
        projection=projection,
        fp32_objective_history=tuple(history),
        exported_squared_error=exported_error,
    )


def learned_hash_storage_projection(
    *,
    input_dim: int,
    output_rows: int,
    sketch_dim: int,
    nonzero_per_group: int,
    group: int = 128,
    header_bytes: int = 64,
) -> dict[str, int | float]:
    """Count bit-packed routing and every projected component byte."""
    if (
        input_dim <= 0
        or output_rows <= 0
        or sketch_dim <= 0
        or input_dim % group
        or not 0 < nonzero_per_group <= group
        or header_bytes < 0
    ):
        raise ValueError("invalid learned-hash storage dimensions")
    width = fixed_support_width_bits(group, nonzero_per_group)
    ternary_bits = output_rows * (input_dim // group) * width
    ternary_bytes = (ternary_bits + 7) // 8
    scale_index_bits = output_rows * (input_dim // group) * 4
    scale_index_bytes = (scale_index_bits + 7) // 8
    hash_code_bytes = output_rows * ((sketch_dim + 1) // 2)
    hash_scale_bytes = output_rows * 2
    row_anchor_bytes = output_rows * 2
    tensor_codebook_bytes = 16 * 2
    bucket_bits = max(1, math.ceil(math.log2(sketch_dim)))
    routing_bits = input_dim * (bucket_bits + 1)
    routing_bytes = (routing_bits + 7) // 8
    total = (
        ternary_bytes
        + scale_index_bytes
        + hash_code_bytes
        + hash_scale_bytes
        + row_anchor_bytes
        + tensor_codebook_bytes
        + routing_bytes
        + header_bytes
    )
    parameters = input_dim * output_rows
    return {
        "ternary_code_bytes": ternary_bytes,
        "scale_index_bytes": scale_index_bytes,
        "hash_code_bytes": hash_code_bytes,
        "hash_scale_bytes": hash_scale_bytes,
        "row_anchor_bytes": row_anchor_bytes,
        "tensor_codebook_bytes": tensor_codebook_bytes,
        "routing_bucket_bits": bucket_bits,
        "routing_bytes": routing_bytes,
        "header_bytes": header_bytes,
        "total_bytes": total,
        "bits_per_weight": total * 8 / parameters,
    }
