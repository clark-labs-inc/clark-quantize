"""Gain-aware signed-hash residual approximation for Qwen3.8-27B.

Clark Hash keeps a compact norm side channel when direction alone is not
enough.  For a model matrix, an analogous low-rate extension is to give every
input coordinate a small positive gain in addition to its learned bucket and
sign:

    W ~= T + A R_g

where column ``i`` of ``R_g`` contains one value ``sign_i * gain_i``.  Gain
indices are shared across all output rows, so even three bits per input
coordinate cost very little per original model weight.  Runtime remains one
scaled signed accumulation per input coordinate followed by a tiny INT4 GEMV.
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
from .qwen38_learned_hash_residual import learn_signed_hash_routing


@dataclass(frozen=True)
class GainHashProjection:
    input_dim: int
    sketch_dim: int
    gain_bits: int
    buckets: torch.Tensor
    signs: torch.Tensor
    gain_codes: torch.Tensor
    gain_codebook: torch.Tensor


@dataclass(frozen=True)
class GainRoutingResult:
    projection: GainHashProjection
    coefficients: torch.Tensor
    objective_history: tuple[float, ...]
    continuous_squared_error: float
    exported_squared_error: float


@dataclass(frozen=True)
class GainHashResidualFit:
    ternary_codes: torch.Tensor
    ternary_scales: torch.Tensor
    hash_coefficients: torch.Tensor
    projection: GainHashProjection
    fp32_objective_history: tuple[float, ...]
    exported_squared_error: float


def _validate_projection(projection: GainHashProjection) -> None:
    levels = 1 << projection.gain_bits
    if (
        projection.input_dim <= 0
        or projection.sketch_dim <= 0
        or not 1 <= projection.gain_bits <= 8
        or projection.buckets.shape != (projection.input_dim,)
        or projection.signs.shape != (projection.input_dim,)
        or projection.gain_codes.shape != (projection.input_dim,)
        or projection.gain_codebook.shape != (levels,)
        or projection.buckets.dtype != torch.int64
        or projection.signs.dtype != torch.int8
        or projection.gain_codes.dtype != torch.uint8
        or not projection.gain_codebook.is_floating_point()
        or bool((projection.buckets < 0).any())
        or bool((projection.buckets >= projection.sketch_dim).any())
        or not bool(((projection.signs == -1) | (projection.signs == 1)).all())
        or bool((projection.gain_codes >= levels).any())
        or bool((projection.gain_codebook < 0).any())
        or not bool(torch.isfinite(projection.gain_codebook).all())
    ):
        raise ValueError("gain-hash projection is invalid")


def gain_values(projection: GainHashProjection) -> torch.Tensor:
    _validate_projection(projection)
    return projection.gain_codebook.to(torch.float32)[
        projection.gain_codes.to(torch.int64)
    ].contiguous()


def gain_hash_sketch_rows(
    values: torch.Tensor,
    projection: GainHashProjection,
) -> torch.Tensor:
    """Return ``values @ R_g.T`` without materializing ``R_g``."""
    _validate_projection(projection)
    if values.ndim != 2 or values.shape[1] != projection.input_dim:
        raise ValueError("gain-hash input dimensions disagree")
    source = values.to(torch.float32)
    output = torch.zeros(
        source.shape[0], projection.sketch_dim,
        dtype=torch.float32, device=source.device,
    )
    indices = projection.buckets.to(source.device).expand(source.shape[0], -1)
    multipliers = (
        projection.signs.to(source.device, torch.float32)
        * gain_values(projection).to(source.device)
    )
    output.scatter_add_(1, indices, source * multipliers)
    return output.contiguous()


def expand_gain_hash(
    coefficients: torch.Tensor,
    projection: GainHashProjection,
) -> torch.Tensor:
    """Return ``A R_g`` without materializing the routing matrix."""
    _validate_projection(projection)
    if coefficients.ndim != 2 or coefficients.shape[1] != projection.sketch_dim:
        raise ValueError("gain-hash coefficient dimensions disagree")
    buckets = projection.buckets.to(coefficients.device)
    multipliers = (
        projection.signs.to(coefficients.device, torch.float32)
        * gain_values(projection).to(coefficients.device)
    )
    return (coefficients.to(torch.float32)[:, buckets] * multipliers).contiguous()


def fit_gain_hash_coefficients(
    target: torch.Tensor,
    projection: GainHashProjection,
) -> torch.Tensor:
    """Exactly fit ``A`` for a frozen one-sparse signed/gain route."""
    _validate_projection(projection)
    if (
        target.ndim != 2
        or not target.is_floating_point()
        or target.shape[1] != projection.input_dim
    ):
        raise ValueError("gain-hash target dimensions disagree")
    source = target.to(torch.float32)
    buckets = projection.buckets.to(source.device)
    multipliers = (
        projection.signs.to(source.device, torch.float32)
        * gain_values(projection).to(source.device)
    )
    coefficients = torch.zeros(
        source.shape[0], projection.sketch_dim,
        dtype=torch.float32, device=source.device,
    )
    coefficients.index_add_(1, buckets, source * multipliers)
    denominator = torch.zeros(
        projection.sketch_dim, dtype=torch.float32, device=source.device
    )
    denominator.index_add_(0, buckets, multipliers.square())
    coefficients /= denominator.clamp_min(1e-20)[None, :]
    return coefficients.contiguous()


def _assign_scaled_signed_clusters(
    points: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prototype_norms = prototypes.square().sum(dim=1).clamp_min(1e-20)
    dots = points @ prototypes.T
    errors = (
        points.square().sum(dim=1, keepdim=True)
        - dots.square() / prototype_norms[None, :]
    ).clamp_min_(0.0)
    buckets = errors.argmin(dim=1)
    selected_dots = dots.gather(1, buckets[:, None]).squeeze(1)
    selected_norms = prototype_norms[buckets]
    signs = torch.where(
        selected_dots >= 0,
        torch.ones_like(buckets, dtype=torch.int8),
        -torch.ones_like(buckets, dtype=torch.int8),
    )
    gains = selected_dots.abs() / selected_norms
    return buckets, signs, gains


def _update_scaled_prototypes(
    points: torch.Tensor,
    prototypes: torch.Tensor,
    buckets: torch.Tensor,
    signs: torch.Tensor,
    gains: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    multiplier = signs.to(torch.float32) * gains
    numerator = torch.zeros_like(prototypes)
    numerator.index_add_(0, buckets, points * multiplier[:, None])
    denominator = torch.zeros(
        prototypes.shape[0], dtype=torch.float32, device=points.device
    )
    denominator.index_add_(0, buckets, gains.square())
    occupied = denominator > 1e-20
    updated = prototypes.clone()
    updated[occupied] = numerator[occupied] / denominator[occupied, None]

    # Resolve the per-bucket scale ambiguity while preserving reconstruction:
    # each occupied bucket's gains have RMS one, and its scale lives in A.
    counts = torch.bincount(buckets, minlength=prototypes.shape[0]).clamp_min(1)
    gain_energy = torch.zeros_like(denominator)
    gain_energy.index_add_(0, buckets, gains.square())
    bucket_scale = torch.sqrt(
        gain_energy / counts.to(torch.float32)
    ).clamp_min(1e-12)
    updated[occupied] *= bucket_scale[occupied, None]
    normalized_gains = gains / bucket_scale[buckets]
    return updated.contiguous(), normalized_gains.contiguous()


def _quantize_positive_codebook(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    bits: int,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or not 1 <= bits <= 8
        or iterations <= 0
        or bool((values < 0).any())
        or bool((weights <= 0).any())
    ):
        raise ValueError("invalid positive gain quantization problem")
    levels = 1 << bits
    quantiles = (
        torch.arange(levels, dtype=torch.float32, device=values.device) + 0.5
    ) / levels
    codebook = torch.quantile(values, quantiles).contiguous()
    for _ in range(iterations):
        distances = (values[:, None] - codebook[None, :]).square()
        codes = distances.argmin(dim=1)
        numerator = torch.zeros_like(codebook)
        denominator = torch.zeros_like(codebook)
        numerator.index_add_(0, codes, values * weights)
        denominator.index_add_(0, codes, weights)
        occupied = denominator > 0
        codebook[occupied] = numerator[occupied] / denominator[occupied]
        codebook, _ = torch.sort(codebook)
    exported_codebook = codebook.to(torch.float16).contiguous()
    exported_distances = (
        values[:, None] - exported_codebook.to(torch.float32)[None, :]
    ).square()
    exported_codes = exported_distances.argmin(dim=1).to(torch.uint8).contiguous()
    reconstructed = exported_codebook.to(torch.float32)[
        exported_codes.to(torch.int64)
    ].contiguous()
    return exported_codes, exported_codebook, reconstructed


@torch.no_grad()
def learn_gain_hash_routing(
    target: torch.Tensor,
    *,
    sketch_dim: int,
    gain_bits: int = 3,
    iterations: int = 6,
) -> GainRoutingResult:
    """Learn signed one-sparse routing with a quantized per-input gain."""
    if (
        target.ndim != 2
        or not target.is_floating_point()
        or target.shape[0] == 0
        or target.shape[1] == 0
        or sketch_dim <= 0
        or sketch_dim > target.shape[1]
        or not 1 <= gain_bits <= 8
        or iterations <= 0
    ):
        raise ValueError("invalid gain-hash routing problem")
    source = target.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(source).all()):
        raise ValueError("gain-hash routing requires finite values")
    points = source.T.contiguous()
    initialization = learn_signed_hash_routing(
        source, sketch_dim=sketch_dim, iterations=1
    )
    prototypes = initialization.coefficients.T.contiguous()
    history: list[float] = []
    buckets = signs = gains = None
    for _ in range(iterations):
        buckets, signs, gains = _assign_scaled_signed_clusters(points, prototypes)
        prototypes, gains = _update_scaled_prototypes(
            points, prototypes, buckets, signs, gains
        )
        reconstructed = (
            prototypes[buckets]
            * signs.to(torch.float32)[:, None]
            * gains[:, None]
        )
        objective = float(
            (points - reconstructed).square().sum(dtype=torch.float64).item()
        )
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-6):
            raise RuntimeError("gain-hash coordinate objective increased")
        history.append(objective)
    if buckets is None or signs is None or gains is None:
        raise AssertionError("gain-hash routing did not execute")

    prototype_norms = prototypes.square().sum(dim=1).clamp_min(1e-20)
    gain_weights = prototype_norms[buckets]
    gain_codes, gain_codebook, _ = _quantize_positive_codebook(
        gains, gain_weights, bits=gain_bits
    )
    projection = GainHashProjection(
        input_dim=source.shape[1],
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
        buckets=buckets.to(torch.int64).contiguous(),
        signs=signs.to(torch.int8).contiguous(),
        gain_codes=gain_codes,
        gain_codebook=gain_codebook,
    )
    coefficients = fit_gain_hash_coefficients(source, projection)
    continuous_error = history[-1]
    _, _, exported_coefficients = quantize_hash_coefficients_int4(coefficients)
    exported_error = float(
        (source - expand_gain_hash(exported_coefficients, projection))
        .square().sum(dtype=torch.float64).item()
    )
    return GainRoutingResult(
        projection=projection,
        coefficients=coefficients,
        objective_history=tuple(history),
        continuous_squared_error=continuous_error,
        exported_squared_error=exported_error,
    )


@torch.no_grad()
def fit_fixed_gain_hash_ternary_residual(
    weight: torch.Tensor,
    projection: GainHashProjection,
    *,
    group: int = 128,
    nonzero_per_group: int,
    iterations: int = 3,
) -> GainHashResidualFit:
    """Fit row-local ternary ``T`` and INT4 ``A`` with routing frozen."""
    _validate_projection(projection)
    if (
        weight.ndim != 2
        or not weight.is_floating_point()
        or weight.shape[1] != projection.input_dim
        or weight.shape[1] % group
        or group <= 0
        or not 0 < nonzero_per_group <= group
        or iterations <= 0
    ):
        raise ValueError("invalid fixed gain-hash fitting problem")
    source = weight.detach().to(torch.float32).contiguous()
    coefficients = torch.zeros(
        source.shape[0], projection.sketch_dim,
        dtype=torch.float32, device=source.device,
    )
    history: list[float] = []
    codes = scales = ternary = None
    for _ in range(iterations):
        codes, scales, ternary = fixed_support_ternary(
            source - expand_gain_hash(coefficients, projection),
            group=group,
            nonzero_per_group=nonzero_per_group,
        )
        coefficients = fit_gain_hash_coefficients(source - ternary, projection)
        objective = float(
            (source - ternary - expand_gain_hash(coefficients, projection))
            .square().sum(dtype=torch.float64).item()
        )
        if history and objective > history[-1] + max(1e-8, history[-1] * 2e-5):
            raise RuntimeError("fixed gain-hash objective increased")
        history.append(objective)
    if codes is None or scales is None or ternary is None:
        raise AssertionError("fixed gain-hash fit did not execute")
    _, _, exported_coefficients = quantize_hash_coefficients_int4(coefficients)
    exported_scales = scales.to(torch.float16).to(torch.float32)
    exported_ternary = (
        codes.to(torch.float32)
        .reshape(source.shape[0], -1, group)
        * exported_scales[:, :, None]
    ).reshape_as(source)
    exported_error = float(
        (
            source
            - exported_ternary
            - expand_gain_hash(exported_coefficients, projection)
        ).square().sum(dtype=torch.float64).item()
    )
    return GainHashResidualFit(
        ternary_codes=codes,
        ternary_scales=scales,
        hash_coefficients=coefficients,
        projection=projection,
        fp32_objective_history=tuple(history),
        exported_squared_error=exported_error,
    )


def gain_hash_storage_projection(
    *,
    input_dim: int,
    output_rows: int,
    sketch_dim: int,
    gain_bits: int,
    nonzero_per_group: int,
    group: int = 128,
    header_bytes: int = 64,
) -> dict[str, int | float]:
    """Count routing, gain indices/codebook, and every component byte."""
    if (
        input_dim <= 0
        or output_rows <= 0
        or sketch_dim <= 0
        or input_dim % group
        or not 1 <= gain_bits <= 8
        or not 0 < nonzero_per_group <= group
        or header_bytes < 0
    ):
        raise ValueError("invalid gain-hash storage dimensions")
    groups = math.ceil(input_dim / group)
    width = fixed_support_width_bits(group, nonzero_per_group)
    ternary_bits = output_rows * groups * width
    ternary_bytes = (ternary_bits + 7) // 8
    scale_index_bytes = (output_rows * groups * 4 + 7) // 8
    hash_code_bytes = output_rows * ((sketch_dim + 1) // 2)
    bucket_bits = max(1, math.ceil(math.log2(sketch_dim)))
    route_bytes = (input_dim * (bucket_bits + 1 + gain_bits) + 7) // 8
    components = {
        "ternary_code_bytes": ternary_bytes,
        "scale_index_bytes": scale_index_bytes,
        "hash_code_bytes": hash_code_bytes,
        "hash_scale_bytes": output_rows * 2,
        "row_anchor_bytes": output_rows * 2,
        "tensor_ratio_codebook_bytes": 16 * 2,
        "routing_and_gain_index_bytes": route_bytes,
        "gain_codebook_bytes": (1 << gain_bits) * 2,
        "header_bytes": header_bytes,
    }
    total = sum(components.values())
    return {
        **components,
        "routing_bucket_bits": bucket_bits,
        "gain_bits": gain_bits,
        "total_bytes": total,
        "bits_per_weight": total * 8 / (input_dim * output_rows),
    }
