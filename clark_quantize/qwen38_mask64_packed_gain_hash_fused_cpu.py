"""Load canonical gain-route bodies into the frozen fused CPU ABI."""
from __future__ import annotations

import numpy as np

from .qwen38_fixed_support_mask64_cpu import Mask64Payload
from .qwen38_gain_route_codec import (
    PackedGainHashRouteBody,
    expand_gain_hash_route_compute,
)
from .qwen38_mask64_gain_hash_fused_cpu import (
    Mask64GainHashFusedCPULibrary,
    native_gain_projection_q8,
    native_mask64_gain_hash_fused_gemv_q8,
)


def native_packed_gain_projection_q8(
    activation: np.ndarray,
    activation_scales: np.ndarray,
    route: PackedGainHashRouteBody,
    *,
    library: Mask64GainHashFusedCPULibrary,
) -> np.ndarray:
    if route.gain_bits != 3 or route.input_dim != np.asarray(activation).size:
        raise ValueError("packed gain route and activation dimensions disagree")
    buckets, signs, gain_codes, gain_codebook = expand_gain_hash_route_compute(
        route
    )
    return native_gain_projection_q8(
        activation,
        activation_scales,
        buckets,
        signs,
        gain_codes,
        gain_codebook,
        sketch_dim=route.sketch_dim,
        library=library,
    )


def native_mask64_packed_gain_hash_fused_gemv_q8(
    encoded: Mask64Payload,
    scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation: np.ndarray,
    activation_scales: np.ndarray,
    route: PackedGainHashRouteBody,
    correction_codes: np.ndarray,
    correction_row_scales: np.ndarray,
    *,
    library: Mask64GainHashFusedCPULibrary,
    implementation: str = "scalar",
    trusted: bool = False,
) -> np.ndarray:
    coefficients = np.asarray(correction_codes)
    if (
        route.gain_bits != 3
        or route.input_dim != encoded.columns
        or coefficients.ndim != 2
        or coefficients.shape != (encoded.rows, route.sketch_dim)
    ):
        raise ValueError(
            "packed gain route, mask64 matrix, and correction dimensions disagree"
        )
    buckets, signs, gain_codes, gain_codebook = expand_gain_hash_route_compute(
        route
    )
    return native_mask64_gain_hash_fused_gemv_q8(
        encoded,
        scale_indices,
        row_anchors,
        ratio_codebook,
        activation,
        activation_scales,
        buckets,
        signs,
        gain_codes,
        gain_codebook,
        coefficients,
        correction_row_scales,
        library=library,
        implementation=implementation,
        trusted=trusted,
    )
