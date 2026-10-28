"""One-call mask64 GEMV plus gain-routed Clark-Hash INT4 correction."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .qwen38_fixed_support_mask64_cpu import (
    Mask64CPULibrary,
    Mask64Payload,
    pack_scale_indices4,
)
from .qwen38_hash_residual_cpu import pack_signed_int4_rows


_ERRORS = {
    1: "invalid fused mask64/gain-hash argument",
    2: "unsupported fused mask64/gain-hash shape",
    3: "invalid fused mask64/gain-hash code or route",
    4: "non-finite fused mask64/gain-hash scale",
    5: "requested fused mask64/gain-hash CPU ISA is unavailable",
}


class Mask64GainHashFusedCPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.mask = Mask64CPULibrary(self.path)
        self.library = self.mask.library
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)
        projection = (
            i8p, f32p, u16p, i8p, u8p, f32p,
            ctypes.c_uint64, ctypes.c_uint32, f32p,
        )
        function = self.library.cq3_mghf_projection_q8_scalar_trusted
        function.argtypes = projection
        function.restype = ctypes.c_int
        safe = (
            u8p, ctypes.c_uint64, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, u16p, i8p, u8p, f32p,
            ctypes.c_uint32, u8p, u16p, f32p, f32p,
        )
        trusted = (
            u8p, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, u16p, i8p, u8p, f32p,
            ctypes.c_uint32, u8p, u16p, f32p, f32p,
        )
        for implementation in ("scalar", "avx2_bmi2"):
            function = getattr(
                self.library,
                f"cq3_mghf_gemv_q8_row_scale4_{implementation}",
            )
            function.argtypes = safe
            function.restype = ctypes.c_int
            trusted_function = getattr(
                self.library,
                f"cq3_mghf_gemv_q8_row_scale4_{implementation}_trusted",
            )
            trusted_function.argtypes = trusted
            trusted_function.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        return self.mask.features()


def load_mask64_gain_hash_fused_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> Mask64GainHashFusedCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_MASK64_GAIN_HASH_FUSED_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_mask64_gain_hash_fused_cpu.so"),
            Path(__file__).with_name("_qwen38_mask64_gain_hash_fused_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return Mask64GainHashFusedCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(
            _ERRORS.get(status, f"native fused mask64/gain-hash error {status}")
        )


def _gain_route(
    activation: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    gain_codes: np.ndarray,
    gain_codebook: np.ndarray,
    *,
    sketch_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(activation, dtype=np.int8)
    route = np.ascontiguousarray(buckets, dtype=np.uint16)
    directions = np.ascontiguousarray(signs, dtype=np.int8)
    gains = np.ascontiguousarray(gain_codes, dtype=np.uint8)
    codebook = np.ascontiguousarray(gain_codebook, dtype=np.float16).astype(
        np.float32
    )
    if values.ndim != 1 or values.size == 0 or values.size % 128:
        raise ValueError("fused gain-hash activation must be group128 aligned")
    if route.shape != values.shape or directions.shape != values.shape:
        raise ValueError("fused gain-hash route dimensions disagree")
    if gains.shape != values.shape or codebook.shape != (8,):
        raise ValueError("fused gain-hash gain dimensions disagree")
    if sketch_dim <= 0 or sketch_dim > 65535 or np.any(route >= sketch_dim):
        raise ValueError("fused gain-hash buckets disagree")
    if not np.all((directions == -1) | (directions == 1)):
        raise ValueError("fused gain-hash signs must be -1 or +1")
    if np.any(gains >= 8):
        raise ValueError("fused gain-hash codes must be in [0, 7]")
    if not np.all(np.isfinite(codebook)) or np.any(codebook < 0):
        raise ValueError("fused gain-hash codebook must be finite and nonnegative")
    return values, route, directions, gains, codebook


def native_gain_projection_q8(
    activation: np.ndarray,
    activation_scales: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    gain_codes: np.ndarray,
    gain_codebook: np.ndarray,
    *,
    sketch_dim: int,
    library: Mask64GainHashFusedCPULibrary,
) -> np.ndarray:
    values, route, directions, gains, codebook = _gain_route(
        activation,
        buckets,
        signs,
        gain_codes,
        gain_codebook,
        sketch_dim=sketch_dim,
    )
    scales = np.ascontiguousarray(activation_scales, dtype=np.float32)
    if scales.shape != (values.size // 128,):
        raise ValueError("fused gain-hash activation scales disagree")
    output = np.empty(sketch_dim, dtype=np.float32)
    _check_status(library.library.cq3_mghf_projection_q8_scalar_trusted(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        gains.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        sketch_dim,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ))
    return output


def native_mask64_gain_hash_fused_gemv_q8(
    encoded: Mask64Payload,
    scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation: np.ndarray,
    activation_scales: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    gain_codes: np.ndarray,
    gain_codebook: np.ndarray,
    correction_codes: np.ndarray,
    correction_row_scales: np.ndarray,
    *,
    library: Mask64GainHashFusedCPULibrary,
    implementation: str = "scalar",
    trusted: bool = False,
) -> np.ndarray:
    if implementation not in {"scalar", "avx2_bmi2"}:
        raise ValueError("unknown fused mask64/gain-hash GEMV implementation")
    groups = encoded.columns // 128
    indices = np.asarray(scale_indices)
    if indices.shape != (encoded.rows, groups):
        raise ValueError("fused mask64 scale-index dimensions disagree")
    packed_indices = pack_scale_indices4(indices)
    anchors = np.ascontiguousarray(row_anchors, dtype=np.float16)
    ratios = np.ascontiguousarray(ratio_codebook, dtype=np.float16)
    coefficients = np.asarray(correction_codes)
    if coefficients.ndim != 2 or coefficients.shape[0] != encoded.rows:
        raise ValueError("fused gain-hash coefficient dimensions disagree")
    sketch_dim = coefficients.shape[1]
    values, route, directions, gains, codebook = _gain_route(
        activation,
        buckets,
        signs,
        gain_codes,
        gain_codebook,
        sketch_dim=sketch_dim,
    )
    activation_scale_values = np.ascontiguousarray(
        activation_scales, dtype=np.float32
    )
    coefficient_scales = np.ascontiguousarray(
        correction_row_scales, dtype=np.float16
    )
    if anchors.shape != (encoded.rows,) or ratios.shape != (16,):
        raise ValueError("fused mask64 row-scale metadata dimensions disagree")
    if values.shape != (encoded.columns,) or activation_scale_values.shape != (groups,):
        raise ValueError("fused mask64 activation dimensions disagree")
    if coefficient_scales.shape != (encoded.rows,):
        raise ValueError("fused gain-hash row scales disagree")
    packed_coefficients = pack_signed_int4_rows(coefficients)
    activation_sums = values.reshape(groups, 128).sum(axis=1, dtype=np.int32)
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    scratch = np.empty(sketch_dim, dtype=np.float32)
    output = np.empty(encoded.rows, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_mghf_gemv_q8_row_scale4_{implementation}{suffix}",
    )
    common = (
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        packed_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        anchors.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ratios.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        activation_scale_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        activation_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        encoded.rows,
        encoded.columns,
        route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        gains.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sketch_dim,
        packed_coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        coefficient_scales.view(np.uint16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        ),
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    args = common if trusted else (common[0], payload.size, *common[1:])
    _check_status(function(*args))
    return output
