"""One-call CPU execution of half-support tail GEMV plus hash correction."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .qwen38_fixed_support_mask64_cpu import pack_scale_indices4
from .qwen38_half_support_tail_cpu import (
    HalfSupportTailCPULibrary,
    HalfSupportTailPayload,
)
from .qwen38_hash_residual_cpu import pack_signed_int4_rows


_ERRORS = {
    1: "invalid fused half-support/hash argument",
    2: "unsupported fused half-support/hash shape",
    3: "invalid fused half-support/hash code or route",
    4: "non-finite fused half-support/hash scale",
    5: "requested fused half-support/hash CPU ISA is unavailable",
}


class HalfSupportTailHashFusedCPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.tail = HalfSupportTailCPULibrary(self.path)
        self.library = self.tail.library
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)
        safe = (
            u8p, ctypes.c_uint64, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, u16p, i8p, ctypes.c_uint32,
            u8p, u16p, f32p, f32p, f32p,
        )
        trusted = (
            u8p, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, u16p, i8p, ctypes.c_uint32,
            u8p, u16p, f32p, f32p, f32p,
        )
        for implementation in ("scalar", "avx2_bmi2"):
            function = getattr(
                self.library,
                f"cq3_htf_gemv_q8_row_scale4_{implementation}",
            )
            function.argtypes = safe
            function.restype = ctypes.c_int
            trusted_function = getattr(
                self.library,
                f"cq3_htf_gemv_q8_row_scale4_{implementation}_trusted",
            )
            trusted_function.argtypes = trusted
            trusted_function.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        return self.tail.features()


def load_half_support_tail_hash_fused_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> HalfSupportTailHashFusedCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_HALF_SUPPORT_TAIL_HASH_FUSED_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_half_support_tail_hash_fused_cpu.so"),
            Path(__file__).with_name("_qwen38_half_support_tail_hash_fused_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return HalfSupportTailHashFusedCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(
            _ERRORS.get(status, f"native fused half-support/hash error {status}")
        )


def _activation_sums(values: np.ndarray) -> np.ndarray:
    groups = (values.size + 127) // 128
    output = np.empty(groups, dtype=np.int32)
    for group in range(groups):
        start = group * 128
        output[group] = values[start : start + 128].sum(dtype=np.int32)
    return output


def native_half_support_tail_hash_fused_gemv_q8(
    encoded: HalfSupportTailPayload,
    scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation: np.ndarray,
    activation_scales: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    correction_codes: np.ndarray,
    correction_row_scales: np.ndarray,
    *,
    library: HalfSupportTailHashFusedCPULibrary,
    implementation: str = "scalar",
    trusted: bool = False,
) -> np.ndarray:
    if implementation not in {"scalar", "avx2_bmi2"}:
        raise ValueError("unknown fused half-support/hash GEMV implementation")
    if encoded.group_size != 128:
        raise ValueError("fused half-support payload metadata is inconsistent")
    groups = (encoded.columns + 127) // 128
    indices = np.asarray(scale_indices)
    if indices.shape != (encoded.rows, groups):
        raise ValueError("fused half-support scale-index dimensions disagree")
    packed_indices = pack_scale_indices4(indices)
    anchors = np.ascontiguousarray(row_anchors, dtype=np.float16)
    ratios = np.ascontiguousarray(ratio_codebook, dtype=np.float16)
    values = np.ascontiguousarray(activation, dtype=np.int8)
    scales = np.ascontiguousarray(activation_scales, dtype=np.float32)
    route = np.ascontiguousarray(buckets, dtype=np.uint16)
    directions = np.ascontiguousarray(signs, dtype=np.int8)
    coefficients = np.asarray(correction_codes)
    coefficient_scales = np.ascontiguousarray(
        correction_row_scales, dtype=np.float16
    )
    if anchors.shape != (encoded.rows,) or ratios.shape != (16,):
        raise ValueError("fused half-support row-scale metadata dimensions disagree")
    if values.shape != (encoded.columns,) or scales.shape != (groups,):
        raise ValueError("fused half-support activation dimensions disagree")
    if route.shape != values.shape or directions.shape != values.shape:
        raise ValueError("fused half-support hash-route dimensions disagree")
    if coefficients.ndim != 2 or coefficients.shape[0] != encoded.rows:
        raise ValueError("fused half-support coefficient dimensions disagree")
    sketch_dim = coefficients.shape[1]
    if sketch_dim <= 0 or sketch_dim > 65535 or np.any(route >= sketch_dim):
        raise ValueError("fused half-support bucket dimensions disagree")
    if not np.all((directions == -1) | (directions == 1)):
        raise ValueError("fused half-support hash signs must be -1 or +1")
    if coefficient_scales.shape != (encoded.rows,):
        raise ValueError("fused half-support correction row scales disagree")
    packed_coefficients = pack_signed_int4_rows(coefficients)
    sums = _activation_sums(values)
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    sketch = np.empty(sketch_dim, dtype=np.float32)
    correction = np.empty(encoded.rows, dtype=np.float32)
    output = np.empty(encoded.rows, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_htf_gemv_q8_row_scale4_{implementation}{suffix}",
    )
    common = (
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        packed_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        anchors.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ratios.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        encoded.rows, encoded.columns,
        route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        sketch_dim,
        packed_coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        coefficient_scales.view(np.uint16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        ),
        sketch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        correction.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    args = common if trusted else (common[0], payload.size, *common[1:])
    _check_status(function(*args))
    return output
