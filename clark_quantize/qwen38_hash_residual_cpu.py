"""Native CPU execution of the compact ``A(Rx)`` Clark-Hash correction.

The canonical artifact stores signed INT4 coefficients and FP16 row scales.
The sparse signed projection is regenerated from its seed; bucket/sign arrays
are load-time compute metadata rather than model bytes.
"""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np


FEATURE_AVX2 = 1
FEATURE_FMA = 2
_ERRORS = {
    1: "invalid pointer argument",
    2: "unsupported tensor shape",
    3: "invalid or non-canonical INT4/projection code",
    4: "non-finite numeric input",
    5: "requested CPU ISA is unavailable",
}


class HashResidualCPULibrary:
    """Typed handle for the separately versioned hash-residual C ABI."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        f32p = ctypes.POINTER(ctypes.c_float)
        self.library.cq3_hash_cpu_features.argtypes = ()
        self.library.cq3_hash_cpu_features.restype = ctypes.c_uint32
        projection_args = (
            f32p,
            u16p,
            i8p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            f32p,
        )
        self.library.cq3_hash_projection_f32_scalar.argtypes = projection_args
        self.library.cq3_hash_projection_f32_scalar.restype = ctypes.c_int
        self.library.cq3_hash_projection_f32_scalar_trusted.argtypes = projection_args
        self.library.cq3_hash_projection_f32_scalar_trusted.restype = ctypes.c_int
        gemv_args = (
            u8p,
            u16p,
            f32p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            f32p,
        )
        for name in (
            "cq3_hash_int4_gemv_f32_scalar",
            "cq3_hash_int4_gemv_f32_scalar_trusted",
            "cq3_hash_int4_gemv_f32_avx2",
            "cq3_hash_int4_gemv_f32_avx2_trusted",
        ):
            function = getattr(self.library, name)
            function.argtypes = gemv_args
            function.restype = ctypes.c_int
        residual_args = (
            f32p,
            u16p,
            i8p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            u8p,
            u16p,
            ctypes.c_uint64,
            f32p,
            f32p,
            ctypes.c_uint32,
        )
        for name in (
            "cq3_hash_residual_f32_scalar",
            "cq3_hash_residual_f32_scalar_trusted",
            "cq3_hash_residual_f32_avx2",
            "cq3_hash_residual_f32_avx2_trusted",
        ):
            function = getattr(self.library, name)
            function.argtypes = residual_args
            function.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        value = int(self.library.cq3_hash_cpu_features())
        return {
            "avx2": bool(value & FEATURE_AVX2),
            "fma": bool(value & FEATURE_FMA),
        }


def load_hash_residual_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> HashResidualCPULibrary | None:
    candidates: list[Path]
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_HASH_RESIDUAL_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_hash_residual_cpu.so"),
            Path(__file__).with_name("_qwen38_hash_residual_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return HashResidualCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def pack_signed_int4_rows(codes: np.ndarray) -> np.ndarray:
    source = np.asarray(codes)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("INT4 coefficient codes must be a non-empty matrix")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("INT4 coefficient codes must be integers")
    source_i8 = np.ascontiguousarray(source, dtype=np.int8)
    if np.any(source_i8 < -7) or np.any(source_i8 > 7):
        raise ValueError("INT4 coefficient codes must lie in [-7, 7]")
    rows, columns = source_i8.shape
    packed = np.zeros((rows, (columns + 1) // 2), dtype=np.uint8)
    nibbles = source_i8.astype(np.int16) & 0x0F
    packed[:, :] = nibbles[:, 0::2].astype(np.uint8)
    if columns > 1:
        packed[:, : columns // 2] |= (
            nibbles[:, 1::2].astype(np.uint8) << np.uint8(4)
        )
    return np.ascontiguousarray(packed)


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"native hash-residual error {status}"))


def native_hash_projection_f32(
    values: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    *,
    sketch_dim: int,
    library: HashResidualCPULibrary,
    trusted: bool = False,
) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    mapping = np.ascontiguousarray(buckets, dtype=np.uint16)
    directions = np.ascontiguousarray(signs, dtype=np.int8)
    if source.ndim != 1 or mapping.ndim != 2 or directions.shape != mapping.shape:
        raise ValueError("hash projection arrays have incompatible dimensions")
    if mapping.shape[1] != source.shape[0] or sketch_dim <= 0 or sketch_dim > 65535:
        raise ValueError("hash projection dimensions disagree")
    output = np.empty(sketch_dim, dtype=np.float32)
    function = getattr(
        library.library,
        "cq3_hash_projection_f32_scalar_trusted"
        if trusted
        else "cq3_hash_projection_f32_scalar",
    )
    status = function(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.shape[0],
        sketch_dim,
        mapping.shape[0],
        np.float32(1.0 / math.sqrt(mapping.shape[0])),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check_status(status)
    return output


def native_hash_int4_gemv_f32(
    packed_codes: np.ndarray,
    row_scales: np.ndarray,
    sketch: np.ndarray,
    *,
    sketch_dim: int,
    library: HashResidualCPULibrary,
    implementation: str = "auto",
    trusted: bool = False,
) -> np.ndarray:
    packed = np.ascontiguousarray(packed_codes, dtype=np.uint8)
    scales = np.ascontiguousarray(row_scales, dtype=np.float16)
    source = np.ascontiguousarray(sketch, dtype=np.float32)
    if packed.ndim != 2 or scales.ndim != 1 or source.ndim != 1:
        raise ValueError("INT4 hash GEMV arrays have incompatible dimensions")
    if (
        packed.shape[0] != scales.shape[0]
        or packed.shape[1] != (sketch_dim + 1) // 2
        or source.shape[0] != sketch_dim
    ):
        raise ValueError("INT4 hash GEMV dimensions disagree")
    if implementation not in {"auto", "scalar", "avx2"}:
        raise ValueError("hash GEMV implementation must be auto, scalar, or avx2")
    selected = implementation
    features = library.features()
    if selected == "auto":
        selected = "avx2" if features["avx2"] and features["fma"] else "scalar"
    if selected == "avx2" and not (features["avx2"] and features["fma"]):
        raise RuntimeError("requested CPU ISA is unavailable")
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_hash_int4_gemv_f32_{selected}{suffix}",
    )
    output = np.empty(packed.shape[0], dtype=np.float32)
    scale_bits = scales.view(np.uint16)
    status = function(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scale_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        packed.shape[0],
        sketch_dim,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check_status(status)
    return output


def native_hash_residual_f32(
    values: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    packed_codes: np.ndarray,
    row_scales: np.ndarray,
    *,
    sketch_dim: int,
    library: HashResidualCPULibrary,
    implementation: str = "auto",
    trusted: bool = False,
    base_output: np.ndarray | None = None,
) -> np.ndarray:
    """Execute ``A(Rx)`` or add it directly to an existing ternary output."""
    source = np.ascontiguousarray(values, dtype=np.float32)
    mapping = np.ascontiguousarray(buckets, dtype=np.uint16)
    directions = np.ascontiguousarray(signs, dtype=np.int8)
    packed = np.ascontiguousarray(packed_codes, dtype=np.uint8)
    scales = np.ascontiguousarray(row_scales, dtype=np.float16)
    if (
        source.ndim != 1
        or mapping.ndim != 2
        or directions.shape != mapping.shape
        or mapping.shape[1] != source.shape[0]
        or packed.ndim != 2
        or scales.ndim != 1
        or packed.shape[0] != scales.shape[0]
        or packed.shape[1] != (sketch_dim + 1) // 2
    ):
        raise ValueError("native hash-residual dimensions disagree")
    if implementation not in {"auto", "scalar", "avx2"}:
        raise ValueError("hash-residual implementation must be auto, scalar, or avx2")
    selected = implementation
    features = library.features()
    if selected == "auto":
        selected = "avx2" if features["avx2"] and features["fma"] else "scalar"
    if selected == "avx2" and not (features["avx2"] and features["fma"]):
        raise RuntimeError("requested CPU ISA is unavailable")
    output = (
        np.zeros(packed.shape[0], dtype=np.float32)
        if base_output is None
        else np.ascontiguousarray(base_output, dtype=np.float32).copy()
    )
    if output.ndim != 1 or output.shape[0] != packed.shape[0]:
        raise ValueError("base ternary output and hash coefficients disagree")
    scratch = np.empty(sketch_dim, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_hash_residual_f32_{selected}{suffix}",
    )
    scale_bits = scales.view(np.uint16)
    status = function(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.shape[0],
        sketch_dim,
        mapping.shape[0],
        np.float32(1.0 / math.sqrt(mapping.shape[0])),
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scale_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        packed.shape[0],
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0 if base_output is None else 1,
    )
    _check_status(status)
    return output
