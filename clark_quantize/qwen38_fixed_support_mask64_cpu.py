"""Native 1.5-bit support-mask runtime for k=64 Qwen ternary groups."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np


_ERRORS = {
    1: "invalid mask64 argument",
    2: "unsupported mask64 shape",
    3: "invalid mask64 code or support",
    4: "non-finite mask64 scale",
    5: "requested mask64 CPU ISA is unavailable",
}


@dataclass(frozen=True)
class Mask64Payload:
    payload: bytes
    rows: int
    columns: int
    group_size: int = 128
    nonzero_per_group: int = 64


class Mask64CPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        library = ctypes.CDLL(str(self.path))
        self.library = library
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        library.cq3_hsm64_cpu_features.argtypes = ()
        library.cq3_hsm64_cpu_features.restype = ctypes.c_uint32
        library.cq3_hsm64_matrix_size.argtypes = (
            ctypes.c_uint64, ctypes.c_uint64, u64p,
        )
        library.cq3_hsm64_matrix_size.restype = ctypes.c_int
        library.cq3_hsm64_pack_matrix.argtypes = (
            i8p, ctypes.c_uint64, ctypes.c_uint64, u8p, ctypes.c_uint64,
        )
        library.cq3_hsm64_pack_matrix.restype = ctypes.c_int
        library.cq3_hsm64_unpack_matrix.argtypes = (
            u8p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, i8p,
        )
        library.cq3_hsm64_unpack_matrix.restype = ctypes.c_int
        safe_gemv = (
            u8p, ctypes.c_uint64, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, f32p,
        )
        trusted_gemv = (
            u8p, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, f32p,
        )
        for implementation in ("scalar", "avx2_bmi2"):
            safe = getattr(library, f"cq3_hsm64_gemv_q8_row_scale4_{implementation}")
            safe.argtypes = safe_gemv
            safe.restype = ctypes.c_int
            trusted = getattr(
                library,
                f"cq3_hsm64_gemv_q8_row_scale4_{implementation}_trusted",
            )
            trusted.argtypes = trusted_gemv
            trusted.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        value = int(self.library.cq3_hsm64_cpu_features())
        return {
            "avx2": bool(value & 1),
            "fma": bool(value & 2),
            "f16c": bool(value & 4),
            "bmi2": bool(value & 8),
        }


def load_mask64_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> Mask64CPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_MASK64_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_fixed_support_mask64_cpu.so"),
            Path(__file__).with_name("_qwen38_fixed_support_mask64_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return Mask64CPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"native mask64 error {status}"))


def native_pack_mask64_matrix(
    codes: np.ndarray,
    *,
    library: Mask64CPULibrary,
) -> Mask64Payload:
    source = np.asarray(codes)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("mask64 codes must be an integer matrix")
    if np.any(source < -1) or np.any(source > 1):
        raise ValueError("mask64 codes must lie in {-1, 0, +1}")
    source = np.ascontiguousarray(source, dtype=np.int8)
    size = ctypes.c_uint64()
    _check_status(library.library.cq3_hsm64_matrix_size(
        source.shape[0], source.shape[1], ctypes.byref(size)
    ))
    payload = np.empty(size.value, dtype=np.uint8)
    _check_status(library.library.cq3_hsm64_pack_matrix(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.shape[0], source.shape[1],
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), payload.size,
    ))
    return Mask64Payload(
        payload=payload.tobytes(), rows=source.shape[0], columns=source.shape[1]
    )


def native_unpack_mask64_matrix(
    encoded: Mask64Payload,
    *,
    library: Mask64CPULibrary,
) -> np.ndarray:
    if encoded.group_size != 128 or encoded.nonzero_per_group != 64:
        raise ValueError("mask64 payload metadata is inconsistent")
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty((encoded.rows, encoded.columns), dtype=np.int8)
    _check_status(library.library.cq3_hsm64_unpack_matrix(
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        payload.size, encoded.rows, encoded.columns,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    ))
    return output


def pack_scale_indices4(indices: np.ndarray) -> np.ndarray:
    source = np.asarray(indices)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("mask64 scale indices must be an integer matrix")
    if np.any(source < 0) or np.any(source > 15):
        raise ValueError("mask64 scale indices must lie in [0, 15]")
    flattened = np.ascontiguousarray(source, dtype=np.uint8).reshape(-1)
    output = np.zeros((flattened.size + 1) // 2, dtype=np.uint8)
    output[: flattened[0::2].size] = flattened[0::2]
    output[: flattened[1::2].size] |= flattened[1::2] << 4
    return output


def native_mask64_gemv_q8_row_scale4(
    encoded: Mask64Payload,
    scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation: np.ndarray,
    activation_scales: np.ndarray,
    *,
    library: Mask64CPULibrary,
    implementation: str = "scalar",
    trusted: bool = False,
) -> np.ndarray:
    if implementation not in {"scalar", "avx2_bmi2"}:
        raise ValueError("unknown mask64 GEMV implementation")
    groups = encoded.columns // 128
    indices = np.asarray(scale_indices)
    if indices.shape != (encoded.rows, groups):
        raise ValueError("mask64 scale-index dimensions disagree")
    packed_indices = pack_scale_indices4(indices)
    anchors = np.ascontiguousarray(row_anchors, dtype=np.float16)
    ratios = np.ascontiguousarray(ratio_codebook, dtype=np.float16)
    values = np.ascontiguousarray(activation, dtype=np.int8)
    activation_scale_values = np.ascontiguousarray(
        activation_scales, dtype=np.float32
    )
    if anchors.shape != (encoded.rows,) or ratios.shape != (16,):
        raise ValueError("mask64 row-scale metadata dimensions disagree")
    if values.shape != (encoded.columns,) or activation_scale_values.shape != (groups,):
        raise ValueError("mask64 activation dimensions disagree")
    activation_sums = values.reshape(groups, 128).sum(axis=1, dtype=np.int32)
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty(encoded.rows, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_hsm64_gemv_q8_row_scale4_{implementation}{suffix}",
    )
    common = (
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        packed_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        anchors.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ratios.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        activation_scale_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        activation_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        encoded.rows, encoded.columns,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    args = common if trusted else (common[0], payload.size, *common[1:])
    _check_status(function(*args))
    return output


def mask64_hash_component_storage(
    *,
    input_dim: int,
    output_rows: int,
    sketch_dim: int,
    header_bytes: int = 64,
) -> dict[str, int | float]:
    """Count mask64 ternary runtime plus learned INT4 hash correction."""
    if (
        input_dim <= 0
        or output_rows <= 0
        or sketch_dim <= 0
        or input_dim % 128
        or header_bytes < 0
    ):
        raise ValueError("invalid mask64 hash-component storage dimensions")
    parameters = input_dim * output_rows
    ternary_code_bytes = output_rows * (input_dim // 128) * 24
    scale_index_bytes = (output_rows * (input_dim // 128) * 4 + 7) // 8
    hash_code_bytes = output_rows * ((sketch_dim + 1) // 2)
    hash_scale_bytes = output_rows * 2
    row_anchor_bytes = output_rows * 2
    tensor_codebook_bytes = 16 * 2
    bucket_bits = max(1, math.ceil(math.log2(sketch_dim)))
    routing_bytes = (input_dim * (bucket_bits + 1) + 7) // 8
    total = (
        ternary_code_bytes
        + scale_index_bytes
        + hash_code_bytes
        + hash_scale_bytes
        + row_anchor_bytes
        + tensor_codebook_bytes
        + routing_bytes
        + header_bytes
    )
    return {
        "ternary_code_bytes": ternary_code_bytes,
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
