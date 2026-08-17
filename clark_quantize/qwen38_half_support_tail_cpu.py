"""Native half-support codec and Qwen vision-tail GEMV."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .qwen38_fixed_support_mask64_cpu import Mask64CPULibrary, pack_scale_indices4


_ERRORS = {
    1: "invalid half-support tail argument",
    2: "unsupported half-support tail shape",
    3: "invalid half-support tail code or support",
    4: "non-finite half-support tail scale",
    5: "requested half-support tail CPU ISA is unavailable",
}


@dataclass(frozen=True)
class HalfSupportTailPayload:
    payload: bytes
    rows: int
    columns: int
    group_size: int = 128

    @property
    def tail_width(self) -> int:
        return self.columns % self.group_size


class HalfSupportTailCPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.mask = Mask64CPULibrary(self.path)
        self.library = self.mask.library
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        self.library.cq3_hst_matrix_size.argtypes = (
            ctypes.c_uint64, ctypes.c_uint64, u64p,
        )
        self.library.cq3_hst_matrix_size.restype = ctypes.c_int
        self.library.cq3_hst_pack_matrix.argtypes = (
            i8p, ctypes.c_uint64, ctypes.c_uint64, u8p, ctypes.c_uint64,
        )
        self.library.cq3_hst_pack_matrix.restype = ctypes.c_int
        self.library.cq3_hst_unpack_matrix.argtypes = (
            u8p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, i8p,
        )
        self.library.cq3_hst_unpack_matrix.restype = ctypes.c_int
        safe = (
            u8p, ctypes.c_uint64, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, f32p,
        )
        trusted = (
            u8p, u8p, u16p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, f32p,
        )
        for implementation in ("scalar", "avx2_bmi2"):
            function = getattr(
                self.library, f"cq3_hst_gemv_q8_row_scale4_{implementation}"
            )
            function.argtypes = safe
            function.restype = ctypes.c_int
            trusted_function = getattr(
                self.library,
                f"cq3_hst_gemv_q8_row_scale4_{implementation}_trusted",
            )
            trusted_function.argtypes = trusted
            trusted_function.restype = ctypes.c_int

    def features(self) -> dict[str, bool]:
        return self.mask.features()


def load_half_support_tail_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> HalfSupportTailCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_HALF_SUPPORT_TAIL_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_half_support_tail_cpu.so"),
            Path(__file__).with_name("_qwen38_half_support_tail_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return HalfSupportTailCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"native half-support tail error {status}"))


def native_pack_half_support_tail_matrix(
    codes: np.ndarray,
    *,
    library: HalfSupportTailCPULibrary,
) -> HalfSupportTailPayload:
    source = np.asarray(codes)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("half-support tail codes must be an integer matrix")
    if np.any(source < -1) or np.any(source > 1):
        raise ValueError("half-support tail codes must lie in {-1,0,+1}")
    source = np.ascontiguousarray(source, dtype=np.int8)
    size = ctypes.c_uint64()
    _check_status(library.library.cq3_hst_matrix_size(
        source.shape[0], source.shape[1], ctypes.byref(size)
    ))
    payload = np.empty(size.value, dtype=np.uint8)
    _check_status(library.library.cq3_hst_pack_matrix(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.shape[0], source.shape[1],
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), payload.size,
    ))
    return HalfSupportTailPayload(
        payload=payload.tobytes(), rows=source.shape[0], columns=source.shape[1]
    )


def native_unpack_half_support_tail_matrix(
    encoded: HalfSupportTailPayload,
    *,
    library: HalfSupportTailCPULibrary,
) -> np.ndarray:
    if encoded.group_size != 128:
        raise ValueError("half-support tail payload metadata is inconsistent")
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty((encoded.rows, encoded.columns), dtype=np.int8)
    _check_status(library.library.cq3_hst_unpack_matrix(
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        payload.size, encoded.rows, encoded.columns,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    ))
    return output


def _activation_sums(values: np.ndarray) -> np.ndarray:
    groups = (values.size + 127) // 128
    output = np.empty(groups, dtype=np.int32)
    for group in range(groups):
        start = group * 128
        output[group] = values[start : start + 128].sum(dtype=np.int32)
    return output


def native_half_support_tail_gemv_q8_row_scale4(
    encoded: HalfSupportTailPayload,
    scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation: np.ndarray,
    activation_scales: np.ndarray,
    *,
    library: HalfSupportTailCPULibrary,
    implementation: str = "scalar",
    trusted: bool = False,
) -> np.ndarray:
    if implementation not in {"scalar", "avx2_bmi2"}:
        raise ValueError("unknown half-support tail GEMV implementation")
    groups = (encoded.columns + 127) // 128
    indices = np.asarray(scale_indices)
    if indices.shape != (encoded.rows, groups):
        raise ValueError("half-support tail scale-index dimensions disagree")
    packed_indices = pack_scale_indices4(indices)
    anchors = np.ascontiguousarray(row_anchors, dtype=np.float16)
    ratios = np.ascontiguousarray(ratio_codebook, dtype=np.float16)
    values = np.ascontiguousarray(activation, dtype=np.int8)
    scales = np.ascontiguousarray(activation_scales, dtype=np.float32)
    if anchors.shape != (encoded.rows,) or ratios.shape != (16,):
        raise ValueError("half-support tail scale metadata dimensions disagree")
    if values.shape != (encoded.columns,) or scales.shape != (groups,):
        raise ValueError("half-support tail activation dimensions disagree")
    sums = _activation_sums(values)
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty(encoded.rows, dtype=np.float32)
    suffix = "_trusted" if trusted else ""
    function = getattr(
        library.library,
        f"cq3_hst_gemv_q8_row_scale4_{implementation}{suffix}",
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
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    args = common if trusted else (common[0], payload.size, *common[1:])
    _check_status(function(*args))
    return output
