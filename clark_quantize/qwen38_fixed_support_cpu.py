"""C-ABI binding for the exact Qwen3.8 fixed-support ternary bitstream."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .qwen38_fixed_support_codec import FixedSupportPayload


_ERRORS = {
    1: "invalid fixed-support argument",
    2: "unsupported fixed-support shape",
    3: "invalid or non-canonical fixed-support code",
    4: "fixed-support output capacity is too small",
}


class FixedSupportCPULibrary:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        self.library.cq3_hfs_group_width.argtypes = (ctypes.c_uint32, u32p)
        self.library.cq3_hfs_group_width.restype = ctypes.c_int
        self.library.cq3_hfs_matrix_size.argtypes = (
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, u64p, u64p,
        )
        self.library.cq3_hfs_matrix_size.restype = ctypes.c_int
        self.library.cq3_hfs_pack_matrix.argtypes = (
            i8p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32,
            u8p, ctypes.c_uint64, u64p,
        )
        self.library.cq3_hfs_pack_matrix.restype = ctypes.c_int
        self.library.cq3_hfs_unpack_matrix.argtypes = (
            u8p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint32, i8p,
        )
        self.library.cq3_hfs_unpack_matrix.restype = ctypes.c_int


def load_fixed_support_cpu_library(
    path: str | os.PathLike[str] | None = None,
) -> FixedSupportCPULibrary | None:
    if path is not None:
        candidates = [Path(path)]
    else:
        explicit = os.environ.get("CLARK_FIXED_SUPPORT_CPU_LIB")
        candidates = [Path(explicit)] if explicit else [
            Path(__file__).with_name("_qwen38_fixed_support_cpu.so"),
            Path(__file__).with_name("_qwen38_fixed_support_cpu.dylib"),
        ]
    for candidate in candidates:
        try:
            return FixedSupportCPULibrary(candidate)
        except (OSError, AttributeError):
            continue
    return None


def _check_status(status: int) -> None:
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"native fixed-support error {status}"))


def native_pack_fixed_support_matrix(
    codes: np.ndarray,
    *,
    nonzero_per_group: int,
    library: FixedSupportCPULibrary,
) -> FixedSupportPayload:
    source = np.ascontiguousarray(codes)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("native fixed-support codes must be an integer matrix")
    if np.any(source < -1) or np.any(source > 1):
        raise ValueError("native fixed-support codes must lie in {-1, 0, +1}")
    source = np.ascontiguousarray(source, dtype=np.int8)
    bits = ctypes.c_uint64()
    size = ctypes.c_uint64()
    _check_status(library.library.cq3_hfs_matrix_size(
        source.shape[0], source.shape[1], nonzero_per_group,
        ctypes.byref(bits), ctypes.byref(size),
    ))
    payload = np.zeros(size.value, dtype=np.uint8)
    observed_bits = ctypes.c_uint64()
    _check_status(library.library.cq3_hfs_pack_matrix(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.shape[0], source.shape[1], nonzero_per_group,
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        payload.size, ctypes.byref(observed_bits),
    ))
    width = ctypes.c_uint32()
    _check_status(library.library.cq3_hfs_group_width(
        nonzero_per_group, ctypes.byref(width)
    ))
    if observed_bits.value != bits.value:
        raise RuntimeError("native fixed-support size and pack disagree")
    return FixedSupportPayload(
        payload=payload.tobytes(),
        bit_length=bits.value,
        rows=source.shape[0],
        columns=source.shape[1],
        group_size=128,
        nonzero_per_group=nonzero_per_group,
        group_width_bits=width.value,
    )


def native_unpack_fixed_support_matrix(
    encoded: FixedSupportPayload,
    *,
    library: FixedSupportCPULibrary,
) -> np.ndarray:
    if encoded.group_size != 128:
        raise ValueError("native fixed-support payload group size must be 128")
    expected_bits = ctypes.c_uint64()
    expected_size = ctypes.c_uint64()
    _check_status(library.library.cq3_hfs_matrix_size(
        encoded.rows, encoded.columns, encoded.nonzero_per_group,
        ctypes.byref(expected_bits), ctypes.byref(expected_size),
    ))
    expected_width = ctypes.c_uint32()
    _check_status(library.library.cq3_hfs_group_width(
        encoded.nonzero_per_group, ctypes.byref(expected_width)
    ))
    if (
        encoded.group_width_bits != expected_width.value
        or encoded.bit_length != expected_bits.value
        or len(encoded.payload) != expected_size.value
    ):
        raise ValueError("native fixed-support payload metadata is inconsistent")
    payload = np.frombuffer(encoded.payload, dtype=np.uint8).copy()
    output = np.empty((encoded.rows, encoded.columns), dtype=np.int8)
    _check_status(library.library.cq3_hfs_unpack_matrix(
        payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        payload.size, encoded.bit_length, encoded.rows, encoded.columns,
        encoded.nonzero_per_group,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    ))
    return output
