"""C-ABI CPU primitives for Clark group-128 ternary weights.

The canonical base-3 artifact remains the storage source of truth. These
helpers compare an aligned two-bit compute image with a direct group-aligned
radix-5 image whose group boundaries match the FP16 scale grid exactly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path

import numpy as np


FEATURE_AVX2 = 1
FEATURE_F16C = 2
FEATURE_FMA = 4
FEATURE_NEON = 8
FEATURE_DOTPROD = 16
_ERRORS = {
    1: "invalid pointer argument",
    2: "unsupported tensor shape",
    3: "invalid or non-canonical ternary code",
    4: "non-finite numeric input",
    5: "requested CPU ISA is unavailable",
}


def _load_library() -> ctypes.CDLL | None:
    explicit = os.environ.get("CLARK_TERNARY_CPU_LIB")
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).with_name("_ternary_cpu.so"),
        Path(__file__).with_name("_ternary_cpu.dylib"),
    ]
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError:
            continue
        try:
            library.cq3_block203_size
            library.cq3_pack_block203
            library.cq3_unpack_block203
            library.cq3_block203_matrix_size
            library.cq3_pack_block203_rows
            library.cq3_unpack_block203_rows
        except AttributeError:
            # A stale library must not make the Python module unimportable.
            # The exact storage codec is part of this ABI revision, so use the
            # audited Python fallback until the local library is rebuilt.
            continue
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)
        library.cq3_cpu_features.argtypes = ()
        library.cq3_cpu_features.restype = ctypes.c_uint32
        library.cq3_block203_size.argtypes = (
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        )
        library.cq3_block203_size.restype = ctypes.c_int
        library.cq3_pack_block203.argtypes = (
            i8p, ctypes.c_uint64, u8p, ctypes.c_uint64,
        )
        library.cq3_pack_block203.restype = ctypes.c_int
        library.cq3_unpack_block203.argtypes = (
            u8p, ctypes.c_uint64, ctypes.c_uint64, i8p,
        )
        library.cq3_unpack_block203.restype = ctypes.c_int
        library.cq3_block203_matrix_size.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        )
        library.cq3_block203_matrix_size.restype = ctypes.c_int
        library.cq3_pack_block203_rows.argtypes = (
            i8p, ctypes.c_uint64, ctypes.c_uint64, u8p, ctypes.c_uint64,
        )
        library.cq3_pack_block203_rows.restype = ctypes.c_int
        library.cq3_unpack_block203_rows.argtypes = (
            u8p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, i8p,
        )
        library.cq3_unpack_block203_rows.restype = ctypes.c_int
        library.cq3_pack_aligned2.argtypes = (i8p, ctypes.c_uint64, u8p)
        library.cq3_pack_aligned2.restype = ctypes.c_int
        library.cq3_unpack_aligned2.argtypes = (u8p, ctypes.c_uint64, i8p)
        library.cq3_unpack_aligned2.restype = ctypes.c_int
        library.cq3_pack_group128_2bit.argtypes = (i8p, ctypes.c_uint64, u8p)
        library.cq3_pack_group128_2bit.restype = ctypes.c_int
        library.cq3_unpack_group128_2bit.argtypes = (u8p, ctypes.c_uint64, i8p)
        library.cq3_unpack_group128_2bit.restype = ctypes.c_int
        library.cq3_pack_group128_radix5.argtypes = (i8p, ctypes.c_uint64, u8p)
        library.cq3_pack_group128_radix5.restype = ctypes.c_int
        library.cq3_unpack_group128_radix5.argtypes = (u8p, ctypes.c_uint64, i8p)
        library.cq3_unpack_group128_radix5.restype = ctypes.c_int
        library.cq3_quantize_f32_q8.argtypes = (
            f32p, ctypes.c_uint64, ctypes.c_uint32, i8p, f32p, i32p,
        )
        library.cq3_quantize_f32_q8.restype = ctypes.c_int
        library.cq3_expand_q8_radix5.argtypes = (i8p, ctypes.c_uint64, i8p)
        library.cq3_expand_q8_radix5.restype = ctypes.c_int
        library.cq3_expand_q8_radix5_planes.argtypes = (i8p, ctypes.c_uint64, i8p)
        library.cq3_expand_q8_radix5_planes.restype = ctypes.c_int
        if hasattr(library, "cq3_decode_row_scale4_scalar"):
            row_scale_args = (
                u8p, u16p, u16p, ctypes.c_uint64, ctypes.c_uint64, f32p,
            )
            library.cq3_decode_row_scale4_scalar.argtypes = row_scale_args
            library.cq3_decode_row_scale4_scalar.restype = ctypes.c_int
            library.cq3_decode_row_scale4_avx2.argtypes = row_scale_args
            library.cq3_decode_row_scale4_avx2.restype = ctypes.c_int
        gemv_args = (
            u8p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, f32p,
        )
        library.cq3_gemv_q8_group128_2bit_scalar.argtypes = gemv_args
        library.cq3_gemv_q8_group128_2bit_scalar.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_2bit_avx2.argtypes = gemv_args
        library.cq3_gemv_q8_group128_2bit_avx2.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_2bit_arm_dotprod.argtypes = gemv_args
        library.cq3_gemv_q8_group128_2bit_arm_dotprod.restype = ctypes.c_int
        if hasattr(library, "cq3_gemv_q8_group128_2bit_row_scale4_scalar"):
            row_scale_gemv_args = (
                u8p, u8p, u16p, u16p, i8p, f32p, i32p,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, f32p,
            )
            library.cq3_gemv_q8_group128_2bit_row_scale4_scalar.argtypes = (
                row_scale_gemv_args
            )
            library.cq3_gemv_q8_group128_2bit_row_scale4_scalar.restype = ctypes.c_int
            library.cq3_gemv_q8_group128_2bit_row_scale4_avx2.argtypes = (
                row_scale_gemv_args
            )
            library.cq3_gemv_q8_group128_2bit_row_scale4_avx2.restype = ctypes.c_int
        swiglu_gemv_args = (
            u8p, u16p, u8p, u16p, i8p, f32p, i32p,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, f32p,
        )
        library.cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod.argtypes = swiglu_gemv_args
        library.cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod.restype = ctypes.c_int
        library.cq3_swiglu_f32.argtypes = (
            f32p, f32p, ctypes.c_uint64, f32p,
        )
        library.cq3_swiglu_f32.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_scalar.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_scalar.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_avx2.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_avx2.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_avx2_nogather.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_avx2_nogather.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_avx2_nogather_static.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_avx2_nogather_static.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_avx2_planes.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_avx2_planes.restype = ctypes.c_int
        library.cq3_gemv_q8_group128_radix5_avx2_nogather_pair.argtypes = gemv_args
        library.cq3_gemv_q8_group128_radix5_avx2_nogather_pair.restype = ctypes.c_int
        library.cq3_sum_u8_scalar.argtypes = (u8p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64))
        library.cq3_sum_u8_scalar.restype = ctypes.c_int
        library.cq3_sum_u8_avx2.argtypes = (u8p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64))
        library.cq3_sum_u8_avx2.restype = ctypes.c_int
        library.cq3_sum_u8_neon.argtypes = (u8p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64))
        library.cq3_sum_u8_neon.restype = ctypes.c_int
        return library
    return None


_LIBRARY = _load_library()


def native_available() -> bool:
    return _LIBRARY is not None


def cpu_features() -> dict[str, bool]:
    flags = int(_LIBRARY.cq3_cpu_features()) if _LIBRARY is not None else 0
    return {
        "avx2": bool(flags & FEATURE_AVX2),
        "f16c": bool(flags & FEATURE_F16C),
        "fma": bool(flags & FEATURE_FMA),
        "neon": bool(flags & FEATURE_NEON),
        "dotprod": bool(flags & FEATURE_DOTPROD),
    }


def _check(status: int, operation: str) -> None:
    if status:
        raise ValueError(f"{operation} failed: {_ERRORS.get(status, f'error {status}')}")


def _array(value, dtype, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array")
    return np.ascontiguousarray(result)


def block203_size(count: int) -> tuple[int, int]:
    """Return the exact meaningful bits and byte envelope for a ternary row."""
    if count < 0:
        raise ValueError("trit count must be non-negative")
    if _LIBRARY is None:
        from .base3_block203 import row_payload_bits, row_payload_bytes

        return row_payload_bits(count), row_payload_bytes(count)
    bits = ctypes.c_uint64()
    size = ctypes.c_uint64()
    status = _LIBRARY.cq3_block203_size(count, ctypes.byref(bits), ctypes.byref(size))
    _check(status, "block-203 size")
    return int(bits.value), int(size.value)


def pack_block203(codes: np.ndarray) -> np.ndarray:
    """Pack ternary codes into canonical little-endian 203-bit base-3 blocks."""
    values = np.asarray(codes).reshape(-1)
    if values.size and not np.all((values == -1) | (values == 0) | (values == 1)):
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    source = np.ascontiguousarray(values, dtype=np.int8)
    _, output_bytes = block203_size(source.size)
    if _LIBRARY is None:
        from .base3_block203 import pack_row

        return np.frombuffer(pack_row(source), dtype=np.uint8).copy()
    output = np.empty(output_bytes, dtype=np.uint8)
    status = _LIBRARY.cq3_pack_block203(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.size,
    )
    _check(status, "block-203 pack")
    return output


def unpack_block203(packed: np.ndarray, count: int) -> np.ndarray:
    """Decode an exact canonical 203-bit-block row into {-1, 0, +1}."""
    source = _array(packed, np.uint8).reshape(-1)
    _, expected_bytes = block203_size(count)
    if source.size != expected_bytes:
        raise ValueError("block-203 dimensions do not match")
    if _LIBRARY is None:
        from .base3_block203 import unpack_row

        return unpack_row(source.tobytes(), count)
    output = np.empty(count, dtype=np.int8)
    status = _LIBRARY.cq3_unpack_block203(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        source.size,
        count,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "block-203 unpack")
    return output


def block203_matrix_size(rows: int, columns: int) -> tuple[int, int]:
    """Return exact bits/bytes for per-row groups in one bit-contiguous matrix."""
    if rows <= 0 or columns <= 0:
        raise ValueError("block-203 matrix dimensions must be positive")
    if _LIBRARY is None:
        from .base3_block203 import matrix_payload_bits, matrix_payload_bytes

        return matrix_payload_bits(rows, columns), matrix_payload_bytes(rows, columns)
    bits = ctypes.c_uint64()
    size = ctypes.c_uint64()
    status = _LIBRARY.cq3_block203_matrix_size(
        rows, columns, ctypes.byref(bits), ctypes.byref(size)
    )
    _check(status, "block-203 matrix size")
    return int(bits.value), int(size.value)


def pack_block203_rows(codes: np.ndarray) -> np.ndarray:
    """Pack a matrix with group boundaries reset per row and no row padding."""
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("block-203 matrix codes must have two positive dimensions")
    if not np.all((values == -1) | (values == 0) | (values == 1)):
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    source = np.ascontiguousarray(values, dtype=np.int8)
    rows, columns = source.shape
    _, output_bytes = block203_matrix_size(rows, columns)
    if _LIBRARY is None:
        from .base3_block203 import pack_rows

        return np.frombuffer(pack_rows(source), dtype=np.uint8).copy()
    output = np.empty(output_bytes, dtype=np.uint8)
    status = _LIBRARY.cq3_pack_block203_rows(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        rows,
        columns,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.size,
    )
    _check(status, "block-203 matrix pack")
    return output


def unpack_block203_rows(
    packed: np.ndarray, rows: int, columns: int
) -> np.ndarray:
    """Decode a bit-contiguous matrix whose group boundaries reset per row."""
    source = _array(packed, np.uint8).reshape(-1)
    _, expected_bytes = block203_matrix_size(rows, columns)
    if source.size != expected_bytes:
        raise ValueError("block-203 matrix dimensions do not match")
    if _LIBRARY is None:
        from .base3_block203 import unpack_rows

        return unpack_rows(source.tobytes(), rows, columns)
    output = np.empty((rows, columns), dtype=np.int8)
    status = _LIBRARY.cq3_unpack_block203_rows(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        source.size,
        rows,
        columns,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "block-203 matrix unpack")
    return output


def pack_aligned2(codes: np.ndarray) -> np.ndarray:
    source = _array(codes, np.int8).reshape(-1)
    if _LIBRARY is None:
        if source.size and (source.min() < -1 or source.max() > 1):
            raise ValueError("ternary codes must be in {-1, 0, +1}")
        digits = source.astype(np.uint8) + 1
        padding = (-digits.size) % 4
        if padding:
            digits = np.pad(digits, (0, padding), constant_values=1)
        if not digits.size:
            return np.empty(0, dtype=np.uint8)
        groups = digits.reshape(-1, 4)
        return np.ascontiguousarray(
            groups[:, 0] | (groups[:, 1] << 2) | (groups[:, 2] << 4) | (groups[:, 3] << 6)
        )
    output = np.empty((source.size + 3) // 4, dtype=np.uint8)
    status = _LIBRARY.cq3_pack_aligned2(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    _check(status, "aligned-2 pack")
    return output


def unpack_aligned2(packed: np.ndarray, count: int) -> np.ndarray:
    source = _array(packed, np.uint8).reshape(-1)
    if count < 0 or source.size != (count + 3) // 4:
        raise ValueError("aligned-2 dimensions do not match")
    if _LIBRARY is None:
        digits = np.empty(source.size * 4, dtype=np.uint8)
        for lane in range(4):
            digits[lane::4] = (source >> (lane * 2)) & 3
        if np.any(digits[:count] == 3) or np.any(digits[count:] != 1):
            raise ValueError("aligned-2 payload is non-canonical")
        return (digits[:count].astype(np.int8) - 1).astype(np.int8, copy=False)
    output = np.empty(count, dtype=np.int8)
    status = _LIBRARY.cq3_unpack_aligned2(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "aligned-2 unpack")
    return output


def pack_group128_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack each 128-code group into 32 bytes arranged for SIMD bit extraction."""
    source = _array(codes, np.int8).reshape(-1)
    if source.size == 0 or source.size % 128:
        raise ValueError("group-128 compute images require a nonzero multiple of 128 codes")
    if _LIBRARY is None:
        if source.min() < -1 or source.max() > 1:
            raise ValueError("ternary codes must be in {-1, 0, +1}")
        digits = (source.astype(np.uint8) + 1).reshape(-1, 128)
        output = np.zeros((digits.shape[0], 32), dtype=np.uint8)
        for quarter in range(4):
            output |= digits[:, quarter * 32:(quarter + 1) * 32] << (quarter * 2)
        return output.reshape(-1)
    output = np.empty(source.size // 4, dtype=np.uint8)
    status = _LIBRARY.cq3_pack_group128_2bit(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    _check(status, "group-128 2-bit pack")
    return output


def unpack_group128_2bit(packed: np.ndarray, count: int) -> np.ndarray:
    source = _array(packed, np.uint8).reshape(-1)
    if count <= 0 or count % 128 or source.size != count // 4:
        raise ValueError("group-128 2-bit dimensions do not match")
    if _LIBRARY is None:
        values = source.reshape(-1, 32)
        digits = np.empty((values.shape[0], 128), dtype=np.uint8)
        for quarter in range(4):
            digits[:, quarter * 32:(quarter + 1) * 32] = (values >> (quarter * 2)) & 3
        if np.any(digits == 3):
            raise ValueError("group-128 2-bit payload is non-canonical")
        return (digits.reshape(-1).astype(np.int8) - 1).astype(np.int8, copy=False)
    output = np.empty(count, dtype=np.int8)
    status = _LIBRARY.cq3_unpack_group128_2bit(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "group-128 2-bit unpack")
    return output


def pack_group128_radix5(codes: np.ndarray) -> np.ndarray:
    """Pack every 128-code scale group into 26 radix-5 bytes."""
    source = _array(codes, np.int8).reshape(-1)
    if source.size == 0 or source.size % 128:
        raise ValueError("group-128 radix-5 images require a nonzero multiple of 128 codes")
    groups = source.size // 128
    if _LIBRARY is None:
        if source.min() < -1 or source.max() > 1:
            raise ValueError("ternary codes must be in {-1, 0, +1}")
        digits = (source.astype(np.uint16) + 1).reshape(groups, 128)
        padded = np.zeros((groups, 130), dtype=np.uint16)
        padded[:, :128] = digits
        powers = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)
        return np.ascontiguousarray((padded.reshape(groups, 26, 5) * powers).sum(axis=2), dtype=np.uint8).reshape(-1)
    output = np.empty(groups * 26, dtype=np.uint8)
    status = _LIBRARY.cq3_pack_group128_radix5(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    _check(status, "group-128 radix-5 pack")
    return output


def unpack_group128_radix5(packed: np.ndarray, count: int) -> np.ndarray:
    source = _array(packed, np.uint8).reshape(-1)
    if count <= 0 or count % 128 or source.size != count // 128 * 26:
        raise ValueError("group-128 radix-5 dimensions do not match")
    if _LIBRARY is None:
        if np.any(source > 242):
            raise ValueError("group-128 radix-5 payload is non-canonical")
        groups = count // 128
        values = source.astype(np.uint16).reshape(groups, 26)
        digits = np.empty((groups, 26, 5), dtype=np.uint8)
        remaining = values
        for lane in range(5):
            digits[:, :, lane] = remaining % 3
            remaining = remaining // 3
        flat = digits.reshape(groups, 130)
        if np.any(flat[:, 128:] != 0):
            raise ValueError("group-128 radix-5 padding is non-canonical")
        return (flat[:, :128].reshape(-1).astype(np.int8) - 1).astype(np.int8, copy=False)
    output = np.empty(count, dtype=np.int8)
    status = _LIBRARY.cq3_unpack_group128_radix5(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "group-128 radix-5 unpack")
    return output


def quantize_q8(activation: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    source = _array(activation, np.float32, ndim=1)
    if source.size == 0 or group_size <= 0 or source.size % group_size:
        raise ValueError("activation length must be a nonzero multiple of group_size")
    groups = source.size // group_size
    quantized = np.empty(source.size, dtype=np.int8)
    scales = np.empty(groups, dtype=np.float32)
    sums = np.empty(groups, dtype=np.int32)
    status = _LIBRARY.cq3_quantize_f32_q8(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        source.size,
        group_size,
        quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    _check(status, "A8 quantization")
    return quantized, scales, sums


def expand_q8_radix5(activation_q8: np.ndarray) -> np.ndarray:
    """Expand A8 into 28 eight-byte segments per group for radix-5 gathers."""
    source = _array(activation_q8, np.int8, ndim=1)
    if source.size == 0 or source.size % 128:
        raise ValueError("radix-5 A8 expansion requires a nonzero multiple of 128 values")
    groups = source.size // 128
    output = np.empty(groups * 28 * 8, dtype=np.int8)
    if _LIBRARY is None:
        shaped = np.zeros((groups, 28, 8), dtype=np.int8)
        source_groups = source.reshape(groups, 128)
        for position in range(128):
            shaped[:, position // 5, position % 5] = source_groups[:, position]
        return shaped.reshape(-1)
    status = _LIBRARY.cq3_expand_q8_radix5(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "radix-5 A8 expansion")
    return output


def expand_q8_radix5_planes(activation_q8: np.ndarray) -> np.ndarray:
    """Transpose A8 into five 32-byte radix-5 digit planes per group."""
    source = _array(activation_q8, np.int8, ndim=1)
    if source.size == 0 or source.size % 128:
        raise ValueError("radix-5 plane expansion requires a nonzero multiple of 128 values")
    groups = source.size // 128
    output = np.empty(groups * 5 * 32, dtype=np.int8)
    if _LIBRARY is None:
        shaped = np.zeros((groups, 5, 32), dtype=np.int8)
        source_groups = source.reshape(groups, 128)
        for position in range(128):
            shaped[:, position % 5, position // 5] = source_groups[:, position]
        return shaped.reshape(-1)
    status = _LIBRARY.cq3_expand_q8_radix5_planes(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        source.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    _check(status, "radix-5 plane A8 expansion")
    return output


def gemv_q8_group128_2bit(
    packed_weights: np.ndarray,
    weight_scales: np.ndarray,
    activation_q8: np.ndarray,
    activation_scales: np.ndarray,
    activation_sums: np.ndarray,
    *,
    implementation: str = "avx2",
    threads: int = 1,
    group_size: int = 128,
    executor: ThreadPoolExecutor | None = None,
    cpu_ids: list[int] | None = None,
    validated: bool = False,
) -> np.ndarray:
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    weights = _array(packed_weights, np.uint8, ndim=2)
    scales_u16 = _array(weight_scales, np.float16, ndim=2).view(np.uint16)
    activation = _array(activation_q8, np.int8, ndim=1)
    act_scales = _array(activation_scales, np.float32, ndim=1)
    act_sums = _array(activation_sums, np.int32, ndim=1)
    rows = weights.shape[0]
    columns = activation.size
    groups = columns // group_size if group_size else 0
    if (
        rows == 0 or columns == 0 or group_size != 128 or columns % group_size
        or weights.shape[1] != columns // 4
        or scales_u16.shape != (rows, groups)
        or act_scales.shape != (groups,)
        or act_sums.shape != (groups,)
    ):
        raise ValueError("group-128 GEMV tensor dimensions do not match")
    if not validated and (
        not np.all(np.isfinite(scales_u16.view(np.float16)))
        or not np.all(np.isfinite(act_scales))
    ):
        raise ValueError("group-128 GEMV scales must be finite")
    functions = {
        "scalar": _LIBRARY.cq3_gemv_q8_group128_2bit_scalar,
        "avx2": _LIBRARY.cq3_gemv_q8_group128_2bit_avx2,
        "arm_dotprod": _LIBRARY.cq3_gemv_q8_group128_2bit_arm_dotprod,
    }
    if implementation not in functions:
        raise ValueError(f"unknown GEMV implementation {implementation!r}")
    function = functions[implementation]
    output = np.empty(rows, dtype=np.float32)
    thread_count = min(max(int(threads), 1), rows)
    if cpu_ids is not None and len(cpu_ids) < thread_count:
        raise ValueError("cpu_ids must supply one CPU for every worker")

    def run(row_start: int, row_stop: int, cpu_id: int | None = None) -> None:
        affinity = os.sched_getaffinity(0) if cpu_id is not None and hasattr(os, "sched_getaffinity") else None
        try:
            if affinity is not None:
                os.sched_setaffinity(0, {cpu_id})
            status = function(
                weights[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                scales_u16[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                activation.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                act_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                act_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                row_stop - row_start,
                columns,
                group_size,
                output[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            _check(status, f"{implementation} GEMV")
        finally:
            if affinity is not None:
                os.sched_setaffinity(0, affinity)

    if thread_count == 1:
        run(0, rows, cpu_ids[0] if cpu_ids else None)
    else:
        boundaries = [rows * index // thread_count for index in range(thread_count + 1)]
        owner = executor is None
        pool = executor or ThreadPoolExecutor(max_workers=thread_count)
        try:
            futures = [
                pool.submit(
                    run,
                    boundaries[index],
                    boundaries[index + 1],
                    cpu_ids[index] if cpu_ids else None,
                )
                for index in range(thread_count)
            ]
            for future in futures:
                future.result()
        finally:
            if owner:
                pool.shutdown()
    return output


def decode_row_scale4(
    packed_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    groups: int,
    *,
    implementation: str = "scalar",
    validated: bool = False,
) -> np.ndarray:
    """Decode the Qwen3.8-27B row-anchor plus 4-bit ratio representation."""
    if _LIBRARY is None or not hasattr(_LIBRARY, "cq3_decode_row_scale4_scalar"):
        raise RuntimeError("Clark ternary CPU library lacks row-scale4 support")
    indices = _array(packed_indices, np.uint8, ndim=1)
    anchors_u16 = _array(row_anchors, np.float16, ndim=1).view(np.uint16)
    codebook_u16 = _array(ratio_codebook, np.float16, ndim=1).view(np.uint16)
    rows = anchors_u16.size
    count = rows * int(groups)
    if (
        rows == 0
        or groups <= 0
        or groups > 256
        or indices.size != (count + 1) // 2
        or codebook_u16.shape != (16,)
    ):
        raise ValueError("row-scale4 tensor dimensions do not match")
    if not validated and (
        not np.all(np.isfinite(anchors_u16.view(np.float16)))
        or not np.all(np.isfinite(codebook_u16.view(np.float16)))
        or codebook_u16.view(np.float16)[0] != np.float16(0.0)
        or (count & 1 and int(indices[-1]) & 0xF0)
    ):
        raise ValueError("row-scale4 payload is non-canonical or non-finite")
    functions = {
        "scalar": _LIBRARY.cq3_decode_row_scale4_scalar,
        "avx2": _LIBRARY.cq3_decode_row_scale4_avx2,
    }
    if implementation not in functions:
        raise ValueError(f"unknown row-scale4 implementation {implementation!r}")
    output = np.empty((rows, groups), dtype=np.float32)
    status = functions[implementation](
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        anchors_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        codebook_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        rows,
        groups,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check(status, f"{implementation} row-scale4 decode")
    return output


def gemv_q8_group128_2bit_row_scale4(
    packed_weights: np.ndarray,
    packed_scale_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    activation_q8: np.ndarray,
    activation_scales: np.ndarray,
    activation_sums: np.ndarray,
    *,
    implementation: str = "avx2",
    group_size: int = 128,
    validated: bool = False,
) -> np.ndarray:
    """Aligned-2 GEMV using the exact Qwen3.8-27B row-scale4 payload."""
    if _LIBRARY is None or not hasattr(
        _LIBRARY, "cq3_gemv_q8_group128_2bit_row_scale4_scalar"
    ):
        raise RuntimeError("Clark ternary CPU library lacks row-scale4 GEMV support")
    weights = _array(packed_weights, np.uint8, ndim=2)
    indices = _array(packed_scale_indices, np.uint8, ndim=1)
    anchors_u16 = _array(row_anchors, np.float16, ndim=1).view(np.uint16)
    codebook_u16 = _array(ratio_codebook, np.float16, ndim=1).view(np.uint16)
    activation = _array(activation_q8, np.int8, ndim=1)
    act_scales = _array(activation_scales, np.float32, ndim=1)
    act_sums = _array(activation_sums, np.int32, ndim=1)
    rows = weights.shape[0]
    columns = activation.size
    groups = columns // group_size if group_size else 0
    count = rows * groups
    if (
        rows == 0
        or columns == 0
        or group_size != 128
        or columns % group_size
        or weights.shape[1] != columns // 4
        or anchors_u16.shape != (rows,)
        or codebook_u16.shape != (16,)
        or indices.size != (count + 1) // 2
        or act_scales.shape != (groups,)
        or act_sums.shape != (groups,)
    ):
        raise ValueError("row-scale4 group-128 GEMV dimensions do not match")
    if not validated and (
        not np.all(np.isfinite(anchors_u16.view(np.float16)))
        or not np.all(np.isfinite(codebook_u16.view(np.float16)))
        or not np.all(np.isfinite(act_scales))
        or codebook_u16.view(np.float16)[0] != np.float16(0.0)
        or (count & 1 and int(indices[-1]) & 0xF0)
    ):
        raise ValueError("row-scale4 group-128 GEMV payload is non-canonical")
    functions = {
        "scalar": _LIBRARY.cq3_gemv_q8_group128_2bit_row_scale4_scalar,
        "avx2": _LIBRARY.cq3_gemv_q8_group128_2bit_row_scale4_avx2,
    }
    if implementation not in functions:
        raise ValueError(f"unknown row-scale4 GEMV implementation {implementation!r}")
    output = np.empty(rows, dtype=np.float32)
    status = functions[implementation](
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        anchors_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        codebook_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        activation.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        act_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        act_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        rows,
        columns,
        group_size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check(status, f"{implementation} row-scale4 GEMV")
    return output


def gemv_q8_group128_radix5(
    packed_weights: np.ndarray,
    weight_scales: np.ndarray,
    activation_radix5: np.ndarray,
    activation_scales: np.ndarray,
    activation_sums: np.ndarray,
    *,
    implementation: str = "avx2",
    threads: int = 1,
    group_size: int = 128,
    executor: ThreadPoolExecutor | None = None,
    cpu_ids: list[int] | None = None,
    validated: bool = False,
) -> np.ndarray:
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    weights = _array(packed_weights, np.uint8, ndim=2)
    scales_u16 = _array(weight_scales, np.float16, ndim=2).view(np.uint16)
    activation = _array(activation_radix5, np.int8, ndim=1)
    act_scales = _array(activation_scales, np.float32, ndim=1)
    act_sums = _array(activation_sums, np.int32, ndim=1)
    rows = weights.shape[0]
    groups = act_scales.size
    columns = groups * group_size
    if (
        rows == 0 or groups == 0 or group_size != 128
        or weights.shape[1] != groups * 26
        or scales_u16.shape != (rows, groups)
        or activation.shape != (
            groups * (5 * 32 if implementation == "avx2_planes" else 28 * 8),
        )
        or act_sums.shape != (groups,)
    ):
        raise ValueError("group-128 radix-5 GEMV tensor dimensions do not match")
    if not validated and (
        np.any(weights > 242)
        or not np.all(np.isfinite(scales_u16.view(np.float16)))
        or not np.all(np.isfinite(act_scales))
    ):
        raise ValueError("group-128 radix-5 GEMV inputs must be canonical and finite")
    functions = {
        "scalar": _LIBRARY.cq3_gemv_q8_group128_radix5_scalar,
        "avx2": _LIBRARY.cq3_gemv_q8_group128_radix5_avx2,
        "avx2_nogather": _LIBRARY.cq3_gemv_q8_group128_radix5_avx2_nogather,
        "avx2_nogather_static": _LIBRARY.cq3_gemv_q8_group128_radix5_avx2_nogather_static,
        "avx2_planes": _LIBRARY.cq3_gemv_q8_group128_radix5_avx2_planes,
        "avx2_nogather_pair": _LIBRARY.cq3_gemv_q8_group128_radix5_avx2_nogather_pair,
    }
    if implementation not in functions:
        raise ValueError(f"unknown radix-5 GEMV implementation {implementation!r}")
    function = functions[implementation]
    output = np.empty(rows, dtype=np.float32)
    thread_count = min(max(int(threads), 1), rows)
    if cpu_ids is not None and len(cpu_ids) < thread_count:
        raise ValueError("cpu_ids must supply one CPU for every worker")

    def run(row_start: int, row_stop: int, cpu_id: int | None = None) -> None:
        affinity = os.sched_getaffinity(0) if cpu_id is not None and hasattr(os, "sched_getaffinity") else None
        try:
            if affinity is not None:
                os.sched_setaffinity(0, {cpu_id})
            status = function(
                weights[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                scales_u16[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                activation.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                act_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                act_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                row_stop - row_start,
                columns,
                group_size,
                output[row_start:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            _check(status, f"{implementation} radix-5 GEMV")
        finally:
            if affinity is not None:
                os.sched_setaffinity(0, affinity)

    if thread_count == 1:
        run(0, rows, cpu_ids[0] if cpu_ids else None)
    else:
        boundaries = [rows * index // thread_count for index in range(thread_count + 1)]
        owner = executor is None
        pool = executor or ThreadPoolExecutor(max_workers=thread_count)
        try:
            futures = [
                pool.submit(
                    run,
                    boundaries[index],
                    boundaries[index + 1],
                    cpu_ids[index] if cpu_ids else None,
                )
                for index in range(thread_count)
            ]
            for future in futures:
                future.result()
        finally:
            if owner:
                pool.shutdown()
    return output


def swiglu_f32(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Apply the native FP32 SiLU(gate) * up reference contract."""
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    gate_values = _array(gate, np.float32, ndim=1)
    up_values = _array(up, np.float32, ndim=1)
    if gate_values.shape != up_values.shape or gate_values.size == 0:
        raise ValueError("SwiGLU inputs must have one identical nonempty shape")
    output = np.empty_like(gate_values)
    status = _LIBRARY.cq3_swiglu_f32(
        gate_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        up_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_values.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check(status, "FP32 SwiGLU")
    return output


def gemv_q8_group128_2bit_swiglu(
    gate_packed_weights: np.ndarray,
    gate_weight_scales: np.ndarray,
    up_packed_weights: np.ndarray,
    up_weight_scales: np.ndarray,
    activation_q8: np.ndarray,
    activation_scales: np.ndarray,
    activation_sums: np.ndarray,
    *,
    implementation: str = "arm_dotprod",
    threads: int = 1,
    group_size: int = 128,
    executor: ThreadPoolExecutor | None = None,
    cpu_ids: list[int] | None = None,
    validated: bool = False,
) -> np.ndarray:
    """Fused two-projection ternary GEMV followed by FP32 SwiGLU."""
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    gate_weights = _array(gate_packed_weights, np.uint8, ndim=2)
    up_weights = _array(up_packed_weights, np.uint8, ndim=2)
    gate_scales_u16 = _array(
        gate_weight_scales, np.float16, ndim=2
    ).view(np.uint16)
    up_scales_u16 = _array(
        up_weight_scales, np.float16, ndim=2
    ).view(np.uint16)
    activation = _array(activation_q8, np.int8, ndim=1)
    act_scales = _array(activation_scales, np.float32, ndim=1)
    act_sums = _array(activation_sums, np.int32, ndim=1)
    rows = gate_weights.shape[0]
    columns = activation.size
    groups = columns // group_size if group_size else 0
    if (
        rows == 0 or columns == 0 or group_size != 128
        or columns % group_size
        or gate_weights.shape != (rows, columns // 4)
        or up_weights.shape != gate_weights.shape
        or gate_scales_u16.shape != (rows, groups)
        or up_scales_u16.shape != gate_scales_u16.shape
        or act_scales.shape != (groups,)
        or act_sums.shape != (groups,)
    ):
        raise ValueError("fused group-128 SwiGLU GEMV tensor dimensions do not match")
    if not validated and (
        not np.all(np.isfinite(gate_scales_u16.view(np.float16)))
        or not np.all(np.isfinite(up_scales_u16.view(np.float16)))
        or not np.all(np.isfinite(act_scales))
    ):
        raise ValueError("fused group-128 SwiGLU GEMV scales must be finite")
    functions = {
        "arm_dotprod": (
            _LIBRARY.cq3_gemv_q8_group128_2bit_swiglu_arm_dotprod
        ),
    }
    if implementation not in functions:
        raise ValueError(f"unknown fused GEMV implementation {implementation!r}")
    function = functions[implementation]
    output = np.empty(rows, dtype=np.float32)
    thread_count = min(max(int(threads), 1), rows)
    if cpu_ids is not None and len(cpu_ids) < thread_count:
        raise ValueError("cpu_ids must supply one CPU for every worker")

    def run(row_start: int, row_stop: int, cpu_id: int | None = None) -> None:
        affinity = (
            os.sched_getaffinity(0)
            if cpu_id is not None and hasattr(os, "sched_getaffinity")
            else None
        )
        try:
            if affinity is not None:
                os.sched_setaffinity(0, {cpu_id})
            status = function(
                gate_weights[row_start:].ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint8)
                ),
                gate_scales_u16[row_start:].ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint16)
                ),
                up_weights[row_start:].ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint8)
                ),
                up_scales_u16[row_start:].ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint16)
                ),
                activation.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                act_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                act_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                row_stop - row_start,
                columns,
                group_size,
                output[row_start:].ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
            )
            _check(status, f"{implementation} fused SwiGLU GEMV")
        finally:
            if affinity is not None:
                os.sched_setaffinity(0, affinity)

    if thread_count == 1:
        run(0, rows, cpu_ids[0] if cpu_ids else None)
    else:
        boundaries = [
            rows * index // thread_count for index in range(thread_count + 1)
        ]
        owner = executor is None
        pool = executor or ThreadPoolExecutor(max_workers=thread_count)
        try:
            futures = [
                pool.submit(
                    run,
                    boundaries[index],
                    boundaries[index + 1],
                    cpu_ids[index] if cpu_ids else None,
                )
                for index in range(thread_count)
            ]
            for future in futures:
                future.result()
        finally:
            if owner:
                pool.shutdown()
    return output


def memory_read_sum(
    data: np.ndarray,
    *,
    implementation: str = "avx2",
    threads: int = 1,
    executor: ThreadPoolExecutor | None = None,
    cpu_ids: list[int] | None = None,
) -> int:
    """Read every byte and return a checksum for a STREAM-like bandwidth control."""
    if _LIBRARY is None:
        raise RuntimeError("Clark ternary CPU library is not built")
    source = _array(data, np.uint8).reshape(-1)
    if source.size == 0:
        raise ValueError("bandwidth buffer must not be empty")
    functions = {
        "scalar": _LIBRARY.cq3_sum_u8_scalar,
        "avx2": _LIBRARY.cq3_sum_u8_avx2,
        "neon": _LIBRARY.cq3_sum_u8_neon,
    }
    if implementation not in functions:
        raise ValueError(f"unknown memory-read implementation {implementation!r}")
    function = functions[implementation]
    thread_count = min(max(int(threads), 1), source.size)
    if cpu_ids is not None and len(cpu_ids) < thread_count:
        raise ValueError("cpu_ids must supply one CPU for every worker")

    def run(start: int, stop: int, cpu_id: int | None = None) -> int:
        affinity = os.sched_getaffinity(0) if cpu_id is not None and hasattr(os, "sched_getaffinity") else None
        try:
            if affinity is not None:
                os.sched_setaffinity(0, {cpu_id})
            result = ctypes.c_uint64()
            status = function(
                source[start:].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                stop - start,
                ctypes.byref(result),
            )
            _check(status, f"{implementation} memory read")
            return int(result.value)
        finally:
            if affinity is not None:
                os.sched_setaffinity(0, affinity)

    if thread_count == 1:
        return run(0, source.size, cpu_ids[0] if cpu_ids else None)
    boundaries = [source.size * index // thread_count for index in range(thread_count + 1)]
    owner = executor is None
    pool = executor or ThreadPoolExecutor(max_workers=thread_count)
    try:
        futures = [
            pool.submit(
                run,
                boundaries[index],
                boundaries[index + 1],
                cpu_ids[index] if cpu_ids else None,
            )
            for index in range(thread_count)
        ]
        return sum(future.result() for future in futures)
    finally:
        if owner:
            pool.shutdown()
