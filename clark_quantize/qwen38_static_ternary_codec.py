"""Exact two-pass static arithmetic coding for nonuniform Qwen ternary codes.

The existing Clark ``CQ3A1`` codec deliberately uses a uniform ternary model
and approaches ``log2(3)``.  ``CQ3S1`` stores one deterministic 15-bit
frequency table per independently decodable frame.  It preserves every trit
exactly while approaching the empirical entropy of a frozen model's
nonuniform codes.  This is a distribution artifact codec; the CPU hot path
still transcodes to an ISA-specific compute image.
"""
from __future__ import annotations

import ctypes
import struct
import zlib

import numpy as np

from . import base3


MAGIC = b"CQ3S1"
FREQUENCY_TOTAL = 1 << 15
_PREFIX = struct.Struct(">5sIIHHH")
_HEADER = struct.Struct(">5sIIHHHI")


def _load_native_decoder() -> ctypes._CFuncPtr | None:
    library = base3._NATIVE_DECODER
    if library is None:
        return None
    try:
        function = library.cq3_decode_static
    except AttributeError:
        return None
    function.argtypes = (
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int8),
    )
    function.restype = ctypes.c_int
    return function


_NATIVE_DECODER = _load_native_decoder()


def _load_native_encoder() -> ctypes._CFuncPtr | None:
    library = base3._NATIVE_DECODER
    if library is None:
        return None
    try:
        function = library.cq3_encode_static
    except AttributeError:
        return None
    function.argtypes = (
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    )
    function.restype = ctypes.c_int
    return function


_NATIVE_ENCODER = _load_native_encoder()


def quantized_frequencies(codes: np.ndarray) -> tuple[int, int, int]:
    """Return deterministic positive-on-support frequencies summing to 32768."""
    array = base3._codes_array(codes)
    counts = np.bincount(array.astype(np.int16) + 1, minlength=3).astype(np.int64)
    if array.size == 0:
        return (10923, 10923, 10922)
    active = counts > 0
    active_count = int(active.sum())
    remaining = FREQUENCY_TOTAL - active_count
    frequencies = active.astype(np.int64)
    numerators = counts * remaining
    frequencies += numerators // array.size
    missing = FREQUENCY_TOTAL - int(frequencies.sum())
    remainders = numerators % array.size
    order = sorted(range(3), key=lambda index: (-int(remainders[index]), index))
    for index in order[:missing]:
        frequencies[index] += 1
    result = tuple(int(value) for value in frequencies)
    if sum(result) != FREQUENCY_TOTAL or any(
        (count > 0) != (frequency > 0)
        for count, frequency in zip(counts, result)
    ):
        raise AssertionError("static ternary frequency apportionment changed")
    return result


def _validate_frequencies(frequencies: tuple[int, int, int]) -> None:
    if len(frequencies) != 3 or any(value < 0 or value > 0xFFFF for value in frequencies):
        raise ValueError("static ternary frequencies must be three uint16 values")
    if sum(frequencies) != FREQUENCY_TOTAL:
        raise ValueError("static ternary frequencies must sum to 32768")


def _encode_payload_python(
    codes: np.ndarray,
    frequencies: tuple[int, int, int],
) -> tuple[bytes, int]:
    array = base3._codes_array(codes)
    cumulative = (0, frequencies[0], frequencies[0] + frequencies[1], FREQUENCY_TOTAL)
    writer = base3._BitWriter()
    low = 0
    high = base3._MASK
    pending = 0

    def emit(bit: int) -> None:
        nonlocal pending
        writer.write(bit)
        inverse = 1 - bit
        for _ in range(pending):
            writer.write(inverse)
        pending = 0

    for code in array:
        symbol = int(code) + 1
        width = high - low + 1
        old_low = low
        low = old_low + width * cumulative[symbol] // FREQUENCY_TOTAL
        high = old_low + width * cumulative[symbol + 1] // FREQUENCY_TOTAL - 1
        while True:
            if high < base3._HALF:
                emit(0)
            elif low >= base3._HALF:
                emit(1)
                low -= base3._HALF
                high -= base3._HALF
            elif low >= base3._QUARTER and high < base3._THREE_QUARTER:
                pending += 1
                low -= base3._QUARTER
                high -= base3._QUARTER
            else:
                break
            low = (low << 1) & base3._MASK
            high = ((high << 1) & base3._MASK) | 1
    pending += 1
    emit(0 if low < base3._QUARTER else 1)
    return writer.finish()


def _encode_payload_native(
    codes: np.ndarray,
    frequencies: tuple[int, int, int],
) -> tuple[bytes, int]:
    if _NATIVE_ENCODER is None:
        raise RuntimeError("native static ternary encoder is unavailable")
    array = np.ascontiguousarray(base3._codes_array(codes))
    capacity_bytes = max(8, (15 * int(array.size) + 66 + 7) // 8)
    output = np.zeros(capacity_bytes, dtype=np.uint8)
    bit_length = ctypes.c_uint64()
    source_pointer = (
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if array.size
        else ctypes.POINTER(ctypes.c_int8)()
    )
    status = _NATIVE_ENCODER(
        source_pointer,
        int(array.size),
        *frequencies,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        capacity_bytes,
        ctypes.byref(bit_length),
    )
    if status != 0:  # pragma: no cover
        raise ValueError(f"native static ternary encoder failed with status {status}")
    payload_bytes = (bit_length.value + 7) // 8
    return output[:payload_bytes].tobytes(), int(bit_length.value)


def encode_payload(
    codes: np.ndarray,
    frequencies: tuple[int, int, int] | None = None,
) -> tuple[bytes, int, tuple[int, int, int]]:
    """Encode exact trits with a deterministic frozen frequency table."""
    array = base3._codes_array(codes)
    frequencies = frequencies or quantized_frequencies(array)
    _validate_frequencies(frequencies)
    counts = np.bincount(array.astype(np.int16) + 1, minlength=3)
    if any(count and frequency == 0 for count, frequency in zip(counts, frequencies)):
        raise ValueError("static frequency table assigns zero mass to an observed symbol")
    if _NATIVE_ENCODER is None:
        payload, bit_length = _encode_payload_python(array, frequencies)
    else:
        payload, bit_length = _encode_payload_native(array, frequencies)
    return payload, bit_length, frequencies


def _decode_payload_python(
    payload: bytes,
    bit_length: int,
    count: int,
    frequencies: tuple[int, int, int],
) -> np.ndarray:
    _validate_frequencies(frequencies)
    reader = base3._BitReader(payload, bit_length)
    cumulative_zero = frequencies[0]
    cumulative_plus = cumulative_zero + frequencies[1]
    cumulative = (0, cumulative_zero, cumulative_plus, FREQUENCY_TOTAL)
    low = 0
    high = base3._MASK
    value = 0
    for _ in range(base3._STATE_BITS):
        value = ((value << 1) & base3._MASK) | reader.read()
    output = np.empty(count, dtype=np.int8)
    for index in range(count):
        width = high - low + 1
        scaled = ((value - low + 1) * FREQUENCY_TOTAL - 1) // width
        symbol = 0 if scaled < cumulative_zero else 1 if scaled < cumulative_plus else 2
        if frequencies[symbol] == 0:
            raise ValueError("static arithmetic payload selected a zero-frequency symbol")
        output[index] = symbol - 1
        old_low = low
        low = old_low + width * cumulative[symbol] // FREQUENCY_TOTAL
        high = old_low + width * cumulative[symbol + 1] // FREQUENCY_TOTAL - 1
        while True:
            if high < base3._HALF:
                pass
            elif low >= base3._HALF:
                low -= base3._HALF
                high -= base3._HALF
                value -= base3._HALF
            elif low >= base3._QUARTER and high < base3._THREE_QUARTER:
                low -= base3._QUARTER
                high -= base3._QUARTER
                value -= base3._QUARTER
            else:
                break
            low = (low << 1) & base3._MASK
            high = ((high << 1) & base3._MASK) | 1
            value = ((value << 1) & base3._MASK) | reader.read()
    return output


def decode_payload(
    payload: bytes,
    bit_length: int,
    count: int,
    frequencies: tuple[int, int, int],
) -> np.ndarray:
    if count < 0 or bit_length < 0 or bit_length > len(payload) * 8:
        raise ValueError("invalid static arithmetic payload dimensions")
    _validate_frequencies(frequencies)
    if _NATIVE_DECODER is None:
        return _decode_payload_python(payload, bit_length, count, frequencies)
    payload_array = np.frombuffer(payload, dtype=np.uint8)
    output = np.empty(count, dtype=np.int8)
    payload_pointer = (
        payload_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if payload_array.size
        else ctypes.POINTER(ctypes.c_uint8)()
    )
    output_pointer = (
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if output.size
        else ctypes.POINTER(ctypes.c_int8)()
    )
    status = _NATIVE_DECODER(
        payload_pointer, bit_length, count, *frequencies, output_pointer
    )
    if status != 0:  # pragma: no cover
        raise ValueError(f"native static ternary decoder failed with status {status}")
    return output


def pack(codes: np.ndarray) -> bytes:
    array = base3._codes_array(codes)
    payload, bit_length, frequencies = encode_payload(array)
    if array.size > 0xFFFFFFFF or bit_length > 0xFFFFFFFF:
        raise ValueError("CQ3S1 frame exceeds its uint32 count or bit-length field")
    prefix = _PREFIX.pack(MAGIC, array.size, bit_length, *frequencies)
    checksum = zlib.crc32(prefix + payload) & 0xFFFFFFFF
    return prefix + struct.pack(">I", checksum) + payload


def unpack(frame: bytes) -> np.ndarray:
    if len(frame) < _HEADER.size:
        raise ValueError("truncated CQ3S1 frame")
    magic, count, bit_length, f_minus, f_zero, f_plus, checksum = _HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise ValueError("invalid CQ3S1 magic")
    payload_bytes = (bit_length + 7) // 8
    if len(frame) != _HEADER.size + payload_bytes:
        raise ValueError("CQ3S1 frame length mismatch")
    prefix = frame[: _PREFIX.size]
    payload = frame[_HEADER.size :]
    if zlib.crc32(prefix + payload) & 0xFFFFFFFF != checksum:
        raise ValueError("CQ3S1 frame checksum mismatch")
    return decode_payload(payload, bit_length, count, (f_minus, f_zero, f_plus))


def frame_metadata(frame: bytes) -> dict[str, object]:
    if len(frame) < _HEADER.size:
        raise ValueError("truncated CQ3S1 frame")
    magic, count, bit_length, f_minus, f_zero, f_plus, checksum = _HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise ValueError("invalid CQ3S1 magic")
    return {
        "trits": count,
        "payload_bits": bit_length,
        "payload_bytes": (bit_length + 7) // 8,
        "frame_bytes": len(frame),
        "header_bytes": _HEADER.size,
        "frequencies": [f_minus, f_zero, f_plus],
        "crc32": checksum,
    }


def native_decoder_available() -> bool:
    return _NATIVE_DECODER is not None


def native_encoder_available() -> bool:
    return _NATIVE_ENCODER is not None
