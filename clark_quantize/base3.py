"""Exact streaming arithmetic packing for ternary {-1, 0, +1} codes.

The uniform arithmetic model approaches log2(3) payload bits per trit without
requiring one unbounded base-3 integer. The framed representation records the
exact trit and bit counts and protects the payload with CRC32.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct
import zlib

import numpy as np


_MAGIC = b"CQ3A1"
_HEADER = struct.Struct(">5sQQI")
_STATE_BITS = 32
_FULL = 1 << _STATE_BITS
_MASK = _FULL - 1
_HALF = _FULL >> 1
_QUARTER = _HALF >> 1
_THREE_QUARTER = _QUARTER * 3


def _load_native_decoder() -> ctypes.CDLL | None:
    explicit = os.environ.get("CLARK_BASE3_NATIVE_LIB")
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).with_name("_base3_native.so"),
        Path(__file__).with_name("_base3_native.dylib"),
    ]
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError:
            continue
        function = library.cq3_decode_uniform
        function.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int8),
        )
        function.restype = ctypes.c_int
        return library
    return None


_NATIVE_DECODER = _load_native_decoder()


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.current = 0
        self.used = 0
        self.bit_length = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | int(bit)
        self.used += 1
        self.bit_length += 1
        if self.used == 8:
            self.data.append(self.current)
            self.current = 0
            self.used = 0

    def finish(self) -> tuple[bytes, int]:
        if self.used:
            self.data.append(self.current << (8 - self.used))
        return bytes(self.data), self.bit_length


class _BitReader:
    def __init__(self, data: bytes, bit_length: int) -> None:
        self.data = data
        self.bit_length = bit_length
        self.position = 0

    def read(self) -> int:
        if self.position >= self.bit_length:
            return 0
        byte = self.data[self.position >> 3]
        bit = (byte >> (7 - (self.position & 7))) & 1
        self.position += 1
        return bit


def _codes_array(codes: np.ndarray) -> np.ndarray:
    array = np.asarray(codes, dtype=np.int8).reshape(-1)
    if array.size and (array.min() < -1 or array.max() > 1):
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    return array


def encode_payload(codes: np.ndarray) -> tuple[bytes, int]:
    """Encode codes into a uniform arithmetic payload and exact bit count."""
    array = _codes_array(codes)
    writer = _BitWriter()
    low = 0
    high = _MASK
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
        low = old_low + width * symbol // 3
        high = old_low + width * (symbol + 1) // 3 - 1
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1

    pending += 1
    emit(0 if low < _QUARTER else 1)
    return writer.finish()


def _decode_payload_python(payload: bytes, bit_length: int, count: int) -> np.ndarray:
    reader = _BitReader(payload, bit_length)
    low = 0
    high = _MASK
    value = 0
    for _ in range(_STATE_BITS):
        value = ((value << 1) & _MASK) | reader.read()

    output = np.empty(count, dtype=np.int8)
    for index in range(count):
        width = high - low + 1
        scaled = ((value - low + 1) * 3 - 1) // width
        symbol = min(max(int(scaled), 0), 2)
        output[index] = symbol - 1
        old_low = low
        low = old_low + width * symbol // 3
        high = old_low + width * (symbol + 1) // 3 - 1
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
            value = ((value << 1) & _MASK) | reader.read()
    return output


def native_decoder_available() -> bool:
    return _NATIVE_DECODER is not None


def decode_payload(payload: bytes, bit_length: int, count: int) -> np.ndarray:
    """Decode an arithmetic payload into exactly ``count`` int8 ternary codes."""
    if count < 0 or bit_length < 0 or bit_length > len(payload) * 8:
        raise ValueError("invalid arithmetic payload dimensions")
    if _NATIVE_DECODER is None:
        return _decode_payload_python(payload, bit_length, count)
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
    result = _NATIVE_DECODER.cq3_decode_uniform(
        payload_pointer,
        bit_length,
        count,
        output_pointer,
    )
    if result != 0:  # pragma: no cover
        raise ValueError(f"native Clark base-3 decoder failed with status {result}")
    return output


def pack_base3(codes: np.ndarray) -> bytes:
    """Return a self-describing, CRC-protected Clark base-3 frame."""
    array = _codes_array(codes)
    payload, bit_length = encode_payload(array)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(_MAGIC, array.size, bit_length, checksum) + payload


def unpack_base3(frame: bytes) -> np.ndarray:
    """Validate and decode a frame produced by :func:`pack_base3`."""
    if len(frame) < _HEADER.size:
        raise ValueError("truncated Clark base-3 frame")
    magic, count, bit_length, checksum = _HEADER.unpack_from(frame)
    if magic != _MAGIC:
        raise ValueError("invalid Clark base-3 magic")
    payload_bytes = (bit_length + 7) // 8
    if len(frame) != _HEADER.size + payload_bytes:
        raise ValueError("Clark base-3 frame length mismatch")
    payload = frame[_HEADER.size :]
    if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
        raise ValueError("Clark base-3 payload checksum mismatch")
    return decode_payload(payload, bit_length, count)


def frame_metadata(frame: bytes) -> dict[str, int]:
    if len(frame) < _HEADER.size:
        raise ValueError("truncated Clark base-3 frame")
    magic, count, bit_length, checksum = _HEADER.unpack_from(frame)
    if magic != _MAGIC:
        raise ValueError("invalid Clark base-3 magic")
    return {
        "trits": count,
        "payload_bits": bit_length,
        "payload_bytes": (bit_length + 7) // 8,
        "frame_bytes": len(frame),
        "header_bytes": _HEADER.size,
        "crc32": checksum,
    }
