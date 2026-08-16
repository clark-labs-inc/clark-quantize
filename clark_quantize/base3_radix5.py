"""Fast canonical five-trits-per-byte packing for ternary codes.

This format is exactly reversible and vectorized, but its payload rate is
1.6 bits/trit rather than the entropy limit log2(3).  It is a deployment
candidate and must not be reported as ideal base-3 arithmetic packing.
"""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np


MAGIC = b"CQ3R5"
HEADER = struct.Struct(">5sQI")
COUNT = struct.Struct(">Q")
POWERS = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)


def _codes(codes: np.ndarray) -> np.ndarray:
    array = np.asarray(codes, dtype=np.int8).reshape(-1)
    if array.size and (array.min() < -1 or array.max() > 1):
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    return array


def pack_radix5(codes: np.ndarray) -> bytes:
    array = _codes(codes)
    digits = array.astype(np.int16, copy=False) + 1
    padding = (-digits.size) % 5
    if padding:
        digits = np.pad(digits, (0, padding), constant_values=0)
    payload = ((digits.reshape(-1, 5).astype(np.uint16) * POWERS).sum(axis=1)).astype(np.uint8).tobytes()
    checksum = zlib.crc32(COUNT.pack(array.size) + payload) & 0xFFFFFFFF
    return HEADER.pack(MAGIC, array.size, checksum) + payload


def unpack_radix5(frame: bytes) -> np.ndarray:
    if len(frame) < HEADER.size:
        raise ValueError("truncated Clark radix-5 frame")
    magic, count, checksum = HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise ValueError("invalid Clark radix-5 magic")
    payload = frame[HEADER.size :]
    expected = math.ceil(count / 5)
    if len(payload) != expected:
        raise ValueError("Clark radix-5 frame length mismatch")
    if zlib.crc32(COUNT.pack(count) + payload) & 0xFFFFFFFF != checksum:
        raise ValueError("Clark radix-5 payload checksum mismatch")
    values = np.frombuffer(payload, dtype=np.uint8).astype(np.uint16)
    if values.size and np.any(values > 242):
        raise ValueError("Clark radix-5 payload contains a non-canonical byte")
    digits = np.empty((values.size, 5), dtype=np.int8)
    remaining = values
    for column in range(5):
        digits[:, column] = (remaining % 3).astype(np.int8)
        remaining = remaining // 3
    flat = digits.reshape(-1)
    if flat.size > count and np.any(flat[count:] != 0):
        raise ValueError("Clark radix-5 padding is non-canonical")
    return (flat[:count] - 1).astype(np.int8, copy=False)


def radix5_metadata(frame: bytes) -> dict[str, int | float]:
    if len(frame) < HEADER.size:
        raise ValueError("truncated Clark radix-5 frame")
    magic, count, checksum = HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise ValueError("invalid Clark radix-5 magic")
    return {
        "trits": count,
        "payload_bytes": len(frame) - HEADER.size,
        "frame_bytes": len(frame),
        "header_bytes": HEADER.size,
        "crc32": checksum,
        "payload_bits_per_trit": (len(frame) - HEADER.size) * 8 / max(count, 1),
    }
