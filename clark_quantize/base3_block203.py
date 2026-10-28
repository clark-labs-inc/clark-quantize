"""Canonical 203-bit blocks for exact groups of 128 ternary symbols.

The layout is a bitstream of independent base-3 integers. A complete group of
128 trits occupies exactly 203 bits because ``3**128 < 2**203``. The last group
uses ``ceil(remainder * log2(3))`` bits. Tensor metadata supplies row lengths;
integrity hashes belong to the enclosing artifact format.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


GROUP_TRITS = 128
GROUP_BITS = 203
LOG2_3 = math.log2(3.0)


def bits_for_trits(count: int) -> int:
    if count < 0:
        raise ValueError("trit count must be non-negative")
    if count == 0:
        return 0
    if count == GROUP_TRITS:
        return GROUP_BITS
    return math.ceil(count * LOG2_3)


def row_payload_bits(count: int) -> int:
    if count < 0:
        raise ValueError("trit count must be non-negative")
    groups, remainder = divmod(count, GROUP_TRITS)
    return groups * GROUP_BITS + bits_for_trits(remainder)


def row_payload_bytes(count: int) -> int:
    return math.ceil(row_payload_bits(count) / 8)


def matrix_payload_bits(rows: int, columns: int) -> int:
    if rows < 0 or columns < 0:
        raise ValueError("matrix dimensions must be non-negative")
    if (rows == 0) != (columns == 0):
        raise ValueError("matrix dimensions must both be zero or both be positive")
    return rows * row_payload_bits(columns)


def matrix_payload_bytes(rows: int, columns: int) -> int:
    return math.ceil(matrix_payload_bits(rows, columns) / 8)


def _codes(codes: np.ndarray | Iterable[int]) -> np.ndarray:
    source = np.asarray(codes).reshape(-1)
    if source.size and not np.all((source == -1) | (source == 0) | (source == 1)):
        raise ValueError("ternary codes must be in {-1, 0, +1}")
    return source.astype(np.int8, copy=False)


def _encode_group(codes: np.ndarray) -> int:
    value = 0
    multiplier = 1
    for code in codes:
        value += (int(code) + 1) * multiplier
        multiplier *= 3
    return value


def _decode_group(value: int, count: int) -> np.ndarray:
    if value < 0 or value >= 3**count:
        raise ValueError("non-canonical base-3 codeword")
    result = np.empty(count, dtype=np.int8)
    for index in range(count):
        value, digit = divmod(value, 3)
        result[index] = digit - 1
    if value:
        raise AssertionError("base-3 decoder did not consume the codeword")
    return result


def pack_row(codes: np.ndarray | Iterable[int]) -> bytes:
    array = _codes(codes)
    output = bytearray()
    accumulator = 0
    accumulator_bits = 0
    for start in range(0, array.size, GROUP_TRITS):
        group = array[start : start + GROUP_TRITS]
        width = bits_for_trits(group.size)
        value = _encode_group(group)
        if value >= 1 << width:
            raise AssertionError("base-3 codeword does not fit its frozen width")
        accumulator |= value << accumulator_bits
        accumulator_bits += width
        while accumulator_bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            accumulator_bits -= 8
    if accumulator_bits:
        output.append(accumulator)
    expected = row_payload_bytes(array.size)
    if len(output) != expected:
        raise AssertionError(f"encoded {len(output)} bytes; expected {expected}")
    return bytes(output)


def unpack_row(payload: bytes | bytearray | memoryview, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("trit count must be non-negative")
    content = bytes(payload)
    expected = row_payload_bytes(count)
    if len(content) != expected:
        raise ValueError(f"base-3 row length is {len(content)} bytes; expected {expected}")
    meaningful_bits = row_payload_bits(count)
    unused_bits = len(content) * 8 - meaningful_bits
    if unused_bits and content and content[-1] >> (8 - unused_bits):
        raise ValueError("non-zero padding bits in base-3 row")

    output = np.empty(count, dtype=np.int8)
    accumulator = 0
    accumulator_bits = 0
    cursor = 0
    output_cursor = 0
    while output_cursor < count:
        group_count = min(GROUP_TRITS, count - output_cursor)
        width = bits_for_trits(group_count)
        while accumulator_bits < width:
            accumulator |= content[cursor] << accumulator_bits
            accumulator_bits += 8
            cursor += 1
        mask = (1 << width) - 1
        value = accumulator & mask
        accumulator >>= width
        accumulator_bits -= width
        output[output_cursor : output_cursor + group_count] = _decode_group(value, group_count)
        output_cursor += group_count
    if accumulator or cursor != len(content):
        raise ValueError("base-3 row has trailing non-canonical data")
    return output


def _copy_bits(
    source: bytes | bytearray | memoryview,
    source_bits: int,
    destination: bytearray,
    destination_cursor: int,
) -> int:
    source_cursor = 0
    while source_cursor < source_bits:
        source_shift = source_cursor % 8
        destination_shift = destination_cursor % 8
        take = min(
            source_bits - source_cursor,
            8 - source_shift,
            8 - destination_shift,
        )
        mask = (1 << take) - 1
        value = (source[source_cursor // 8] >> source_shift) & mask
        destination[destination_cursor // 8] |= value << destination_shift
        source_cursor += take
        destination_cursor += take
    return destination_cursor


def pack_rows(codes: np.ndarray | Iterable[Iterable[int]]) -> bytes:
    """Pack a matrix with per-row group resets and no padding between rows."""
    values = np.asarray(codes)
    if values.ndim != 2:
        raise ValueError("base-3 matrix codes must be two-dimensional")
    rows, columns = values.shape
    if rows == 0 or columns == 0:
        raise ValueError("base-3 matrix dimensions must be positive")
    output = bytearray(matrix_payload_bytes(rows, columns))
    cursor = 0
    row_bits = row_payload_bits(columns)
    for row in values:
        cursor = _copy_bits(pack_row(row), row_bits, output, cursor)
    if cursor != matrix_payload_bits(rows, columns):
        raise AssertionError("base-3 matrix encoder emitted an unexpected bit count")
    return bytes(output)


def unpack_rows(
    payload: bytes | bytearray | memoryview,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Decode a bit-contiguous matrix whose base-3 groups reset every row."""
    expected = matrix_payload_bytes(rows, columns)
    content = bytes(payload)
    if rows <= 0 or columns <= 0 or len(content) != expected:
        raise ValueError("base-3 matrix dimensions or payload length do not match")
    meaningful_bits = matrix_payload_bits(rows, columns)
    unused_bits = len(content) * 8 - meaningful_bits
    if unused_bits and content[-1] >> (8 - unused_bits):
        raise ValueError("non-zero padding bits in base-3 matrix")
    output = np.empty((rows, columns), dtype=np.int8)
    row_bits = row_payload_bits(columns)
    source_cursor = 0
    for row in range(rows):
        row_payload = bytearray(row_payload_bytes(columns))
        destination_cursor = 0
        while destination_cursor < row_bits:
            source_shift = source_cursor % 8
            destination_shift = destination_cursor % 8
            take = min(
                row_bits - destination_cursor,
                8 - source_shift,
                8 - destination_shift,
            )
            mask = (1 << take) - 1
            value = (content[source_cursor // 8] >> source_shift) & mask
            row_payload[destination_cursor // 8] |= value << destination_shift
            source_cursor += take
            destination_cursor += take
        output[row] = unpack_row(row_payload, columns)
    if source_cursor != meaningful_bits:
        raise AssertionError("base-3 matrix decoder consumed an unexpected bit count")
    return output


def metadata(count: int) -> dict[str, int | float]:
    bits = row_payload_bits(count)
    return {
        "trits": count,
        "groups": math.ceil(count / GROUP_TRITS) if count else 0,
        "payload_bits": bits,
        "payload_bytes": row_payload_bytes(count),
        "payload_bits_per_trit": bits / count if count else 0.0,
    }
