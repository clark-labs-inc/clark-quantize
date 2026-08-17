"""Exact enumerative codec for fixed-support Qwen3.8 ternary groups.

A group with exactly ``k`` nonzero ternary values has only
``binomial(n, k) * 2**k`` states: choose the support and then store one sign bit
per supported coordinate.  This is strictly smaller than coding an unrestricted
three-symbol sequence when ``k`` is fixed by the quantizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FixedSupportPayload:
    payload: bytes
    bit_length: int
    rows: int
    columns: int
    group_size: int
    nonzero_per_group: int
    group_width_bits: int


def fixed_support_state_count(group_size: int, nonzero_per_group: int) -> int:
    if group_size <= 0 or not 0 <= nonzero_per_group <= group_size:
        raise ValueError("invalid fixed-support group dimensions")
    return math.comb(group_size, nonzero_per_group) << nonzero_per_group


def fixed_support_width_bits(group_size: int, nonzero_per_group: int) -> int:
    """Return the minimum fixed width that represents every valid group."""
    return (fixed_support_state_count(group_size, nonzero_per_group) - 1).bit_length()


def rank_support(positions: tuple[int, ...], group_size: int) -> int:
    """Rank an ascending support in standard lexicographic combination order."""
    if group_size <= 0 or len(positions) > group_size:
        raise ValueError("invalid support dimensions")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError("support positions must be strictly increasing")
    if positions and (positions[0] < 0 or positions[-1] >= group_size):
        raise ValueError("support position lies outside the group")
    rank = 0
    previous = -1
    remaining = len(positions)
    for position in positions:
        for candidate in range(previous + 1, position):
            rank += math.comb(group_size - candidate - 1, remaining - 1)
        previous = position
        remaining -= 1
    return rank


def unrank_support(rank: int, group_size: int, support_size: int) -> tuple[int, ...]:
    if group_size <= 0 or not 0 <= support_size <= group_size:
        raise ValueError("invalid support dimensions")
    combinations = math.comb(group_size, support_size)
    if rank < 0 or rank >= combinations:
        raise ValueError("support rank lies outside the enumerative codebook")
    positions: list[int] = []
    start = 0
    for remaining in range(support_size, 0, -1):
        for candidate in range(start, group_size - remaining + 1):
            count = math.comb(group_size - candidate - 1, remaining - 1)
            if rank < count:
                positions.append(candidate)
                start = candidate + 1
                break
            rank -= count
        else:  # pragma: no cover - protected by the rank bound
            raise AssertionError("enumerative support unrank failed")
    return tuple(positions)


def encode_fixed_support_group(codes: np.ndarray, nonzero_per_group: int) -> int:
    source = np.asarray(codes)
    if source.ndim != 1 or source.shape[0] == 0:
        raise ValueError("fixed-support group must be a non-empty vector")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("fixed-support group codes must be integers")
    if np.any(source < -1) or np.any(source > 1):
        raise ValueError("fixed-support group codes must lie in {-1, 0, +1}")
    source_i8 = np.asarray(source, dtype=np.int8)
    positions_array = np.flatnonzero(source_i8)
    if positions_array.size != nonzero_per_group:
        raise ValueError("fixed-support group has the wrong nonzero count")
    positions = tuple(int(value) for value in positions_array)
    support_rank = rank_support(positions, source_i8.shape[0])
    sign_bits = 0
    for sign_index, position in enumerate(positions):
        if source_i8[position] > 0:
            sign_bits |= 1 << sign_index
    return (support_rank << nonzero_per_group) | sign_bits


def decode_fixed_support_group(
    encoded: int,
    group_size: int,
    nonzero_per_group: int,
) -> np.ndarray:
    state_count = fixed_support_state_count(group_size, nonzero_per_group)
    if encoded < 0 or encoded >= state_count:
        raise ValueError("encoded fixed-support group is non-canonical")
    sign_mask = (1 << nonzero_per_group) - 1
    signs = encoded & sign_mask
    support_rank = encoded >> nonzero_per_group
    positions = unrank_support(support_rank, group_size, nonzero_per_group)
    output = np.zeros(group_size, dtype=np.int8)
    for sign_index, position in enumerate(positions):
        output[position] = 1 if (signs >> sign_index) & 1 else -1
    return output


def _write_bits(payload: bytearray, cursor: int, value: int, width: int) -> int:
    remaining = width
    while remaining:
        byte_index = cursor >> 3
        bit_offset = cursor & 7
        count = min(remaining, 8 - bit_offset)
        mask = (1 << count) - 1
        payload[byte_index] |= (value & mask) << bit_offset
        value >>= count
        cursor += count
        remaining -= count
    if value:
        raise AssertionError("enumerative group exceeded its frozen bit width")
    return cursor


def _read_bits(payload: bytes, cursor: int, width: int) -> tuple[int, int]:
    remaining = width
    shift = 0
    value = 0
    while remaining:
        byte_index = cursor >> 3
        bit_offset = cursor & 7
        count = min(remaining, 8 - bit_offset)
        mask = (1 << count) - 1
        value |= ((payload[byte_index] >> bit_offset) & mask) << shift
        cursor += count
        shift += count
        remaining -= count
    return value, cursor


def pack_fixed_support_matrix(
    codes: np.ndarray,
    *,
    group_size: int = 128,
    nonzero_per_group: int,
) -> FixedSupportPayload:
    source = np.asarray(codes)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("fixed-support codes must be a non-empty matrix")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("fixed-support matrix codes must be integers")
    if np.any(source < -1) or np.any(source > 1):
        raise ValueError("fixed-support matrix codes must lie in {-1, 0, +1}")
    if source.shape[1] % group_size:
        raise ValueError("fixed-support matrix width must be group aligned")
    source_i8 = np.ascontiguousarray(source, dtype=np.int8)
    groups_per_row = source_i8.shape[1] // group_size
    group_width = fixed_support_width_bits(group_size, nonzero_per_group)
    bit_length = source_i8.shape[0] * groups_per_row * group_width
    payload = bytearray((bit_length + 7) // 8)
    cursor = 0
    grouped = source_i8.reshape(-1, group_size)
    for group in grouped:
        encoded = encode_fixed_support_group(group, nonzero_per_group)
        cursor = _write_bits(payload, cursor, encoded, group_width)
    if cursor != bit_length:
        raise AssertionError("fixed-support payload length changed")
    return FixedSupportPayload(
        payload=bytes(payload),
        bit_length=bit_length,
        rows=source_i8.shape[0],
        columns=source_i8.shape[1],
        group_size=group_size,
        nonzero_per_group=nonzero_per_group,
        group_width_bits=group_width,
    )


def unpack_fixed_support_matrix(encoded: FixedSupportPayload) -> np.ndarray:
    if encoded.group_size <= 0:
        raise ValueError("fixed-support payload group size is invalid")
    expected_groups = encoded.rows * (encoded.columns // encoded.group_size)
    expected_bits = expected_groups * encoded.group_width_bits
    if (
        encoded.rows <= 0
        or encoded.columns <= 0
        or encoded.columns % encoded.group_size
        or encoded.group_width_bits
        != fixed_support_width_bits(encoded.group_size, encoded.nonzero_per_group)
        or encoded.bit_length != expected_bits
        or len(encoded.payload) != (encoded.bit_length + 7) // 8
    ):
        raise ValueError("fixed-support payload metadata is inconsistent")
    if encoded.bit_length % 8:
        valid = encoded.bit_length % 8
        if encoded.payload[-1] >> valid:
            raise ValueError("fixed-support payload has nonzero padding bits")
    output = np.empty((expected_groups, encoded.group_size), dtype=np.int8)
    cursor = 0
    for index in range(expected_groups):
        value, cursor = _read_bits(encoded.payload, cursor, encoded.group_width_bits)
        output[index] = decode_fixed_support_group(
            value, encoded.group_size, encoded.nonzero_per_group
        )
    if cursor != encoded.bit_length:
        raise AssertionError("fixed-support decode length changed")
    return output.reshape(encoded.rows, encoded.columns)


def exact_hash_residual_component_bits_per_weight(
    *,
    input_dim: int,
    sketch_dim: int,
    nonzero_per_group: int,
    group_size: int = 128,
    hash_bits: int = 4,
    scale_index_bits: int = 4,
) -> float:
    """Count exact fixed-support codes plus every per-row side channel.

    The two 16-bit terms are the ternary row anchor and the INT4 coefficient
    row scale.  A tensor-level 16-entry FP16 ratio codebook is intentionally
    excluded here and must be added once per complete tensor by the model
    storage ledger.
    """
    if input_dim <= 0 or sketch_dim <= 0 or input_dim % group_size:
        raise ValueError("hash-residual component dimensions are invalid")
    return (
        fixed_support_width_bits(group_size, nonzero_per_group) / group_size
        + hash_bits * sketch_dim / input_dim
        + scale_index_bits / group_size
        + 16 / input_dim
        + 16 / input_dim
    )
