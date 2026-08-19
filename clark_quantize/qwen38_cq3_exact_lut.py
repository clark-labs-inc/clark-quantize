"""Exact group-of-four LUT algebra for the Qwen3.8 CQ3 CPU representation.

This module is a small mathematical reference, not an inference runtime.  It
expresses the existing ternary ``CQ3 x CQ8`` integer dot as two subset-table
lookups per four weights.  A native AVX2 implementation can therefore test a
T-MAC-like lookup schedule without changing the trained weights, activation
quantizer, group scales, or integer result.
"""

from __future__ import annotations

import numpy as np


GROUP = 128
QUARTET = 4
SUBSETS = 1 << QUARTET


def pack_ternary_bitplane_quartets(codes: np.ndarray) -> np.ndarray:
    """Pack ``{-1, 0, +1}`` codes into one byte per four weights.

    The low nibble is the four-bit mask for bit zero of ``code + 1`` and the
    high nibble is the corresponding bit-one mask.  Four ternary weights
    consume one byte, exactly matching the existing two-bit CQ3 payload.
    """

    values = np.asarray(codes)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("CQ3 LUT codes require a nonempty final dimension")
    if values.shape[-1] % QUARTET:
        raise ValueError("CQ3 LUT codes must contain complete quartets")
    if not np.issubdtype(values.dtype, np.signedinteger):
        raise TypeError("CQ3 LUT codes must use a signed integer dtype")
    if not bool(np.isin(values, (-1, 0, 1)).all()):
        raise ValueError("CQ3 LUT codes must be ternary")

    digits = (values.astype(np.int16, copy=False) + 1).astype(np.uint8)
    grouped = digits.reshape(*digits.shape[:-1], -1, QUARTET)
    shifts = np.arange(QUARTET, dtype=np.uint8)
    low = np.sum((grouped & 1) << shifts, axis=-1, dtype=np.uint16)
    high = np.sum(((grouped >> 1) & 1) << shifts, axis=-1, dtype=np.uint16)
    return (low | (high << 4)).astype(np.uint8)


def build_int16_subset_lut(activations: np.ndarray) -> np.ndarray:
    """Build exact signed-INT16 subset sums for CQ8 activation quartets."""

    values = np.asarray(activations)
    if values.ndim != 1 or values.size == 0 or values.size % QUARTET:
        raise ValueError("CQ8 LUT activations require complete quartets")
    if values.dtype != np.int8:
        raise TypeError("CQ8 LUT activations must be int8")

    grouped = values.astype(np.int16).reshape(-1, QUARTET)
    masks = np.arange(SUBSETS, dtype=np.uint8)
    membership = ((masks[:, None] >> np.arange(QUARTET)) & 1).astype(np.int16)
    table = grouped @ membership.T
    if table.dtype != np.int16:
        table = table.astype(np.int16)
    return table


def exact_lut_group_dot(
    packed_weights: np.ndarray,
    subset_lut: np.ndarray,
    activation_sum: int,
) -> np.ndarray:
    """Evaluate the exact ternary integer dot for one activation group.

    For a digit ``d = code + 1 = bit0 + 2*bit1``, the original signed weight
    is ``d - 1``.  Summing lookup values therefore gives

    ``L(bit0) + 2*L(bit1) - sum(activations)``.
    """

    packed = np.asarray(packed_weights)
    table = np.asarray(subset_lut)
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise TypeError("packed CQ3 LUT weights must be a uint8 row matrix")
    if table.ndim != 2 or table.dtype != np.int16 or table.shape[1] != SUBSETS:
        raise TypeError("subset LUT must have shape [quartets, 16] and int16 dtype")
    if packed.shape[1] != table.shape[0]:
        raise ValueError("weight and activation quartet counts differ")
    if not np.iinfo(np.int16).min <= activation_sum <= np.iinfo(np.int16).max:
        raise ValueError("CQ8 activation sum does not fit int16")

    quartet = np.arange(packed.shape[1])
    bit0 = packed & 0x0F
    bit1 = packed >> 4
    selected0 = table[quartet[None, :], bit0]
    selected1 = table[quartet[None, :], bit1]
    return (
        selected0.astype(np.int32).sum(axis=1)
        + 2 * selected1.astype(np.int32).sum(axis=1)
        - int(activation_sum)
    ).astype(np.int32)


def direct_ternary_group_dot(
    codes: np.ndarray, activations: np.ndarray
) -> np.ndarray:
    """Reference the ordinary signed ternary dot for parity tests."""

    weights = np.asarray(codes)
    values = np.asarray(activations)
    if weights.ndim != 2 or values.ndim != 1:
        raise ValueError("direct CQ3 dot expects a weight matrix and activation row")
    if weights.shape[1] != values.size:
        raise ValueError("direct CQ3 dimensions differ")
    if weights.shape[1] != GROUP:
        raise ValueError("the frozen CQ3 scale group is 128")
    if values.dtype != np.int8 or not np.issubdtype(weights.dtype, np.signedinteger):
        raise TypeError("direct CQ3 dot expects signed ternary and int8 inputs")
    if not bool(np.isin(weights, (-1, 0, 1)).all()):
        raise ValueError("direct CQ3 weights must be ternary")
    return weights.astype(np.int32) @ values.astype(np.int32)
