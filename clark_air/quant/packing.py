"""Offline packing helpers for Clark Air artifacts."""

from __future__ import annotations

import numpy as np

POW3_40 = np.array([3**i for i in range(40)], dtype=np.uint64)


def pack_trits_40(codes: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Pack {-1,0,+1} trits into uint64 chunks, 40 trits per word."""

    flat = np.asarray(codes, dtype=np.int8).reshape(-1)
    if not np.isin(flat, [-1, 0, 1]).all():
        raise ValueError("trit codes must be in {-1, 0, +1}")
    trits = flat.astype(np.int16) + 1
    trits = trits.astype(np.uint64)
    pad = (-trits.size) % 40
    if pad:
        trits = np.concatenate([trits, np.ones(pad, dtype=np.uint64)])
    packed = (trits.reshape(-1, 40) * POW3_40.reshape(1, 40)).sum(axis=1, dtype=np.uint64)
    return packed, tuple(codes.shape), pad


def unpack_trits_40(packed: np.ndarray, shape: tuple[int, ...], pad: int) -> np.ndarray:
    """Unpack uint64 chunks produced by :func:`pack_trits_40`."""

    values = np.asarray(packed, dtype=np.uint64).copy()
    out = np.empty(values.size * 40, dtype=np.int8)
    for i in range(40):
        out[i::40] = (values % 3).astype(np.int8)
        values //= 3
    if pad:
        out = out[:-pad]
    return (out.reshape(shape) - 1).astype(np.int8)


def pack_bits(codes: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Pack binary {-1,+1} or {0,1} codes into uint8 bytes."""

    flat = np.asarray(codes).reshape(-1)
    if np.isin(flat, [-1, 1]).all():
        bits = (flat > 0).astype(np.uint8)
    elif np.isin(flat, [0, 1]).all():
        bits = flat.astype(np.uint8)
    else:
        raise ValueError("binary codes must be {-1,+1} or {0,1}")
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little"), tuple(codes.shape), pad


def unpack_bits(packed: np.ndarray, shape: tuple[int, ...], pad: int, *, signed: bool = True) -> np.ndarray:
    """Unpack uint8 binary codes."""

    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    if pad:
        bits = bits[:-pad]
    bits = bits.reshape(shape).astype(np.int8)
    return bits * 2 - 1 if signed else bits
