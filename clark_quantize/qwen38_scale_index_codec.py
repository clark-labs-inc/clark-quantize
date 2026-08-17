"""Lossless canonical Huffman coding for Qwen3.8-27B scale indices.

The canonical CQ3 artifact may entropy-code each tensor's row-normalized
4-bit scale indices.  The aligned CPU cache still expands them to the existing
nibble layout, so this codec changes storage and load-time transcode only; it
cannot change a weight, scale, or model output.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import struct
from typing import Iterable

import numpy as np


ALPHABET_SIZE = 16
MAGIC = b"QSI1"
HEADER = struct.Struct("<4sQQH16B2x")


@dataclass(frozen=True)
class ScaleIndexHuffmanLedger:
    count: int
    frequencies: tuple[int, ...]
    code_lengths: tuple[int, ...]
    symbol_mask: int
    payload_bits: int
    payload_bytes: int
    section_bytes: int

    @property
    def fixed_width_bytes(self) -> int:
        return (self.count + 1) // 2

    @property
    def savings_bytes_before_alignment(self) -> int:
        return self.fixed_width_bytes - self.section_bytes


def _frequency_array(values: np.ndarray | Iterable[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("scale indices must be a non-empty integer vector")
    if np.any(array < 0) or np.any(array >= ALPHABET_SIZE):
        raise ValueError("scale index is outside the 4-bit alphabet")
    return np.bincount(array.astype(np.int64, copy=False), minlength=ALPHABET_SIZE)


def _code_lengths_from_frequencies(frequencies: np.ndarray) -> tuple[int, ...]:
    counts = np.asarray(frequencies, dtype=np.int64)
    if counts.shape != (ALPHABET_SIZE,) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("invalid scale-index frequencies")
    active = [int(symbol) for symbol in np.flatnonzero(counts)]
    if len(active) == 1:
        lengths = [0] * ALPHABET_SIZE
        return tuple(lengths)

    order = 0
    heap: list[tuple[int, int, int, object]] = []
    for symbol in active:
        heapq.heappush(
            heap,
            (int(counts[symbol]), symbol, order, symbol),
        )
        order += 1
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = (left[3], right[3])
        heapq.heappush(
            heap,
            (left[0] + right[0], min(left[1], right[1]), order, node),
        )
        order += 1

    lengths = [0] * ALPHABET_SIZE

    def visit(node: object, depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = depth
            return
        left, right = node
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(heap[0][3], 0)
    if any(length > ALPHABET_SIZE - 1 for length in lengths):
        raise AssertionError("16-symbol Huffman depth exceeded 15 bits")
    return tuple(lengths)


def _canonical_codes(
    code_lengths: tuple[int, ...], symbol_mask: int
) -> dict[int, tuple[int, int]]:
    if len(code_lengths) != ALPHABET_SIZE or not 0 < symbol_mask < (1 << ALPHABET_SIZE):
        raise ValueError("invalid scale-index codebook dimensions")
    active = [symbol for symbol in range(ALPHABET_SIZE) if symbol_mask & (1 << symbol)]
    if len(active) == 1:
        if any(code_lengths):
            raise ValueError("single-symbol scale-index codebook must use zero bits")
        return {active[0]: (0, 0)}
    if any(
        (symbol in active) != (0 < int(code_lengths[symbol]) <= ALPHABET_SIZE - 1)
        for symbol in range(ALPHABET_SIZE)
    ):
        raise ValueError("scale-index mask and code lengths disagree")

    ordered = sorted((int(code_lengths[symbol]), symbol) for symbol in active)
    code = 0
    previous_length = ordered[0][0]
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in ordered:
        code <<= length - previous_length
        if code >= (1 << length):
            raise ValueError("oversubscribed canonical scale-index codebook")
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    if code != (1 << previous_length):
        raise ValueError("incomplete canonical scale-index codebook")
    return result


def scale_index_huffman_ledger_from_frequencies(
    frequencies: Iterable[int],
) -> ScaleIndexHuffmanLedger:
    counts = np.asarray(tuple(frequencies), dtype=np.int64)
    if counts.shape != (ALPHABET_SIZE,) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("scale-index frequency ledger must contain 16 nonnegative counts")
    lengths = _code_lengths_from_frequencies(counts)
    active = np.flatnonzero(counts)
    symbol_mask = sum(1 << int(symbol) for symbol in active)
    _canonical_codes(lengths, symbol_mask)
    payload_bits = int(sum(int(counts[symbol]) * lengths[symbol] for symbol in active))
    payload_bytes = (payload_bits + 7) // 8
    return ScaleIndexHuffmanLedger(
        count=int(counts.sum()),
        frequencies=tuple(int(value) for value in counts),
        code_lengths=lengths,
        symbol_mask=symbol_mask,
        payload_bits=payload_bits,
        payload_bytes=payload_bytes,
        section_bytes=HEADER.size + payload_bytes,
    )


def scale_index_huffman_ledger(
    values: np.ndarray | Iterable[int],
) -> ScaleIndexHuffmanLedger:
    return scale_index_huffman_ledger_from_frequencies(_frequency_array(values))


def encode_scale_indices_huffman(
    values: np.ndarray | Iterable[int],
) -> bytes:
    array = np.asarray(values)
    frequencies = _frequency_array(array)
    ledger = scale_index_huffman_ledger_from_frequencies(frequencies)
    codes = _canonical_codes(ledger.code_lengths, ledger.symbol_mask)
    payload = bytearray(ledger.payload_bytes)
    bit_position = 0
    for raw_symbol in array:
        symbol = int(raw_symbol)
        code, length = codes[symbol]
        for shift in range(length - 1, -1, -1):
            if code & (1 << shift):
                payload[bit_position // 8] |= 1 << (7 - bit_position % 8)
            bit_position += 1
    if bit_position != ledger.payload_bits:
        raise AssertionError("scale-index Huffman encoder bit ledger diverged")
    header = HEADER.pack(
        MAGIC,
        ledger.count,
        ledger.payload_bits,
        ledger.symbol_mask,
        *ledger.code_lengths,
    )
    return header + bytes(payload)


def decode_scale_indices_huffman(content: bytes | bytearray | memoryview) -> np.ndarray:
    data = bytes(content)
    if len(data) < HEADER.size:
        raise ValueError("scale-index Huffman section is truncated")
    unpacked = HEADER.unpack_from(data)
    magic, count, payload_bits, symbol_mask = unpacked[:4]
    lengths = tuple(int(value) for value in unpacked[4:])
    if magic != MAGIC or count == 0:
        raise ValueError("invalid scale-index Huffman header")
    payload_bytes = (payload_bits + 7) // 8
    if len(data) != HEADER.size + payload_bytes:
        raise ValueError("scale-index Huffman payload length changed")
    payload = data[HEADER.size:]
    if payload_bits % 8 and payload and payload[-1] & ((1 << (8 - payload_bits % 8)) - 1):
        raise ValueError("nonzero scale-index Huffman padding bits")
    codes = _canonical_codes(lengths, symbol_mask)
    if len(codes) == 1:
        if payload_bits != 0 or payload:
            raise ValueError("single-symbol scale-index payload must be empty")
        return np.full(int(count), next(iter(codes)), dtype=np.uint8)

    decode = {(length, code): symbol for symbol, (code, length) in codes.items()}
    output = np.empty(int(count), dtype=np.uint8)
    bit_position = 0
    for output_position in range(int(count)):
        code = 0
        matched = False
        for length in range(1, ALPHABET_SIZE):
            if bit_position >= payload_bits:
                raise ValueError("scale-index Huffman payload ended mid-symbol")
            byte = payload[bit_position // 8]
            code = (code << 1) | ((byte >> (7 - bit_position % 8)) & 1)
            bit_position += 1
            symbol = decode.get((length, code))
            if symbol is not None:
                output[output_position] = symbol
                matched = True
                break
        if not matched:
            raise ValueError("scale-index Huffman payload contains an invalid code")
    if bit_position != payload_bits:
        raise ValueError("scale-index Huffman payload has trailing codes")
    return output
