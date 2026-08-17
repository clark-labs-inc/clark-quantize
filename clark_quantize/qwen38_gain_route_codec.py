"""Canonical bit packing for Qwen3.8-27B gain-routed Clark Hash.

The artifact body stores, for every input coordinate, a bucket index followed
by one sign bit and a quantized positive-gain index.  An FP16 gain codebook is
appended once.  Load-time expansion produces the wider arrays consumed by the
native fused CPU kernel; those compute-layout bytes are not confused with the
canonical on-disk representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PackedGainHashRouteBody:
    input_dim: int
    sketch_dim: int
    gain_bits: int
    bucket_bits: int
    route_payload: bytes
    gain_codebook: np.ndarray

    @property
    def bits_per_input(self) -> int:
        return self.bucket_bits + 1 + self.gain_bits

    @property
    def route_payload_bytes(self) -> int:
        return len(self.route_payload)

    @property
    def codebook_bytes(self) -> int:
        return int(self.gain_codebook.nbytes)

    @property
    def body_bytes(self) -> int:
        return self.route_payload_bytes + self.codebook_bytes

    @property
    def compute_layout_bytes(self) -> int:
        return self.input_dim * (2 + 1 + 1) + (1 << self.gain_bits) * 4


def gain_route_bucket_bits(sketch_dim: int) -> int:
    if sketch_dim <= 0 or sketch_dim > 65535:
        raise ValueError("gain route sketch dimension must lie in [1, 65535]")
    return max(1, math.ceil(math.log2(sketch_dim)))


def gain_route_payload_bytes(
    input_dim: int,
    sketch_dim: int,
    *,
    gain_bits: int = 3,
) -> int:
    if input_dim <= 0 or not 1 <= gain_bits <= 8:
        raise ValueError("gain route dimensions or gain bits are invalid")
    bits = gain_route_bucket_bits(sketch_dim) + 1 + gain_bits
    return (input_dim * bits + 7) // 8


def _canonical_codebook(values: np.ndarray, *, gain_bits: int) -> np.ndarray:
    source = np.asarray(values)
    levels = 1 << gain_bits
    if source.shape != (levels,) or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("gain route codebook dimensions or dtype are invalid")
    codebook = np.array(source, dtype=np.dtype("<f2"), copy=True, order="C")
    as_float = codebook.astype(np.float32)
    if (
        not np.all(np.isfinite(as_float))
        or np.any(as_float < 0)
        or np.any(as_float[1:] < as_float[:-1])
        or np.any((as_float == 0) & np.signbit(as_float))
    ):
        raise ValueError(
            "gain route codebook must be finite, nonnegative, sorted, and canonical"
        )
    codebook.setflags(write=False)
    return codebook


def _validate_route_arrays(
    buckets: np.ndarray,
    signs: np.ndarray,
    gain_codes: np.ndarray,
    *,
    sketch_dim: int,
    gain_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= gain_bits <= 8:
        raise ValueError("gain bits must lie in [1, 8]")
    gain_route_bucket_bits(sketch_dim)
    bucket_source = np.asarray(buckets)
    sign_source = np.asarray(signs)
    gain_source = np.asarray(gain_codes)
    if (
        bucket_source.ndim != 1
        or bucket_source.size == 0
        or sign_source.shape != bucket_source.shape
        or gain_source.shape != bucket_source.shape
        or not np.issubdtype(bucket_source.dtype, np.integer)
        or not np.issubdtype(sign_source.dtype, np.integer)
        or not np.issubdtype(gain_source.dtype, np.integer)
    ):
        raise ValueError("gain route arrays must be equal non-empty integer vectors")
    bucket_values = np.ascontiguousarray(bucket_source, dtype=np.uint16)
    sign_values = np.ascontiguousarray(sign_source, dtype=np.int8)
    gain_values = np.ascontiguousarray(gain_source, dtype=np.uint8)
    if (
        np.any(bucket_source < 0)
        or np.any(bucket_source >= sketch_dim)
        or not np.all((sign_source == -1) | (sign_source == 1))
        or np.any(gain_source < 0)
        or np.any(gain_source >= (1 << gain_bits))
    ):
        raise ValueError("gain route contains an out-of-range bucket, sign, or gain")
    return bucket_values, sign_values, gain_values


def pack_gain_hash_route_body(
    buckets: np.ndarray,
    signs: np.ndarray,
    gain_codes: np.ndarray,
    gain_codebook: np.ndarray,
    *,
    sketch_dim: int,
    gain_bits: int = 3,
) -> PackedGainHashRouteBody:
    """Pack a route body using a canonical LSB-first coordinate bitstream."""
    bucket_values, sign_values, gain_values = _validate_route_arrays(
        buckets,
        signs,
        gain_codes,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
    )
    codebook = _canonical_codebook(gain_codebook, gain_bits=gain_bits)
    input_dim = int(bucket_values.size)
    bucket_bits = gain_route_bucket_bits(sketch_dim)
    bits_per_input = bucket_bits + 1 + gain_bits
    output = bytearray(
        gain_route_payload_bytes(input_dim, sketch_dim, gain_bits=gain_bits)
    )
    accumulator = 0
    accumulator_bits = 0
    cursor = 0
    for bucket, sign, gain in zip(bucket_values, sign_values, gain_values):
        word = (
            int(bucket)
            | ((1 if int(sign) > 0 else 0) << bucket_bits)
            | (int(gain) << (bucket_bits + 1))
        )
        accumulator |= word << accumulator_bits
        accumulator_bits += bits_per_input
        while accumulator_bits >= 8:
            output[cursor] = accumulator & 0xFF
            cursor += 1
            accumulator >>= 8
            accumulator_bits -= 8
    if accumulator_bits:
        output[cursor] = accumulator & 0xFF
        cursor += 1
    if cursor != len(output) or accumulator >> accumulator_bits:
        raise AssertionError("gain route packing length changed")
    return PackedGainHashRouteBody(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
        bucket_bits=bucket_bits,
        route_payload=bytes(output),
        gain_codebook=codebook,
    )


def serialize_gain_hash_route_body(route: PackedGainHashRouteBody) -> bytes:
    _validate_packed_route(route)
    return route.route_payload + route.gain_codebook.tobytes(order="C")


def deserialize_gain_hash_route_body(
    body: bytes | bytearray | memoryview,
    *,
    input_dim: int,
    sketch_dim: int,
    gain_bits: int = 3,
) -> PackedGainHashRouteBody:
    raw = bytes(body)
    payload_bytes = gain_route_payload_bytes(
        input_dim, sketch_dim, gain_bits=gain_bits
    )
    codebook_bytes = (1 << gain_bits) * 2
    if len(raw) != payload_bytes + codebook_bytes:
        raise ValueError("gain route body length is not canonical")
    codebook = np.frombuffer(raw[payload_bytes:], dtype=np.dtype("<f2")).copy()
    route = PackedGainHashRouteBody(
        input_dim=input_dim,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
        bucket_bits=gain_route_bucket_bits(sketch_dim),
        route_payload=raw[:payload_bytes],
        gain_codebook=_canonical_codebook(codebook, gain_bits=gain_bits),
    )
    _validate_packed_route(route)
    return route


def _validate_packed_route(route: PackedGainHashRouteBody) -> None:
    if (
        route.input_dim <= 0
        or route.bucket_bits != gain_route_bucket_bits(route.sketch_dim)
        or not 1 <= route.gain_bits <= 8
        or len(route.route_payload)
        != gain_route_payload_bytes(
            route.input_dim, route.sketch_dim, gain_bits=route.gain_bits
        )
    ):
        raise ValueError("packed gain route metadata or payload length is invalid")
    canonical_codebook = _canonical_codebook(
        route.gain_codebook, gain_bits=route.gain_bits
    )
    if (
        route.gain_codebook.dtype != np.dtype("<f2")
        or not route.gain_codebook.flags.c_contiguous
        or route.gain_codebook.flags.writeable
        or route.gain_codebook.tobytes(order="C")
        != canonical_codebook.tobytes(order="C")
    ):
        raise ValueError("packed gain route codebook storage is not canonical")
    used_bits = route.input_dim * route.bits_per_input
    padding_bits = len(route.route_payload) * 8 - used_bits
    if padding_bits:
        padding_mask = ((1 << padding_bits) - 1) << (8 - padding_bits)
        if route.route_payload[-1] & padding_mask:
            raise ValueError("packed gain route has nonzero padding bits")


def unpack_gain_hash_route_body(
    route: PackedGainHashRouteBody,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode canonical disk codes without widening the gain codebook."""
    _validate_packed_route(route)
    buckets = np.empty(route.input_dim, dtype=np.uint16)
    signs = np.empty(route.input_dim, dtype=np.int8)
    gain_codes = np.empty(route.input_dim, dtype=np.uint8)
    bucket_mask = (1 << route.bucket_bits) - 1
    gain_mask = (1 << route.gain_bits) - 1
    accumulator = 0
    accumulator_bits = 0
    cursor = 0
    for index in range(route.input_dim):
        while accumulator_bits < route.bits_per_input:
            accumulator |= route.route_payload[cursor] << accumulator_bits
            cursor += 1
            accumulator_bits += 8
        word = accumulator & ((1 << route.bits_per_input) - 1)
        accumulator >>= route.bits_per_input
        accumulator_bits -= route.bits_per_input
        bucket = word & bucket_mask
        sign_bit = (word >> route.bucket_bits) & 1
        gain = (word >> (route.bucket_bits + 1)) & gain_mask
        if bucket >= route.sketch_dim:
            raise ValueError("packed gain route contains a noncanonical bucket code")
        buckets[index] = bucket
        signs[index] = 1 if sign_bit else -1
        gain_codes[index] = gain
    if cursor != len(route.route_payload) or accumulator != 0:
        raise ValueError("packed gain route has a noncanonical terminal state")
    return buckets, signs, gain_codes


def expand_gain_hash_route_compute(
    route: PackedGainHashRouteBody,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand the disk body into the exact arrays consumed by the CPU kernel."""
    buckets, signs, gain_codes = unpack_gain_hash_route_body(route)
    codebook = np.ascontiguousarray(route.gain_codebook, dtype=np.float16).astype(
        np.float32
    )
    observed = buckets.nbytes + signs.nbytes + gain_codes.nbytes + codebook.nbytes
    if observed != route.compute_layout_bytes:
        raise AssertionError("gain route compute-layout byte count changed")
    return buckets, signs, gain_codes, codebook
