"""Canonical complete-layer body for Qwen3.8-27B gain-hash ternary weights.

The on-disk body keeps the exact enumerative fixed-support ternary code.  A
loader may transcode that code to the wider mask64 compute image consumed by
the fused CPU kernel, but those derived bytes are not charged to model storage.

Every byte counted by :func:`gain_hash_storage_projection` is present exactly
once in this body: a 64-byte header, ternary codes, scale indices and side
channels, the packed gain route, and signed-INT4 correction coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from .qwen38_fixed_support_codec import (
    FixedSupportPayload,
    fixed_support_width_bits,
    pack_fixed_support_matrix,
    unpack_fixed_support_matrix,
)
from .qwen38_fixed_support_mask64_cpu import pack_scale_indices4
from .qwen38_fixed_support_mask64_cpu import (
    Mask64CPULibrary,
    Mask64Payload,
    native_pack_mask64_matrix,
)
from .qwen38_gain_route_codec import (
    PackedGainHashRouteBody,
    deserialize_gain_hash_route_body,
    gain_route_bucket_bits,
    gain_route_payload_bytes,
    serialize_gain_hash_route_body,
)
from .qwen38_hash_residual_cpu import pack_signed_int4_rows


MAGIC = b"CQ3GHLY1"
VERSION = 1
HEADER_BYTES = 64
HEADER = struct.Struct("<8sHHIIHHHBB7I8s")

if HEADER.size != HEADER_BYTES:
    raise AssertionError("gain-hash layer header must remain exactly 64 bytes")


@dataclass(frozen=True)
class GainHashLayerComponents:
    ternary: FixedSupportPayload
    scale_indices: np.ndarray
    row_anchors: np.ndarray
    ratio_codebook: np.ndarray
    route: PackedGainHashRouteBody
    correction_codes: np.ndarray
    correction_row_scales: np.ndarray

    @property
    def rows(self) -> int:
        return self.ternary.rows

    @property
    def columns(self) -> int:
        return self.ternary.columns

    @property
    def sketch_dim(self) -> int:
        return self.route.sketch_dim


@dataclass(frozen=True)
class Mask64GainHashLayerCompute:
    """Load-time compute image; its wider bytes are not model storage."""

    ternary: Mask64Payload
    scale_indices: np.ndarray
    row_anchors: np.ndarray
    ratio_codebook: np.ndarray
    route: PackedGainHashRouteBody
    correction_codes: np.ndarray
    correction_row_scales: np.ndarray


def gain_hash_layer_body_layout(
    *,
    rows: int,
    columns: int,
    sketch_dim: int,
    gain_bits: int = 3,
    group_size: int = 128,
    nonzero_per_group: int = 64,
) -> dict[str, int]:
    if (
        rows <= 0
        or columns <= 0
        or columns % group_size
        or group_size > 65535
        or not 0 < nonzero_per_group <= group_size
        or sketch_dim <= 0
        or sketch_dim > 65535
        or not 1 <= gain_bits <= 8
    ):
        raise ValueError("gain-hash layer dimensions are invalid")
    groups_per_row = columns // group_size
    group_width_bits = fixed_support_width_bits(group_size, nonzero_per_group)
    ternary_bits = rows * groups_per_row * group_width_bits
    lengths = {
        "header_bytes": HEADER_BYTES,
        "ternary_bits": ternary_bits,
        "ternary_bytes": (ternary_bits + 7) // 8,
        "scale_index_bytes": (rows * groups_per_row * 4 + 7) // 8,
        "row_anchor_bytes": rows * 2,
        "ratio_codebook_bytes": 16 * 2,
        "route_body_bytes": gain_route_payload_bytes(
            columns, sketch_dim, gain_bits=gain_bits
        )
        + (1 << gain_bits) * 2,
        "correction_code_bytes": rows * ((sketch_dim + 1) // 2),
        "correction_scale_bytes": rows * 2,
        "groups_per_row": groups_per_row,
        "group_width_bits": group_width_bits,
        "bucket_bits": gain_route_bucket_bits(sketch_dim),
    }
    lengths["body_bytes"] = sum(
        lengths[name]
        for name in (
            "header_bytes",
            "ternary_bytes",
            "scale_index_bytes",
            "row_anchor_bytes",
            "ratio_codebook_bytes",
            "route_body_bytes",
            "correction_code_bytes",
            "correction_scale_bytes",
        )
    )
    if any(value > 0xFFFFFFFF for name, value in lengths.items() if name.endswith("_bytes")):
        raise ValueError("gain-hash layer component exceeds the v1 length field")
    return lengths


def _canonical_fp16(
    values: np.ndarray,
    *,
    shape: tuple[int, ...],
    label: str,
    sorted_values: bool = False,
) -> np.ndarray:
    source = np.asarray(values)
    if source.shape != shape or not np.issubdtype(source.dtype, np.floating):
        raise ValueError(f"{label} dimensions or dtype are invalid")
    result = np.array(source, dtype=np.dtype("<f2"), copy=True, order="C")
    as_float = result.astype(np.float32)
    if (
        not np.all(np.isfinite(as_float))
        or np.any(as_float < 0)
        or np.any((as_float == 0) & np.signbit(as_float))
        or (sorted_values and np.any(as_float[1:] < as_float[:-1]))
    ):
        raise ValueError(f"{label} must be finite, nonnegative, and canonical")
    result.setflags(write=False)
    return result


def _unpack_scale_indices4(
    payload: bytes, *, rows: int, groups_per_row: int
) -> np.ndarray:
    count = rows * groups_per_row
    if len(payload) != (count + 1) // 2:
        raise ValueError("gain-hash scale-index payload length changed")
    packed = np.frombuffer(payload, dtype=np.uint8)
    if count & 1 and packed[-1] & 0xF0:
        raise ValueError("gain-hash scale-index padding nibble is nonzero")
    flattened = np.empty(count, dtype=np.uint8)
    flattened[0::2] = packed & 0x0F
    flattened[1::2] = (packed[: count // 2] >> 4) & 0x0F
    return flattened.reshape(rows, groups_per_row)


def _unpack_signed_int4_rows(
    payload: bytes, *, rows: int, columns: int
) -> np.ndarray:
    row_bytes = (columns + 1) // 2
    if len(payload) != rows * row_bytes:
        raise ValueError("gain-hash correction payload length changed")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(rows, row_bytes)
    if columns & 1 and np.any(packed[:, -1] & 0xF0):
        raise ValueError("gain-hash correction padding nibble is nonzero")
    nibbles = np.empty((rows, columns), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = (packed[:, : columns // 2] >> 4) & 0x0F
    if np.any(nibbles == 8):
        raise ValueError("gain-hash correction contains forbidden INT4 -8")
    signed = nibbles.astype(np.int16)
    signed[signed >= 8] -= 16
    return np.ascontiguousarray(signed, dtype=np.int8)


def _component_lengths(layout: dict[str, int]) -> tuple[int, ...]:
    return tuple(
        layout[name]
        for name in (
            "ternary_bytes",
            "scale_index_bytes",
            "row_anchor_bytes",
            "ratio_codebook_bytes",
            "route_body_bytes",
            "correction_code_bytes",
            "correction_scale_bytes",
        )
    )


def pack_gain_hash_layer(components: GainHashLayerComponents) -> bytes:
    rows = components.rows
    columns = components.columns
    route = components.route
    layout = gain_hash_layer_body_layout(
        rows=rows,
        columns=columns,
        sketch_dim=route.sketch_dim,
        gain_bits=route.gain_bits,
        group_size=components.ternary.group_size,
        nonzero_per_group=components.ternary.nonzero_per_group,
    )
    # Re-decode once so a caller cannot smuggle a malformed enumerative body
    # through a manually constructed FixedSupportPayload.
    ternary_codes = unpack_fixed_support_matrix(components.ternary)
    canonical_ternary = pack_fixed_support_matrix(
        ternary_codes,
        group_size=components.ternary.group_size,
        nonzero_per_group=components.ternary.nonzero_per_group,
    )
    if canonical_ternary.payload != components.ternary.payload:
        raise ValueError("gain-hash ternary payload is not canonical")
    indices = np.asarray(components.scale_indices)
    if indices.shape != (rows, layout["groups_per_row"]):
        raise ValueError("gain-hash scale-index dimensions disagree")
    packed_indices = pack_scale_indices4(indices).tobytes(order="C")
    anchors = _canonical_fp16(
        components.row_anchors, shape=(rows,), label="gain-hash row anchors"
    )
    ratios = _canonical_fp16(
        components.ratio_codebook,
        shape=(16,),
        label="gain-hash ratio codebook",
        sorted_values=True,
    )
    if (
        route.input_dim != columns
        or route.bucket_bits != layout["bucket_bits"]
        or route.body_bytes != layout["route_body_bytes"]
    ):
        raise ValueError("gain-hash route and layer dimensions disagree")
    route_body = serialize_gain_hash_route_body(route)
    correction_codes = np.asarray(components.correction_codes)
    if correction_codes.shape != (rows, route.sketch_dim):
        raise ValueError("gain-hash correction-code dimensions disagree")
    packed_correction = pack_signed_int4_rows(correction_codes).tobytes(order="C")
    correction_scales = _canonical_fp16(
        components.correction_row_scales,
        shape=(rows,),
        label="gain-hash correction row scales",
    )
    payloads = (
        canonical_ternary.payload,
        packed_indices,
        anchors.tobytes(order="C"),
        ratios.tobytes(order="C"),
        route_body,
        packed_correction,
        correction_scales.tobytes(order="C"),
    )
    if tuple(map(len, payloads)) != _component_lengths(layout):
        raise AssertionError("gain-hash layer component byte count changed")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        rows,
        columns,
        components.ternary.group_size,
        components.ternary.nonzero_per_group,
        route.sketch_dim,
        route.gain_bits,
        route.bucket_bits,
        *_component_lengths(layout),
        bytes(8),
    )
    body = header + b"".join(payloads)
    if len(body) != layout["body_bytes"]:
        raise AssertionError("gain-hash layer body byte count changed")
    return body


def deserialize_gain_hash_layer(body: bytes | bytearray | memoryview) -> GainHashLayerComponents:
    raw = bytes(body)
    if len(raw) < HEADER_BYTES:
        raise ValueError("gain-hash layer body is truncated")
    (
        magic,
        version,
        header_bytes,
        rows,
        columns,
        group_size,
        nonzero_per_group,
        sketch_dim,
        gain_bits,
        bucket_bits,
        *tail,
    ) = HEADER.unpack(raw[:HEADER_BYTES])
    lengths = tuple(tail[:-1])
    reserved = tail[-1]
    if (
        magic != MAGIC
        or version != VERSION
        or header_bytes != HEADER_BYTES
        or reserved != bytes(8)
    ):
        raise ValueError("gain-hash layer header is not canonical")
    layout = gain_hash_layer_body_layout(
        rows=rows,
        columns=columns,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
        group_size=group_size,
        nonzero_per_group=nonzero_per_group,
    )
    if (
        bucket_bits != layout["bucket_bits"]
        or lengths != _component_lengths(layout)
        or len(raw) != layout["body_bytes"]
    ):
        raise ValueError("gain-hash layer header ledger is inconsistent")
    cursor = HEADER_BYTES
    payloads = []
    for length in lengths:
        payloads.append(raw[cursor : cursor + length])
        cursor += length
    if cursor != len(raw):
        raise AssertionError("gain-hash layer parse length changed")
    ternary = FixedSupportPayload(
        payload=payloads[0],
        bit_length=layout["ternary_bits"],
        rows=rows,
        columns=columns,
        group_size=group_size,
        nonzero_per_group=nonzero_per_group,
        group_width_bits=layout["group_width_bits"],
    )
    # Fail closed on non-canonical support ranks or terminal padding.
    unpack_fixed_support_matrix(ternary)
    scale_indices = _unpack_scale_indices4(
        payloads[1], rows=rows, groups_per_row=layout["groups_per_row"]
    )
    row_anchors = _canonical_fp16(
        np.frombuffer(payloads[2], dtype=np.dtype("<f2")),
        shape=(rows,),
        label="gain-hash row anchors",
    )
    ratio_codebook = _canonical_fp16(
        np.frombuffer(payloads[3], dtype=np.dtype("<f2")),
        shape=(16,),
        label="gain-hash ratio codebook",
        sorted_values=True,
    )
    route = deserialize_gain_hash_route_body(
        payloads[4],
        input_dim=columns,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
    )
    correction_codes = _unpack_signed_int4_rows(
        payloads[5], rows=rows, columns=sketch_dim
    )
    correction_scales = _canonical_fp16(
        np.frombuffer(payloads[6], dtype=np.dtype("<f2")),
        shape=(rows,),
        label="gain-hash correction row scales",
    )
    return GainHashLayerComponents(
        ternary=ternary,
        scale_indices=scale_indices,
        row_anchors=row_anchors,
        ratio_codebook=ratio_codebook,
        route=route,
        correction_codes=correction_codes,
        correction_row_scales=correction_scales,
    )


def gain_hash_layer_storage_bits_per_weight(
    *, rows: int, columns: int, sketch_dim: int, gain_bits: int = 3
) -> float:
    layout = gain_hash_layer_body_layout(
        rows=rows,
        columns=columns,
        sketch_dim=sketch_dim,
        gain_bits=gain_bits,
    )
    return layout["body_bytes"] * 8 / (rows * columns)


def materialize_mask64_gain_hash_layer(
    components: GainHashLayerComponents,
    *,
    library: Mask64CPULibrary,
) -> Mask64GainHashLayerCompute:
    """Transcode the canonical enumerative body into the fused-kernel layout."""
    if (
        components.ternary.group_size != 128
        or components.ternary.nonzero_per_group != 64
    ):
        raise ValueError("mask64 materialization requires exact 64-of-128 support")
    ternary_codes = unpack_fixed_support_matrix(components.ternary)
    mask64 = native_pack_mask64_matrix(ternary_codes, library=library)
    return Mask64GainHashLayerCompute(
        ternary=mask64,
        scale_indices=np.ascontiguousarray(components.scale_indices, dtype=np.uint8),
        row_anchors=np.ascontiguousarray(components.row_anchors, dtype=np.float16),
        ratio_codebook=np.ascontiguousarray(
            components.ratio_codebook, dtype=np.float16
        ),
        route=components.route,
        correction_codes=np.ascontiguousarray(
            components.correction_codes, dtype=np.int8
        ),
        correction_row_scales=np.ascontiguousarray(
            components.correction_row_scales, dtype=np.float16
        ),
    )
