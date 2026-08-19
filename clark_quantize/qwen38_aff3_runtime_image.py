"""Self-authenticating AFF3_X16 runtime images for Qwen3.8-27B.

The canonical affine-base3 artifact remains the storage source of truth.  This
module derives the exact mmap-ready two-bit compute image consumed by the AMD
AVX2 kernel: row-tiled codes and scale indices, FP16 row anchors/codebooks,
FP16 Givens angles, and explicit uint16 pairings.  It never materializes a
dense floating-point weight or a decoded per-group scale matrix.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .ternary_cpu import (
    pack_block203_rows,
    transcode_block203_rows_to_group128_2bit,
    unpack_block203_rows,
)

MAGIC = b"AFF3X161"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_BYTES = 256
PAYLOAD_OFFSET = 4096
SECTION_ALIGNMENT = 64
GROUP_SIZE = 128
ROWS_PER_TILE = 16
ROW_GROUP_BYTES = 33
X16_GROUP_BYTES = ROW_GROUP_BYTES * ROWS_PER_TILE
SECTION_NAMES = (
    "weights_x16",
    "alpha_anchors",
    "mu_anchors",
    "alpha_codebook",
    "mu_codebook",
    "angles",
    "pairings",
)
HEADER = struct.Struct("<8sHHI6I16Q32s56s")

if HEADER.size != HEADER_BYTES:
    raise AssertionError("AFF3_X16 runtime header size changed")


class Aff3RuntimeImageError(ValueError):
    """The AFF3 runtime image is malformed or failed authentication."""


def align_up(value: int, alignment: int = SECTION_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment inputs must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def sha256_bytes(content: bytes | memoryview) -> str:
    return hashlib.sha256(content).hexdigest()


def pack_x16_weights(
    codes: np.ndarray,
    alpha_indices: np.ndarray,
    mu_indices: np.ndarray,
) -> bytes:
    """Losslessly transcode canonical trits into the frozen AFF3_X16 layout."""
    codes = np.asarray(codes, dtype=np.int8)
    alpha_indices = np.asarray(alpha_indices, dtype=np.uint8)
    mu_indices = np.asarray(mu_indices, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError("AFF3 codes must be a matrix")
    rows, columns = codes.shape
    groups = columns // GROUP_SIZE
    if (
        rows <= 0
        or rows % ROWS_PER_TILE
        or columns <= 0
        or columns % GROUP_SIZE
        or alpha_indices.shape != (rows, groups)
        or mu_indices.shape != (rows, groups)
        or bool((alpha_indices > 15).any())
        or bool((mu_indices > 15).any())
        or bool(((codes < -1) | (codes > 1)).any())
    ):
        raise ValueError("AFF3_X16 dimensions or values are invalid")

    canonical = pack_block203_rows(codes)
    rebuilt = unpack_block203_rows(canonical, rows, columns)
    if not np.array_equal(rebuilt, codes):
        raise AssertionError("canonical AFF3 trits changed before transcode")
    two_bit = transcode_block203_rows_to_group128_2bit(
        canonical, rows, columns
    ).reshape(rows, groups, 8, 4)
    combined = np.asarray(alpha_indices | (mu_indices << 4), dtype=np.uint8)
    tiled_codes = two_bit.reshape(
        rows // ROWS_PER_TILE, ROWS_PER_TILE, groups, 8, 4
    ).transpose(0, 2, 3, 1, 4)
    tiled_indices = combined.reshape(
        rows // ROWS_PER_TILE, ROWS_PER_TILE, groups
    ).transpose(0, 2, 1)
    image = np.empty((rows // ROWS_PER_TILE, groups, X16_GROUP_BYTES), dtype=np.uint8)
    image[..., :ROWS_PER_TILE] = tiled_indices
    image[..., ROWS_PER_TILE:] = tiled_codes.reshape(rows // ROWS_PER_TILE, groups, 512)
    return image.tobytes(order="C")


def unpack_x16_weights(
    content: bytes | memoryview, *, rows: int, columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact inverse used to authenticate the derived compute image."""
    if rows <= 0 or rows % ROWS_PER_TILE or columns <= 0 or columns % GROUP_SIZE:
        raise ValueError("AFF3_X16 dimensions are invalid")
    groups = columns // GROUP_SIZE
    expected = rows // ROWS_PER_TILE * groups * X16_GROUP_BYTES
    if len(content) != expected:
        raise ValueError("AFF3_X16 byte count changed")
    image = np.frombuffer(content, dtype=np.uint8).reshape(
        rows // ROWS_PER_TILE, groups, X16_GROUP_BYTES
    )
    combined = image[..., :ROWS_PER_TILE].transpose(0, 2, 1).reshape(rows, groups)
    tiled = image[..., ROWS_PER_TILE:].reshape(
        rows // ROWS_PER_TILE, groups, 8, ROWS_PER_TILE, 4
    )
    packed = tiled.transpose(0, 3, 1, 2, 4).reshape(rows, groups, 32)
    digits = np.empty((rows, groups, GROUP_SIZE), dtype=np.int8)
    for quarter in range(4):
        digits[..., quarter * 32 : (quarter + 1) * 32] = (
            (packed >> (2 * quarter)) & 3
        ).astype(np.int8)
    if bool((digits > 2).any()):
        raise Aff3RuntimeImageError("AFF3_X16 contains reserved code 3")
    return (
        (digits - 1).reshape(rows, columns),
        np.asarray(combined & 15, dtype=np.uint8),
        np.asarray(combined >> 4, dtype=np.uint8),
    )


def _little_f16(values: np.ndarray, shape: tuple[int, ...], label: str) -> bytes:
    array = np.asarray(values, dtype="<f2")
    if array.shape != shape or not bool(np.isfinite(array.astype(np.float32)).all()):
        raise ValueError(f"AFF3 {label} shape or finiteness changed")
    return array.tobytes(order="C")


def _little_u16(values: np.ndarray, shape: tuple[int, ...], label: str) -> bytes:
    array = np.asarray(values)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"AFF3 {label} shape or dtype changed")
    if bool((array < 0).any()) or bool((array > np.iinfo(np.uint16).max).any()):
        raise ValueError(f"AFF3 {label} exceeds uint16")
    return array.astype("<u2", copy=False).tobytes(order="C")


def build_sections(
    *,
    codes: np.ndarray,
    alpha_indices: np.ndarray,
    mu_indices: np.ndarray,
    alpha_anchors: np.ndarray,
    mu_anchors: np.ndarray,
    alpha_codebook: np.ndarray,
    mu_codebook: np.ndarray,
    angles: np.ndarray,
    pairings: np.ndarray,
) -> tuple[dict[str, bytes], dict[str, int]]:
    codes = np.asarray(codes, dtype=np.int8)
    if codes.ndim != 2:
        raise ValueError("AFF3 runtime source codes must be a matrix")
    rows, columns = codes.shape
    angles = np.asarray(angles)
    pairings = np.asarray(pairings)
    if angles.ndim != 2:
        raise ValueError("AFF3 rotation angles must be a matrix")
    stages, pairs = angles.shape
    if pairs * 2 != columns or pairings.shape != (stages, pairs, 2):
        raise ValueError("AFF3 rotation does not cover its input dimension")
    expected_indices = np.arange(columns)
    for stage in pairings:
        if not np.array_equal(np.sort(stage.reshape(-1)), expected_indices):
            raise ValueError("each AFF3 rotation stage must be a perfect matching")

    weights = pack_x16_weights(codes, alpha_indices, mu_indices)
    rebuilt_codes, rebuilt_alpha, rebuilt_mu = unpack_x16_weights(
        weights, rows=rows, columns=columns
    )
    if (
        not np.array_equal(rebuilt_codes, codes)
        or not np.array_equal(rebuilt_alpha, alpha_indices)
        or not np.array_equal(rebuilt_mu, mu_indices)
    ):
        raise AssertionError("AFF3_X16 derived image changed its algebra")
    sections = {
        "weights_x16": weights,
        "alpha_anchors": _little_f16(alpha_anchors, (rows,), "alpha anchors"),
        "mu_anchors": _little_f16(mu_anchors, (rows,), "mu anchors"),
        "alpha_codebook": _little_f16(alpha_codebook, (16,), "alpha codebook"),
        "mu_codebook": _little_f16(mu_codebook, (16,), "mu codebook"),
        "angles": _little_f16(angles, (stages, pairs), "angles"),
        "pairings": _little_u16(pairings, (stages, pairs, 2), "pairings"),
    }
    return sections, {
        "rows": rows,
        "columns": columns,
        "groups": columns // GROUP_SIZE,
        "stages": stages,
        "pairs": pairs,
    }


def _header(
    *,
    layer: int,
    dimensions: Mapping[str, int],
    payload_bytes: int,
    section_records: Mapping[str, tuple[int, int]],
    root_sha256: bytes,
) -> bytes:
    section_values = tuple(
        value for name in SECTION_NAMES for value in section_records[name]
    )
    return HEADER.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        HEADER_BYTES,
        layer,
        dimensions["rows"],
        dimensions["columns"],
        GROUP_SIZE,
        dimensions["stages"],
        dimensions["pairs"],
        PAYLOAD_OFFSET,
        payload_bytes,
        *section_values,
        root_sha256,
        bytes(56),
    )


def write_runtime_image(
    path: Path,
    *,
    layer: int,
    sections: Mapping[str, bytes],
    dimensions: Mapping[str, int],
) -> dict[str, Any]:
    if layer < 0 or layer >= 64 or set(sections) != set(SECTION_NAMES):
        raise ValueError("AFF3 runtime image identity changed")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    records: dict[str, tuple[int, int]] = {}
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            handle.write(bytes(PAYLOAD_OFFSET))
            for name in SECTION_NAMES:
                offset = align_up(handle.tell())
                handle.write(bytes(offset - handle.tell()))
                content = bytes(sections[name])
                handle.write(content)
                records[name] = (offset, len(content))
            file_bytes = handle.tell()
            payload_bytes = file_bytes - PAYLOAD_OFFSET
            zero_header = _header(
                layer=layer,
                dimensions=dimensions,
                payload_bytes=payload_bytes,
                section_records=records,
                root_sha256=bytes(32),
            )
            handle.seek(PAYLOAD_OFFSET)
            digest = hashlib.sha256(zero_header)
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            root_sha = digest.digest()
            handle.seek(0)
            handle.write(
                _header(
                    layer=layer,
                    dimensions=dimensions,
                    payload_bytes=payload_bytes,
                    section_records=records,
                    root_sha256=root_sha,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with Aff3RuntimeImage(path) as image:
        image.verify_all_sections()
        return image.receipt()


class Aff3RuntimeImage:
    """Fail-closed reader for one mmap-compatible AFF3_X16 layer image."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        self.file_bytes = self.path.stat().st_size
        content = self._handle.read(HEADER_BYTES)
        if len(content) != HEADER_BYTES:
            self.close()
            raise Aff3RuntimeImageError("truncated AFF3 runtime header")
        unpacked = HEADER.unpack(content)
        (
            magic,
            major,
            minor,
            header_bytes,
            layer,
            rows,
            columns,
            group_size,
            stages,
            pairs,
            payload_offset,
            payload_bytes,
            *tail,
        ) = unpacked
        section_values = tail[:14]
        root_sha = tail[14]
        reserved = tail[15]
        if (
            magic != MAGIC
            or (major, minor) != (VERSION_MAJOR, VERSION_MINOR)
            or header_bytes != HEADER_BYTES
            or layer < 0
            or layer >= 64
            or rows <= 0
            or rows % ROWS_PER_TILE
            or columns <= 0
            or columns % GROUP_SIZE
            or group_size != GROUP_SIZE
            or stages <= 0
            or pairs * 2 != columns
            or payload_offset != PAYLOAD_OFFSET
            or payload_bytes != self.file_bytes - PAYLOAD_OFFSET
            or reserved != bytes(56)
        ):
            self.close()
            raise Aff3RuntimeImageError("AFF3 runtime header contract changed")
        records = {
            name: (int(section_values[index * 2]), int(section_values[index * 2 + 1]))
            for index, name in enumerate(SECTION_NAMES)
        }
        previous = PAYLOAD_OFFSET
        for name in SECTION_NAMES:
            offset, size = records[name]
            if (
                offset % SECTION_ALIGNMENT
                or offset < previous
                or size <= 0
                or offset + size > self.file_bytes
            ):
                self.close()
                raise Aff3RuntimeImageError(f"invalid AFF3 section bounds: {name}")
            self._verify_zero_range(previous, offset)
            previous = offset + size
        if previous != self.file_bytes:
            self.close()
            raise Aff3RuntimeImageError("AFF3 runtime image has trailing bytes")
        expected_sizes = {
            "weights_x16": rows
            // ROWS_PER_TILE
            * (columns // GROUP_SIZE)
            * X16_GROUP_BYTES,
            "alpha_anchors": rows * 2,
            "mu_anchors": rows * 2,
            "alpha_codebook": 32,
            "mu_codebook": 32,
            "angles": stages * pairs * 2,
            "pairings": stages * pairs * 2 * 2,
        }
        if any(records[name][1] != expected_sizes[name] for name in SECTION_NAMES):
            self.close()
            raise Aff3RuntimeImageError("AFF3 runtime section size changed")
        zero_header = _header(
            layer=layer,
            dimensions={
                "rows": rows,
                "columns": columns,
                "stages": stages,
                "pairs": pairs,
            },
            payload_bytes=payload_bytes,
            section_records=records,
            root_sha256=bytes(32),
        )
        self._handle.seek(PAYLOAD_OFFSET)
        digest = hashlib.sha256(zero_header)
        for block in iter(lambda: self._handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.digest() != root_sha:
            self.close()
            raise Aff3RuntimeImageError("AFF3 runtime root hash mismatch")
        self.layer = layer
        self.rows = rows
        self.columns = columns
        self.groups = columns // GROUP_SIZE
        self.stages = stages
        self.pairs = pairs
        self.sections = records
        self.root_sha256 = root_sha.hex()

    def __enter__(self) -> Aff3RuntimeImage:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _verify_zero_range(self, start: int, stop: int) -> None:
        self._handle.seek(start)
        remaining = stop - start
        while remaining:
            content = self._handle.read(min(1024 * 1024, remaining))
            if not content or any(content):
                raise Aff3RuntimeImageError("AFF3 alignment padding is nonzero")
            remaining -= len(content)

    def read_section(self, name: str) -> bytes:
        try:
            offset, size = self.sections[name]
        except KeyError as error:
            raise KeyError(name) from error
        self._handle.seek(offset)
        content = self._handle.read(size)
        if len(content) != size:
            raise Aff3RuntimeImageError(f"truncated AFF3 section: {name}")
        return content

    def verify_all_sections(self) -> None:
        codes, alpha_indices, mu_indices = unpack_x16_weights(
            self.read_section("weights_x16"),
            rows=self.rows,
            columns=self.columns,
        )
        if (
            codes.shape != (self.rows, self.columns)
            or alpha_indices.shape != (self.rows, self.groups)
            or mu_indices.shape != (self.rows, self.groups)
        ):
            raise Aff3RuntimeImageError("AFF3 decoded dimensions changed")
        for name in (
            "alpha_anchors",
            "mu_anchors",
            "alpha_codebook",
            "mu_codebook",
            "angles",
        ):
            values = np.frombuffer(self.read_section(name), dtype="<f2")
            if not bool(np.isfinite(values.astype(np.float32)).all()):
                raise Aff3RuntimeImageError(f"AFF3 {name} contains non-finite values")
        pairings = np.frombuffer(self.read_section("pairings"), dtype="<u2").reshape(
            self.stages, self.pairs, 2
        )
        expected = np.arange(self.columns)
        for stage in pairings:
            if not np.array_equal(np.sort(stage.reshape(-1)), expected):
                raise Aff3RuntimeImageError("AFF3 stored pairing is not perfect")

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "qwen38_aff3_x16_runtime_image_v1",
            "path": str(self.path),
            "bytes": self.file_bytes,
            "root_sha256": self.root_sha256,
            "layer": self.layer,
            "shape": [self.rows, self.columns],
            "group_size": GROUP_SIZE,
            "rows_per_tile": ROWS_PER_TILE,
            "stages": self.stages,
            "pairs_per_stage": self.pairs,
            "sections": {
                name: {
                    "offset": self.sections[name][0],
                    "bytes": self.sections[name][1],
                    "sha256": sha256_bytes(self.read_section(name)),
                }
                for name in SECTION_NAMES
            },
            "persistent_dense_weight_bytes": 0,
            "decoded_group_scale_matrix_bytes": 0,
            "runtime_compute_image_bytes": sum(
                self.sections[name][1] for name in SECTION_NAMES
            ),
            "complete": True,
            "terminal": True,
        }


def build_from_compressed_artifact(
    *,
    artifact_root: Path,
    descriptor: Mapping[str, Any],
    angles: Any,
    pairings: Any,
    layer: int,
    path: Path,
) -> dict[str, Any]:
    """Load an authenticated canonical artifact and derive one runtime image."""
    from .qwen38_affine_artifact import (
        load_compressed_affine_tensor,
        verify_descriptor,
    )

    verify_descriptor(artifact_root, dict(descriptor))
    compressed = load_compressed_affine_tensor(artifact_root, dict(descriptor))
    sections, dimensions = build_sections(
        codes=compressed.payload.codes.numpy(),
        alpha_indices=compressed.alpha.indices,
        mu_indices=compressed.mu.indices,
        alpha_anchors=compressed.alpha.anchors,
        mu_anchors=compressed.mu.anchors,
        alpha_codebook=compressed.alpha.values,
        mu_codebook=compressed.mu.values,
        angles=np.asarray(angles),
        pairings=np.asarray(pairings),
    )
    return write_runtime_image(
        path,
        layer=layer,
        sections=sections,
        dimensions=dimensions,
    )
