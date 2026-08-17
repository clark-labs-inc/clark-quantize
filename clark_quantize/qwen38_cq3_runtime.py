"""Read-only native-compute views over a Qwen3.8-27B CQ3 container.

The canonical container stays the storage source of truth.  This module maps
its authenticated sections directly into the block203 plus row-scale4 CPU
kernel without expanding the ternary weights into a larger compute image.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
from typing import Mapping

import numpy as np

from .base3_block203 import matrix_payload_bits, matrix_payload_bytes
from .qwen38_cq3_container import CQ3ContainerError, Qwen38CQ3Reader
from .ternary_cpu import (
    causal_conv4_silu_f32,
    gated_delta_parameters_f32,
    gqa_decode_f32,
    gemv_q8_group128_2bit_row_scale4,
    gemv_q8_group128_block203_row_scale4,
    quantize_q8,
    recurrent_gated_delta_f32,
    rmsnorm_offset_f32,
    rmsnorm_rope_f32,
    rmsnorm_silu_gate_f32,
    sigmoid_gate_f32,
    swiglu_f32,
    transcode_block203_rows_to_group128_2bit,
    unpack_block203_rows,
    unpack_group128_2bit,
)


ROW_NORMALIZED_ENCODING = "base3_group128_row_normalized_codebook4"
ROW_NORMALIZED_SCALE_KIND = (
    "scale_indices_u4_anchors_f16_and_codebook_f16"
)
GROUP_SIZE = 128
FULL_GROUP_BITS = 203


def _decode_row_scales(
    packed_indices: np.ndarray,
    row_anchors: np.ndarray,
    ratio_codebook: np.ndarray,
    row: int,
    groups: int,
) -> np.ndarray:
    positions = np.arange(row * groups, (row + 1) * groups, dtype=np.int64)
    packed = packed_indices[positions // 2]
    shifts = ((positions & 1) * 4).astype(np.uint8)
    indices = (packed >> shifts) & np.uint8(0x0F)
    return (
        np.float32(row_anchors[row])
        * ratio_codebook[indices].astype(np.float32)
    )


def _merge_topk(
    retained_values: np.ndarray,
    retained_indices: np.ndarray,
    values: np.ndarray,
    offset: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(offset, offset + values.size, dtype=np.int64)
    candidate_values = np.concatenate((retained_values, values))
    candidate_indices = np.concatenate((retained_indices, indices))
    order = np.lexsort((candidate_indices, -candidate_values))[:count]
    return candidate_values[order], candidate_indices[order]


class CQ3RuntimeError(CQ3ContainerError):
    """The storage container cannot be used by the frozen native runtime."""


@dataclass(frozen=True)
class Qwen38CQ3TernaryTensorView:
    """Zero-copy arrays for one row-normalized ternary matrix."""

    name: str
    shape: tuple[int, ...]
    rows: int
    columns: int
    packed_weights: np.ndarray
    packed_scale_indices: np.ndarray
    row_anchors: np.ndarray
    ratio_codebook: np.ndarray

    def derive_aligned2(self) -> "Qwen38CQ3AlignedTensor":
        """Derive the wider SIMD compute image without an int8 weight matrix."""
        weights = transcode_block203_rows_to_group128_2bit(
            self.packed_weights, self.rows, self.columns
        ).reshape(self.rows, self.columns // 4)
        weights.flags.writeable = False
        return Qwen38CQ3AlignedTensor(
            name=self.name,
            shape=self.shape,
            rows=self.rows,
            columns=self.columns,
            packed_weights=weights,
            packed_scale_indices=self.packed_scale_indices,
            row_anchors=self.row_anchors,
            ratio_codebook=self.ratio_codebook,
        )

    def gemv_q8(
        self,
        activation_q8: np.ndarray,
        activation_scales: np.ndarray,
        activation_sums: np.ndarray,
        *,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Run the native direct-storage GEMV without expanding the weights."""
        return gemv_q8_group128_block203_row_scale4(
            self.packed_weights,
            self.packed_scale_indices,
            self.row_anchors,
            self.ratio_codebook,
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            group_size=GROUP_SIZE,
            # The native ABI still validates finite scales, nibble padding,
            # block padding, and every decoded base-3 integer.  This only
            # avoids duplicating those scans in NumPy before every token.
            validated=True,
        )

    def gemv_f32(
        self,
        activation: np.ndarray,
        *,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Quantize one activation vector to A8 and run direct-storage GEMV."""
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.columns:
            raise ValueError(
                f"{self.name} requires one activation vector of length "
                f"{self.columns}"
            )
        activation_q8, activation_scales, activation_sums = quantize_q8(
            values, GROUP_SIZE
        )
        return self.gemv_q8(
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
        )

    def decode_row_f32(self, row: int) -> np.ndarray:
        """Decode one authenticated row without materializing the matrix."""
        if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < self.rows:
            raise ValueError(f"{self.name} row index is out of range")
        row_bits = matrix_payload_bits(1, self.columns)
        if row_bits % 8:
            raise CQ3RuntimeError(
                f"{self.name} rows are not independently byte-addressable"
            )
        row_bytes = matrix_payload_bytes(1, self.columns)
        start = row * row_bytes
        codes = unpack_block203_rows(
            self.packed_weights[start : start + row_bytes], 1, self.columns
        )[0]
        groups = self.columns // GROUP_SIZE
        scales = _decode_row_scales(
            self.packed_scale_indices,
            self.row_anchors,
            self.ratio_codebook,
            row,
            groups,
        )
        return (
            codes.reshape(groups, GROUP_SIZE).astype(np.float32)
            * scales[:, None]
        ).reshape(self.columns)

    def topk_q8(
        self,
        activation_q8: np.ndarray,
        activation_scales: np.ndarray,
        activation_sums: np.ndarray,
        count: int,
        *,
        row_chunk: int = 4096,
        implementation: str = "scalar_chunk16",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stream rows through GEMV and retain only deterministic top-k."""
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > self.rows
            or not isinstance(row_chunk, int)
            or isinstance(row_chunk, bool)
            or row_chunk <= 0
        ):
            raise ValueError(f"{self.name} streaming top-k arguments are invalid")
        row_bits = matrix_payload_bits(1, self.columns)
        groups = self.columns // GROUP_SIZE
        if row_bits % 8 or groups % 2:
            raise CQ3RuntimeError(
                f"{self.name} rows are not independently byte-addressable"
            )
        row_bytes = row_bits // 8
        scale_bytes = groups // 2
        retained_values = np.empty(0, dtype=np.float32)
        retained_indices = np.empty(0, dtype=np.int64)
        for start in range(0, self.rows, row_chunk):
            stop = min(start + row_chunk, self.rows)
            chunk = Qwen38CQ3TernaryTensorView(
                name=f"{self.name}[{start}:{stop}]",
                shape=(stop - start, self.columns),
                rows=stop - start,
                columns=self.columns,
                packed_weights=self.packed_weights[
                    start * row_bytes : stop * row_bytes
                ],
                packed_scale_indices=self.packed_scale_indices[
                    start * scale_bytes : stop * scale_bytes
                ],
                row_anchors=self.row_anchors[start:stop],
                ratio_codebook=self.ratio_codebook,
            )
            values = chunk.gemv_q8(
                activation_q8,
                activation_scales,
                activation_sums,
                implementation=implementation,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
            )
            retained_values, retained_indices = _merge_topk(
                retained_values, retained_indices, values, start, count
            )
        return retained_values, retained_indices


@dataclass(frozen=True)
class Qwen38CQ3AlignedTensor:
    """Derived aligned-2 weight bytes plus canonical row-scale4 views."""

    name: str
    shape: tuple[int, ...]
    rows: int
    columns: int
    packed_weights: np.ndarray
    packed_scale_indices: np.ndarray
    row_anchors: np.ndarray
    ratio_codebook: np.ndarray

    def gemv_q8(
        self,
        activation_q8: np.ndarray,
        activation_scales: np.ndarray,
        activation_sums: np.ndarray,
        *,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> np.ndarray:
        return gemv_q8_group128_2bit_row_scale4(
            self.packed_weights,
            self.packed_scale_indices,
            self.row_anchors,
            self.ratio_codebook,
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            group_size=GROUP_SIZE,
            validated=True,
        )

    def gemv_f32(
        self,
        activation: np.ndarray,
        *,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> np.ndarray:
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.columns:
            raise ValueError(
                f"{self.name} requires one activation vector of length "
                f"{self.columns}"
            )
        activation_q8, activation_scales, activation_sums = quantize_q8(
            values, GROUP_SIZE
        )
        return self.gemv_q8(
            activation_q8,
            activation_scales,
            activation_sums,
            implementation=implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
        )

    def decode_row_f32(self, row: int) -> np.ndarray:
        """Decode one row from the aligned working representation."""
        if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < self.rows:
            raise ValueError(f"{self.name} row index is out of range")
        codes = unpack_group128_2bit(
            self.packed_weights[row], self.columns
        )
        groups = self.columns // GROUP_SIZE
        scales = _decode_row_scales(
            self.packed_scale_indices,
            self.row_anchors,
            self.ratio_codebook,
            row,
            groups,
        )
        return (
            codes.reshape(groups, GROUP_SIZE).astype(np.float32)
            * scales[:, None]
        ).reshape(self.columns)

    def topk_q8(
        self,
        activation_q8: np.ndarray,
        activation_scales: np.ndarray,
        activation_sums: np.ndarray,
        count: int,
        *,
        row_chunk: int = 4096,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > self.rows
            or not isinstance(row_chunk, int)
            or isinstance(row_chunk, bool)
            or row_chunk <= 0
        ):
            raise ValueError(f"{self.name} streaming top-k arguments are invalid")
        groups = self.columns // GROUP_SIZE
        if groups % 2:
            raise CQ3RuntimeError(
                f"{self.name} scales are not independently byte-addressable"
            )
        scale_bytes = groups // 2
        retained_values = np.empty(0, dtype=np.float32)
        retained_indices = np.empty(0, dtype=np.int64)
        for start in range(0, self.rows, row_chunk):
            stop = min(start + row_chunk, self.rows)
            chunk = Qwen38CQ3AlignedTensor(
                name=f"{self.name}[{start}:{stop}]",
                shape=(stop - start, self.columns),
                rows=stop - start,
                columns=self.columns,
                packed_weights=self.packed_weights[start:stop],
                packed_scale_indices=self.packed_scale_indices[
                    start * scale_bytes : stop * scale_bytes
                ],
                row_anchors=self.row_anchors[start:stop],
                ratio_codebook=self.ratio_codebook,
            )
            values = chunk.gemv_q8(
                activation_q8,
                activation_scales,
                activation_sums,
                implementation=implementation,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
            )
            retained_values, retained_indices = _merge_topk(
                retained_values, retained_indices, values, start, count
            )
        return retained_values, retained_indices


@dataclass(frozen=True)
class Qwen38CQ3ScaleTensorView:
    """Authenticated row-scale4 views without reading the base-3 payload."""

    name: str
    shape: tuple[int, ...]
    rows: int
    columns: int
    packed_scale_indices: np.ndarray
    row_anchors: np.ndarray
    ratio_codebook: np.ndarray


@dataclass(frozen=True)
class Qwen38CQ3ExactTensorView:
    """Zero-copy raw view of one exactly retained Qwen3.8-27B tensor."""

    name: str
    shape: tuple[int, ...]
    source_dtype: str
    raw: np.ndarray

    def as_f32(self) -> np.ndarray:
        """Decode an authenticated retained tensor into a shaped FP32 array."""
        if self.source_dtype == "BF16":
            bits = self.raw.view("<u2").astype(np.uint32) << np.uint32(16)
            values = bits.view(np.float32)
        elif self.source_dtype == "F16":
            values = self.raw.view("<f2").astype(np.float32)
        elif self.source_dtype == "F32":
            values = self.raw.view("<f4").astype(np.float32, copy=False)
        else:
            raise CQ3RuntimeError(
                f"unsupported retained tensor dtype {self.source_dtype!r}"
            )
        return values.reshape(self.shape)


class Qwen38CQ3RuntimeImage:
    """Fail-closed mmap bridge from the canonical container to CPU kernels."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = Qwen38CQ3Reader(self.path)
        strategy = self._reader.directory.get("strategy")
        expected_strategy = {
            "name": ROW_NORMALIZED_ENCODING,
            "codec": "block203",
            "group_size": GROUP_SIZE,
            "scale_bits": 4,
            "row_anchor_dtype": "F16",
            "scale_codebook_entries": 16,
            "scale_codebook_dtype": "F16",
        }
        if strategy != expected_strategy:
            self._reader.close()
            raise CQ3RuntimeError(
                "CQ3 strategy is not the frozen Qwen3.8-27B row-normalized "
                "block203 runtime strategy"
            )
        self._mapping: np.memmap | None = np.memmap(
            self.path, dtype=np.uint8, mode="r"
        )
        self._closed = False
        self._verified_sections: set[str] = set()
        self.tensors: dict[str, dict] = {
            record["name"]: record
            for record in self._reader.directory["tensors"]
        }
        self.auxiliary_files: dict[str, dict] = {
            record["name"]: record
            for record in self._reader.directory["auxiliary_files"]
        }

    def __enter__(self) -> "Qwen38CQ3RuntimeImage":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close directory I/O; outstanding NumPy views keep their mmap alive."""
        if not self._closed:
            self._reader.close()
            self._mapping = None
            self._closed = True

    def _require_open(self) -> np.memmap:
        if self._closed or self._mapping is None:
            raise RuntimeError("Qwen3.8 CQ3 runtime image is closed")
        return self._mapping

    def _record(self, name: str) -> dict:
        self._require_open()
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"unknown Qwen3.8 CQ3 tensor {name!r}") from error

    def _verify_section(self, section_id: str) -> Mapping[str, object]:
        if section_id not in self._verified_sections:
            self._reader.verify_section(section_id)
            self._verified_sections.add(section_id)
        return self._reader.sections[section_id]

    def _u8_view(
        self, section: Mapping[str, object], start: int = 0, size: int | None = None
    ) -> np.ndarray:
        mapping = self._require_open()
        section_bytes = int(section["bytes"])
        stop = section_bytes if size is None else start + size
        if start < 0 or stop < start or stop > section_bytes:
            raise CQ3RuntimeError("runtime section slice exceeds its bounds")
        offset = int(section["offset"])
        result = mapping[offset + start : offset + stop]
        result.flags.writeable = False
        return result

    def _ternary_parts(
        self, name: str, *, authenticate_weights: bool
    ) -> tuple[
        tuple[int, ...],
        int,
        int,
        Mapping[str, object],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        tensor = self._record(name)
        if tensor.get("encoding") != ROW_NORMALIZED_ENCODING:
            raise CQ3RuntimeError(f"{name} is not a row-normalized ternary matrix")
        shape = tuple(int(value) for value in tensor["shape"])
        rows = shape[0]
        columns = prod(shape[1:])
        if len(shape) < 2 or columns <= 0 or columns % GROUP_SIZE:
            raise CQ3RuntimeError(f"{name} is not executable by group-128 GEMV")
        section_ids = tensor["sections"]
        if len(section_ids) != 2:
            raise CQ3RuntimeError(
                f"{name} does not have exactly code and scale sections"
            )
        code = self._reader.sections[section_ids[0]]
        if authenticate_weights:
            self._verify_section(section_ids[0])
        scales = self._verify_section(section_ids[1])
        if (
            code["kind"] != "ternary_block203"
            or scales["kind"] != ROW_NORMALIZED_SCALE_KIND
        ):
            raise CQ3RuntimeError(f"{name} section kinds differ from the runtime ABI")

        code_metadata = code["metadata"]
        expected_code_metadata = {
            "rows": rows,
            "columns": columns,
            "trits": rows * columns,
            "payload_bits": matrix_payload_bits(rows, columns),
            "group_trits": GROUP_SIZE,
            "full_group_bits": FULL_GROUP_BITS,
        }
        if (
            code_metadata != expected_code_metadata
            or int(code["bytes"]) != matrix_payload_bytes(rows, columns)
        ):
            raise CQ3RuntimeError(f"{name} block203 metadata or byte ledger changed")

        groups = columns // GROUP_SIZE
        scale_count = rows * groups
        index_bytes = (scale_count + 1) // 2
        anchor_bytes = rows * 2
        codebook_bytes = 16 * 2
        expected_scale_metadata = {
            "scale_count": scale_count,
            "index_bytes": index_bytes,
            "codebook_bytes": codebook_bytes,
            "codebook_entries": 16,
            "index_order": "low_nibble_first",
            "row_count": rows,
            "anchor_bytes": anchor_bytes,
            "factor_order": "indices_then_row_anchors_then_codebook",
            "decode": "float32(fp16_anchor) * float32(fp16_ratio)",
        }
        if (
            scales["metadata"] != expected_scale_metadata
            or int(scales["bytes"]) != index_bytes + anchor_bytes + codebook_bytes
        ):
            raise CQ3RuntimeError(f"{name} row-scale4 metadata or byte ledger changed")

        indices = self._u8_view(scales, 0, index_bytes)
        anchors = self._u8_view(scales, index_bytes, anchor_bytes).view("<f2")
        codebook = self._u8_view(
            scales, index_bytes + anchor_bytes, codebook_bytes
        ).view("<f2")
        for array in (indices, anchors, codebook):
            array.flags.writeable = False
        return shape, rows, columns, code, indices, anchors, codebook

    def ternary_tensor(self, name: str) -> Qwen38CQ3TernaryTensorView:
        """Authenticate and map one directly executable ternary matrix."""
        shape, rows, columns, code, indices, anchors, codebook = (
            self._ternary_parts(name, authenticate_weights=True)
        )
        weights = self._u8_view(code)
        weights.flags.writeable = False
        return Qwen38CQ3TernaryTensorView(
            name=name,
            shape=shape,
            rows=rows,
            columns=columns,
            packed_weights=weights,
            packed_scale_indices=indices,
            row_anchors=anchors,
            ratio_codebook=codebook,
        )

    def scale_tensor(self, name: str) -> Qwen38CQ3ScaleTensorView:
        """Authenticate only scales for a separately authenticated weight cache."""
        shape, rows, columns, _, indices, anchors, codebook = self._ternary_parts(
            name, authenticate_weights=False
        )
        return Qwen38CQ3ScaleTensorView(
            name=name,
            shape=shape,
            rows=rows,
            columns=columns,
            packed_scale_indices=indices,
            row_anchors=anchors,
            ratio_codebook=codebook,
        )

    def exact_tensor(self, name: str) -> Qwen38CQ3ExactTensorView:
        """Authenticate and map one exactly retained tensor as raw bytes."""
        tensor = self._record(name)
        encoding = tensor.get("encoding")
        section_ids = tensor["sections"]
        if (
            not isinstance(encoding, str)
            or not encoding.startswith("retained_exact_")
            or len(section_ids) != 1
        ):
            raise CQ3RuntimeError(f"{name} is not an exactly retained tensor")
        section = self._verify_section(section_ids[0])
        source_dtype = tensor.get("source_dtype")
        bytes_per_value = {"BF16": 2, "F16": 2, "F32": 4}.get(source_dtype)
        if (
            section["kind"] != encoding
            or section["metadata"] != {"dtype": source_dtype}
            or bytes_per_value is None
            or int(section["bytes"]) != int(tensor["parameters"]) * bytes_per_value
        ):
            raise CQ3RuntimeError(f"{name} exact tensor byte ledger changed")
        return Qwen38CQ3ExactTensorView(
            name=name,
            shape=tuple(int(value) for value in tensor["shape"]),
            source_dtype=source_dtype,
            raw=self._u8_view(section),
        )

    def auxiliary_file(self, name: str) -> np.ndarray:
        """Authenticate and map one bundled Qwen3.8-27B auxiliary file."""
        self._require_open()
        try:
            record = self.auxiliary_files[name]
        except KeyError as error:
            raise KeyError(f"unknown Qwen3.8 CQ3 auxiliary file {name!r}") from error
        section = self._verify_section(record["section"])
        if section["kind"] != "auxiliary_file" or section["metadata"] != {"name": name}:
            raise CQ3RuntimeError(f"auxiliary file {name!r} metadata changed")
        return self._u8_view(section)

    def verify_all(self) -> dict:
        """Authenticate every payload section and return the container receipt."""
        self._require_open()
        receipt = self._reader.verify_all()
        self._verified_sections.update(self._reader.sections)
        return receipt


@dataclass(frozen=True)
class Qwen38CQ3MLP:
    """Direct-storage Qwen3.8-27B gate/up/SwiGLU/down inference path."""

    gate: Qwen38CQ3TernaryTensorView
    up: Qwen38CQ3TernaryTensorView
    down: Qwen38CQ3TernaryTensorView

    def __post_init__(self) -> None:
        if (
            self.gate.columns != self.up.columns
            or self.gate.rows != self.up.rows
            or self.down.columns != self.gate.rows
            or self.down.rows != self.gate.columns
        ):
            raise CQ3RuntimeError(
                "Qwen3.8-27B MLP gate, up, and down dimensions do not compose"
            )

    @classmethod
    def from_image(
        cls, image: Qwen38CQ3RuntimeImage, layer: int
    ) -> "Qwen38CQ3MLP":
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be a non-negative integer")
        prefix = f"model.language_model.layers.{layer}.mlp"
        return cls(
            gate=image.ternary_tensor(f"{prefix}.gate_proj.weight"),
            up=image.ternary_tensor(f"{prefix}.up_proj.weight"),
            down=image.ternary_tensor(f"{prefix}.down_proj.weight"),
        )

    def forward(
        self,
        activation: np.ndarray,
        *,
        implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Execute one token's MLP with two A8 quantizations and no W expansion."""
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.gate.columns:
            raise ValueError(
                "Qwen3.8-27B MLP activation does not match the hidden dimension"
            )
        activation_q8, activation_scales, activation_sums = quantize_q8(
            values, GROUP_SIZE
        )
        arguments = {
            "implementation": implementation,
            "threads": threads,
            "executor": executor,
            "cpu_ids": cpu_ids,
        }
        gate = self.gate.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        )
        up = self.up.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        )
        intermediate = swiglu_f32(gate, up)
        intermediate_q8, intermediate_scales, intermediate_sums = quantize_q8(
            intermediate, GROUP_SIZE
        )
        return self.down.gemv_q8(
            intermediate_q8,
            intermediate_scales,
            intermediate_sums,
            **arguments,
        )


@dataclass(frozen=True)
class Qwen38CQ3DeltaNet:
    """One exact-shape Qwen3.8-27B single-token linear-attention path."""

    qkv: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    z: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    a: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    b: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    out: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    conv_weights: np.ndarray
    a_log: np.ndarray
    dt_bias: np.ndarray
    norm_weights: np.ndarray

    HIDDEN_SIZE = 5120
    KEY_HEADS = 16
    VALUE_HEADS = 48
    HEAD_DIM = 128
    KEY_DIM = KEY_HEADS * HEAD_DIM
    VALUE_DIM = VALUE_HEADS * HEAD_DIM
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM

    def __post_init__(self) -> None:
        expected_shapes = {
            "qkv": (self.CONV_DIM, self.HIDDEN_SIZE),
            "z": (self.VALUE_DIM, self.HIDDEN_SIZE),
            "a": (self.VALUE_HEADS, self.HIDDEN_SIZE),
            "b": (self.VALUE_HEADS, self.HIDDEN_SIZE),
            "out": (self.HIDDEN_SIZE, self.VALUE_DIM),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise CQ3RuntimeError(
                    f"Qwen3.8 DeltaNet {name} projection shape changed"
                )
        arrays = {
            "conv_weights": (4, self.CONV_DIM),
            "a_log": (self.VALUE_HEADS,),
            "dt_bias": (self.VALUE_HEADS,),
            "norm_weights": (self.HEAD_DIM,),
        }
        for name, shape in arrays.items():
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or not value.flags.c_contiguous
                or value.shape != shape
                or not np.all(np.isfinite(value))
            ):
                raise CQ3RuntimeError(
                    f"Qwen3.8 DeltaNet retained {name} contract changed"
                )

    @classmethod
    def from_image(
        cls, image: Qwen38CQ3RuntimeImage, layer: int
    ) -> "Qwen38CQ3DeltaNet":
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be non-negative")
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        conv = image.exact_tensor(f"{prefix}.conv1d.weight").as_f32()
        if conv.shape != (cls.CONV_DIM, 1, 4):
            raise CQ3RuntimeError("Qwen3.8 DeltaNet conv1d shape changed")
        return cls(
            qkv=image.ternary_tensor(f"{prefix}.in_proj_qkv.weight"),
            z=image.ternary_tensor(f"{prefix}.in_proj_z.weight"),
            a=image.ternary_tensor(f"{prefix}.in_proj_a.weight"),
            b=image.ternary_tensor(f"{prefix}.in_proj_b.weight"),
            out=image.ternary_tensor(f"{prefix}.out_proj.weight"),
            conv_weights=np.ascontiguousarray(conv[:, 0, :].T),
            a_log=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.A_log").as_f32()
            ),
            dt_bias=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.dt_bias").as_f32()
            ),
            norm_weights=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.norm.weight").as_f32()
            ),
        )

    def forward(
        self,
        activation: np.ndarray,
        conv_state: np.ndarray,
        recurrent_state: np.ndarray,
        *,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> np.ndarray:
        """Execute one token, mutating compact conv and recurrent state in place."""
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.HIDDEN_SIZE:
            raise ValueError(
                "Qwen3.8 DeltaNet activation does not match hidden size"
            )
        if (
            conv_state.dtype != np.float32
            or not conv_state.flags.c_contiguous
            or not conv_state.flags.writeable
            or conv_state.shape != (3, self.CONV_DIM)
            or recurrent_state.dtype != np.float32
            or not recurrent_state.flags.c_contiguous
            or not recurrent_state.flags.writeable
            or recurrent_state.shape
            != (self.VALUE_HEADS, self.HEAD_DIM, self.HEAD_DIM)
        ):
            raise ValueError("Qwen3.8 DeltaNet state shapes do not compose")

        activation_q8, activation_scales, activation_sums = quantize_q8(
            values, GROUP_SIZE
        )
        arguments = {
            "implementation": gemv_implementation,
            "threads": threads,
            "executor": executor,
            "cpu_ids": cpu_ids,
        }
        mixed_qkv = self.qkv.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        )
        z = self.z.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        ).reshape(self.VALUE_HEADS, self.HEAD_DIM)
        projected_a = self.a.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        )
        projected_b = self.b.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        )
        mixed_qkv = causal_conv4_silu_f32(
            mixed_qkv,
            conv_state,
            self.conv_weights,
            implementation=elementwise_implementation,
        )
        query = mixed_qkv[: self.KEY_DIM].reshape(
            self.KEY_HEADS, self.HEAD_DIM
        )
        key = mixed_qkv[self.KEY_DIM : 2 * self.KEY_DIM].reshape(
            self.KEY_HEADS, self.HEAD_DIM
        )
        value = mixed_qkv[2 * self.KEY_DIM :].reshape(
            self.VALUE_HEADS, self.HEAD_DIM
        )
        log_decay, beta = gated_delta_parameters_f32(
            projected_a, projected_b, self.a_log, self.dt_bias
        )
        recurrent = recurrent_gated_delta_f32(
            query,
            key,
            value,
            log_decay,
            beta,
            recurrent_state,
            implementation=elementwise_implementation,
            validated=validated_state,
        )
        gated = rmsnorm_silu_gate_f32(
            recurrent,
            self.norm_weights,
            z,
            implementation=elementwise_implementation,
        ).reshape(self.VALUE_DIM)
        gated_q8, gated_scales, gated_sums = quantize_q8(gated, GROUP_SIZE)
        return self.out.gemv_q8(
            gated_q8, gated_scales, gated_sums, **arguments
        )


@dataclass
class Qwen38AttentionState:
    """Head-major FP32 KV state for one Qwen3.8 full-attention layer."""

    key: np.ndarray
    value: np.ndarray
    length: int = 0

    @classmethod
    def allocate(cls, capacity: int) -> "Qwen38AttentionState":
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Qwen3.8 attention cache capacity must be positive")
        shape = (Qwen38CQ3FullAttention.KV_HEADS, capacity, Qwen38CQ3FullAttention.HEAD_DIM)
        return cls(np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        expected = (
            Qwen38CQ3FullAttention.KV_HEADS,
            Qwen38CQ3FullAttention.HEAD_DIM,
        )
        if (
            self.key.dtype != np.float32
            or self.value.dtype != np.float32
            or not self.key.flags.c_contiguous
            or not self.value.flags.c_contiguous
            or not self.key.flags.writeable
            or not self.value.flags.writeable
            or self.key.shape != self.value.shape
            or self.key.ndim != 3
            or self.key.shape[0] != expected[0]
            or self.key.shape[2] != expected[1]
            or key.shape != expected
            or value.shape != expected
            or self.length < 0
            or self.length >= self.key.shape[1]
        ):
            raise ValueError("Qwen3.8 attention KV state does not compose")
        self.key[:, self.length, :] = key
        self.value[:, self.length, :] = value
        self.length += 1


@dataclass(frozen=True)
class Qwen38CQ3FullAttention:
    """One exact-shape Qwen3.8-27B single-token full-attention path."""

    q: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    k: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    v: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    out: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    q_norm_offsets: np.ndarray
    k_norm_offsets: np.ndarray

    HIDDEN_SIZE = 5120
    QUERY_HEADS = 24
    KV_HEADS = 4
    HEAD_DIM = 256
    QUERY_DIM = QUERY_HEADS * HEAD_DIM
    KV_DIM = KV_HEADS * HEAD_DIM
    ROTARY_DIM = 64
    ROPE_THETA = 10_000_000.0

    def __post_init__(self) -> None:
        expected_shapes = {
            "q": (self.QUERY_DIM * 2, self.HIDDEN_SIZE),
            "k": (self.KV_DIM, self.HIDDEN_SIZE),
            "v": (self.KV_DIM, self.HIDDEN_SIZE),
            "out": (self.HIDDEN_SIZE, self.QUERY_DIM),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise CQ3RuntimeError(
                    f"Qwen3.8 full-attention {name} projection shape changed"
                )
        for name in ("q_norm_offsets", "k_norm_offsets"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or not value.flags.c_contiguous
                or value.shape != (self.HEAD_DIM,)
                or not np.all(np.isfinite(value))
            ):
                raise CQ3RuntimeError(
                    f"Qwen3.8 full-attention retained {name} contract changed"
                )

    @classmethod
    def from_image(
        cls, image: Qwen38CQ3RuntimeImage, layer: int
    ) -> "Qwen38CQ3FullAttention":
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be non-negative")
        prefix = f"model.language_model.layers.{layer}.self_attn"
        return cls(
            q=image.ternary_tensor(f"{prefix}.q_proj.weight"),
            k=image.ternary_tensor(f"{prefix}.k_proj.weight"),
            v=image.ternary_tensor(f"{prefix}.v_proj.weight"),
            out=image.ternary_tensor(f"{prefix}.o_proj.weight"),
            q_norm_offsets=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.q_norm.weight").as_f32()
            ),
            k_norm_offsets=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.k_norm.weight").as_f32()
            ),
        )

    @classmethod
    def text_rope(cls, position: int) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("Qwen3.8 text position must be non-negative")
        exponents = np.arange(0, cls.ROTARY_DIM, 2, dtype=np.float32) / np.float32(
            cls.ROTARY_DIM
        )
        inverse_frequencies = np.float32(1.0) / np.power(
            np.float32(cls.ROPE_THETA), exponents
        )
        frequencies = np.float32(position) * inverse_frequencies
        cosine_half = np.cos(frequencies).astype(np.float32)
        sine_half = np.sin(frequencies).astype(np.float32)
        return (
            np.concatenate((cosine_half, cosine_half)),
            np.concatenate((sine_half, sine_half)),
        )

    def forward(
        self,
        activation: np.ndarray,
        state: Qwen38AttentionState,
        position: int,
        *,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> np.ndarray:
        """Execute one text token and append normalized/rotated K/V to cache."""
        values = np.asarray(activation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.HIDDEN_SIZE:
            raise ValueError(
                "Qwen3.8 full-attention activation does not match hidden size"
            )
        if position != state.length:
            raise ValueError(
                "Qwen3.8 text position must equal the sequential KV length"
            )
        activation_q8, activation_scales, activation_sums = quantize_q8(
            values, GROUP_SIZE
        )
        arguments = {
            "implementation": gemv_implementation,
            "threads": threads,
            "executor": executor,
            "cpu_ids": cpu_ids,
        }
        projected_q = self.q.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        ).reshape(self.QUERY_HEADS, self.HEAD_DIM * 2)
        query = projected_q[:, : self.HEAD_DIM]
        gate = projected_q[:, self.HEAD_DIM :].reshape(self.QUERY_DIM)
        key = self.k.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        ).reshape(self.KV_HEADS, self.HEAD_DIM)
        value = self.v.gemv_q8(
            activation_q8, activation_scales, activation_sums, **arguments
        ).reshape(self.KV_HEADS, self.HEAD_DIM)
        cosine, sine = self.text_rope(position)
        query = rmsnorm_rope_f32(
            query,
            self.q_norm_offsets,
            cosine,
            sine,
            implementation=elementwise_implementation,
        )
        key = rmsnorm_rope_f32(
            key,
            self.k_norm_offsets,
            cosine,
            sine,
            implementation=elementwise_implementation,
        )
        state.append(key, value)
        attended = gqa_decode_f32(
            query,
            state.key,
            state.value,
            state.length,
            implementation=elementwise_implementation,
            validated=validated_state,
        ).reshape(self.QUERY_DIM)
        gated = sigmoid_gate_f32(attended, gate)
        gated_q8, gated_scales, gated_sums = quantize_q8(gated, GROUP_SIZE)
        return self.out.gemv_q8(
            gated_q8, gated_scales, gated_sums, **arguments
        )


@dataclass(frozen=True)
class Qwen38CQ3DecoderLayer:
    """One Qwen3.8 decoder layer with both residual paths composed."""

    mixer_kind: str
    mixer: Qwen38CQ3DeltaNet | Qwen38CQ3FullAttention
    mlp: Qwen38CQ3MLP
    input_norm_offsets: np.ndarray
    post_attention_norm_offsets: np.ndarray

    HIDDEN_SIZE = 5120

    def __post_init__(self) -> None:
        if self.mixer_kind not in ("linear_attention", "full_attention"):
            raise CQ3RuntimeError("Qwen3.8 decoder mixer kind changed")
        for name in ("input_norm_offsets", "post_attention_norm_offsets"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or not value.flags.c_contiguous
                or value.shape != (self.HIDDEN_SIZE,)
                or not np.all(np.isfinite(value))
            ):
                raise CQ3RuntimeError(
                    f"Qwen3.8 decoder retained {name} contract changed"
                )

    @classmethod
    def from_image(
        cls, image: Qwen38CQ3RuntimeImage, layer: int
    ) -> "Qwen38CQ3DecoderLayer":
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be non-negative")
        config = json.loads(image.auxiliary_file("config.json").tobytes())
        text_config = config.get("text_config", {})
        layer_types = text_config.get("layer_types")
        if (
            not isinstance(layer_types, list)
            or layer >= len(layer_types)
            or text_config.get("hidden_size") != cls.HIDDEN_SIZE
        ):
            raise CQ3RuntimeError("Qwen3.8 decoder config changed")
        mixer_kind = layer_types[layer]
        if mixer_kind == "linear_attention":
            mixer = Qwen38CQ3DeltaNet.from_image(image, layer)
        elif mixer_kind == "full_attention":
            mixer = Qwen38CQ3FullAttention.from_image(image, layer)
        else:
            raise CQ3RuntimeError("Qwen3.8 decoder layer type changed")
        prefix = f"model.language_model.layers.{layer}"
        return cls(
            mixer_kind=mixer_kind,
            mixer=mixer,
            mlp=Qwen38CQ3MLP.from_image(image, layer),
            input_norm_offsets=np.ascontiguousarray(
                image.exact_tensor(f"{prefix}.input_layernorm.weight").as_f32()
            ),
            post_attention_norm_offsets=np.ascontiguousarray(
                image.exact_tensor(
                    f"{prefix}.post_attention_layernorm.weight"
                ).as_f32()
            ),
        )

    def forward(
        self,
        hidden_state: np.ndarray,
        mixer_state: tuple[np.ndarray, np.ndarray] | Qwen38AttentionState,
        position: int,
        *,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> np.ndarray:
        values = np.asarray(hidden_state, dtype=np.float32)
        if values.ndim != 1 or values.size != self.HIDDEN_SIZE:
            raise ValueError("Qwen3.8 decoder hidden state shape changed")
        normalized = rmsnorm_offset_f32(
            values,
            self.input_norm_offsets,
            implementation=elementwise_implementation,
        )
        common = {
            "gemv_implementation": gemv_implementation,
            "elementwise_implementation": elementwise_implementation,
            "threads": threads,
            "executor": executor,
            "cpu_ids": cpu_ids,
            "validated_state": validated_state,
        }
        if self.mixer_kind == "linear_attention":
            if not isinstance(mixer_state, tuple) or len(mixer_state) != 2:
                raise ValueError("Qwen3.8 DeltaNet decoder state is absent")
            mixed = self.mixer.forward(
                normalized, mixer_state[0], mixer_state[1], **common
            )
        else:
            if not isinstance(mixer_state, Qwen38AttentionState):
                raise ValueError("Qwen3.8 full-attention decoder state is absent")
            mixed = self.mixer.forward(
                normalized, mixer_state, position, **common
            )
        after_mixer = values + mixed
        post_normalized = rmsnorm_offset_f32(
            after_mixer,
            self.post_attention_norm_offsets,
            implementation=elementwise_implementation,
        )
        mlp_output = self.mlp.forward(
            post_normalized,
            implementation=gemv_implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
        )
        output = after_mixer + mlp_output
        if not np.all(np.isfinite(output)):
            raise ValueError("Qwen3.8 decoder produced non-finite output")
        return output


@dataclass
class Qwen38TextState:
    """All mutable batch-one text state for the 64-layer Qwen3.8 graph."""

    layer_states: list[tuple[np.ndarray, np.ndarray] | Qwen38AttentionState]
    position: int = 0

    @classmethod
    def allocate(
        cls, layer_types: tuple[str, ...], context_capacity: int
    ) -> "Qwen38TextState":
        if len(layer_types) != Qwen38CQ3TextModel.LAYERS:
            raise ValueError("Qwen3.8 text layer inventory changed")
        states: list[tuple[np.ndarray, np.ndarray] | Qwen38AttentionState] = []
        for layer_type in layer_types:
            if layer_type == "linear_attention":
                states.append(
                    (
                        np.zeros(
                            (3, Qwen38CQ3DeltaNet.CONV_DIM), dtype=np.float32
                        ),
                        np.zeros(
                            (
                                Qwen38CQ3DeltaNet.VALUE_HEADS,
                                Qwen38CQ3DeltaNet.HEAD_DIM,
                                Qwen38CQ3DeltaNet.HEAD_DIM,
                            ),
                            dtype=np.float32,
                        ),
                    )
                )
            elif layer_type == "full_attention":
                states.append(Qwen38AttentionState.allocate(context_capacity))
            else:
                raise ValueError("Qwen3.8 text layer type changed")
        return cls(states)


@dataclass(frozen=True)
class Qwen38CQ3TextModel:
    """Text-only batch-one Qwen3.8-27B decode graph over CQ3 tensors."""

    embedding: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor
    layers: tuple[Qwen38CQ3DecoderLayer, ...]
    layer_types: tuple[str, ...]
    final_norm_offsets: np.ndarray
    lm_head: Qwen38CQ3TernaryTensorView | Qwen38CQ3AlignedTensor

    HIDDEN_SIZE = 5120
    VOCAB_SIZE = 248320
    LAYERS = 64

    def __post_init__(self) -> None:
        if (
            tuple(self.embedding.shape) != (self.VOCAB_SIZE, self.HIDDEN_SIZE)
            or tuple(self.lm_head.shape) != (self.VOCAB_SIZE, self.HIDDEN_SIZE)
            or len(self.layers) != self.LAYERS
            or len(self.layer_types) != self.LAYERS
            or tuple(layer.mixer_kind for layer in self.layers) != self.layer_types
            or self.final_norm_offsets.dtype != np.float32
            or not self.final_norm_offsets.flags.c_contiguous
            or self.final_norm_offsets.shape != (self.HIDDEN_SIZE,)
            or not np.all(np.isfinite(self.final_norm_offsets))
        ):
            raise CQ3RuntimeError("Qwen3.8 text graph contract changed")

    @classmethod
    def from_image(
        cls, image: Qwen38CQ3RuntimeImage
    ) -> "Qwen38CQ3TextModel":
        config = json.loads(image.auxiliary_file("config.json").tobytes())
        text_config = config.get("text_config", {})
        layer_types = tuple(text_config.get("layer_types", ()))
        if (
            text_config.get("hidden_size") != cls.HIDDEN_SIZE
            or text_config.get("vocab_size") != cls.VOCAB_SIZE
            or text_config.get("num_hidden_layers") != cls.LAYERS
            or len(layer_types) != cls.LAYERS
        ):
            raise CQ3RuntimeError("Qwen3.8 text config changed")
        return cls(
            embedding=image.ternary_tensor(
                "model.language_model.embed_tokens.weight"
            ),
            layers=tuple(
                Qwen38CQ3DecoderLayer.from_image(image, layer)
                for layer in range(cls.LAYERS)
            ),
            layer_types=layer_types,
            final_norm_offsets=np.ascontiguousarray(
                image.exact_tensor("model.language_model.norm.weight").as_f32()
            ),
            lm_head=image.ternary_tensor("lm_head.weight"),
        )

    def allocate_state(self, context_capacity: int) -> Qwen38TextState:
        return Qwen38TextState.allocate(self.layer_types, context_capacity)

    def _forward_hidden(
        self,
        token_id: int,
        state: Qwen38TextState,
        *,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> np.ndarray:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < self.VOCAB_SIZE
        ):
            raise ValueError("Qwen3.8 token id is out of range")
        if len(state.layer_states) != self.LAYERS or state.position < 0:
            raise ValueError("Qwen3.8 text state inventory changed")
        hidden = self.embedding.decode_row_f32(token_id)
        for layer, layer_state in zip(self.layers, state.layer_states):
            hidden = layer.forward(
                hidden,
                layer_state,
                state.position,
                gemv_implementation=gemv_implementation,
                elementwise_implementation=elementwise_implementation,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
                validated_state=validated_state,
            )
        hidden = rmsnorm_offset_f32(
            hidden,
            self.final_norm_offsets,
            implementation=elementwise_implementation,
        )
        return hidden

    def forward_token(
        self,
        token_id: int,
        state: Qwen38TextState,
        *,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> np.ndarray:
        hidden = self._forward_hidden(
            token_id,
            state,
            gemv_implementation=gemv_implementation,
            elementwise_implementation=elementwise_implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            validated_state=validated_state,
        )
        hidden_q8, hidden_scales, hidden_sums = quantize_q8(hidden, GROUP_SIZE)
        logits = self.lm_head.gemv_q8(
            hidden_q8,
            hidden_scales,
            hidden_sums,
            implementation=gemv_implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
        )
        if logits.shape != (self.VOCAB_SIZE,) or not np.all(np.isfinite(logits)):
            raise ValueError("Qwen3.8 text graph produced invalid logits")
        state.position += 1
        return logits

    def forward_token_topk(
        self,
        token_id: int,
        state: Qwen38TextState,
        count: int,
        *,
        row_chunk: int = 4096,
        gemv_implementation: str = "scalar_chunk16",
        elementwise_implementation: str = "scalar",
        threads: int = 1,
        executor: ThreadPoolExecutor | None = None,
        cpu_ids: list[int] | None = None,
        validated_state: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden = self._forward_hidden(
            token_id,
            state,
            gemv_implementation=gemv_implementation,
            elementwise_implementation=elementwise_implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            validated_state=validated_state,
        )
        hidden_q8, hidden_scales, hidden_sums = quantize_q8(hidden, GROUP_SIZE)
        values, indices = self.lm_head.topk_q8(
            hidden_q8,
            hidden_scales,
            hidden_sums,
            count,
            row_chunk=row_chunk,
            implementation=gemv_implementation,
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
        )
        if (
            values.shape != (count,)
            or indices.shape != (count,)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Qwen3.8 streaming LM head produced invalid top-k")
        state.position += 1
        return values, indices

    @staticmethod
    def topk(logits: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(logits, dtype=np.float32)
        if (
            values.ndim != 1
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > values.size
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Qwen3.8 top-k inputs are invalid")
        all_indices = np.arange(values.size, dtype=np.int64)
        indices = np.lexsort((all_indices, -values))[:count]
        return values[indices], indices
