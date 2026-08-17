"""Read-only native-compute views over a Qwen3.8-27B CQ3 container.

The canonical container stays the storage source of truth.  This module maps
its authenticated sections directly into the block203 plus row-scale4 CPU
kernel without expanding the ternary weights into a larger compute image.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Mapping

import numpy as np

from .base3_block203 import matrix_payload_bits, matrix_payload_bytes
from .qwen38_cq3_container import CQ3ContainerError, Qwen38CQ3Reader
from .ternary_cpu import (
    causal_conv4_silu_f32,
    gated_delta_parameters_f32,
    gemv_q8_group128_2bit_row_scale4,
    gemv_q8_group128_block203_row_scale4,
    quantize_q8,
    recurrent_gated_delta_f32,
    rmsnorm_silu_gate_f32,
    swiglu_f32,
    transcode_block203_rows_to_group128_2bit,
)


ROW_NORMALIZED_ENCODING = "base3_group128_row_normalized_codebook4"
ROW_NORMALIZED_SCALE_KIND = (
    "scale_indices_u4_anchors_f16_and_codebook_f16"
)
GROUP_SIZE = 128
FULL_GROUP_BITS = 203


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
