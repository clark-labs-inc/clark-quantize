"""Restartable derived aligned-2 cache for the Qwen3.8-27B CQ3 image.

The row-normalized CQ3 container remains the canonical storage artifact.  This
module derives only the wider SIMD weight bytes, keeps per-tensor commit
records, and publishes an authenticated manifest after every tensor has been
independently verified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np

from .base3_chunks import atomic_bytes, canonical_json
from .qwen38_cq3_container import MODEL_ID, Qwen38CQ3Reader, align_up
from .qwen38_cq3_runtime import (
    ROW_NORMALIZED_ENCODING,
    CQ3RuntimeError,
    Qwen38CQ3AlignedTensor,
    Qwen38CQ3DeltaNet,
    Qwen38CQ3MLP,
    Qwen38CQ3RuntimeImage,
)
from .ternary_cpu import native_block203_transcode_available


CACHE_ALIGNMENT = 64
CACHE_STATE_SCHEMA = "qwen38_27b_aligned2_cache_state_v1"
CACHE_MANIFEST_SCHEMA = "qwen38_27b_aligned2_cache_manifest_v1"
COPY_BYTES = 8 * 1024 * 1024


class Qwen38Aligned2CacheError(ValueError):
    """A derived-cache identity, byte ledger, or checkpoint is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for content in iter(lambda: handle.read(COPY_BYTES), b""):
            digest.update(content)
    return digest.hexdigest()


def _sha256_region(handle: BinaryIO, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    handle.seek(offset)
    remaining = size
    while remaining:
        content = handle.read(min(COPY_BYTES, remaining))
        if not content:
            raise Qwen38Aligned2CacheError("aligned2 cache region is truncated")
        digest.update(content)
        remaining -= len(content)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all_at(descriptor: int, offset: int, content: memoryview) -> None:
    written = 0
    while written < len(content):
        count = os.pwrite(descriptor, content[written:], offset + written)
        if count <= 0:
            raise OSError("aligned2 cache pwrite made no progress")
        written += count


def _load_canonical_json(path: Path) -> dict:
    content = path.read_bytes()
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen38Aligned2CacheError(f"invalid JSON checkpoint {path}") from error
    if not isinstance(value, dict) or canonical_json(value) != content:
        raise Qwen38Aligned2CacheError(f"non-canonical JSON checkpoint {path}")
    return value


class Qwen38Aligned2CacheBuilder:
    """Build one fixed-layout cache with an atomic checkpoint per tensor."""

    def __init__(
        self,
        container_path: str | Path,
        cache_path: str | Path,
        *,
        state_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.container_path = Path(container_path)
        self.cache_path = Path(cache_path)
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else Path(str(self.cache_path) + ".partial.json")
        )
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else Path(str(self.cache_path) + ".manifest.json")
        )
        if len({self.cache_path, self.state_path, self.manifest_path}) != 3:
            raise ValueError("cache, checkpoint, and manifest paths must differ")

    def _source_and_layout(self) -> tuple[dict, list[dict], dict]:
        if not self.container_path.is_file():
            raise Qwen38Aligned2CacheError("canonical Qwen3.8-27B container is absent")
        container_sha256 = _sha256_file(self.container_path)
        with Qwen38CQ3Reader(self.container_path) as reader:
            model = reader.directory["model"]
            if model.get("id") != MODEL_ID:
                raise Qwen38Aligned2CacheError("cache source is not Qwen3.8-27B")
            source = {
                "path": str(self.container_path.resolve()),
                "bytes": reader.file_bytes,
                "sha256": container_sha256,
                "root_sha256": reader.header["root_sha256"],
                "directory_sha256": reader.header["directory_sha256"],
                "model": dict(model),
                "parameters": reader.header["parameter_count"],
            }
            layout: list[dict] = []
            offset = 0
            ternary_parameters = 0
            row_scale_bytes = 0
            retained_exact_bytes = 0
            retained_exact_parameters = 0
            for tensor in reader.directory["tensors"]:
                encoding = tensor.get("encoding")
                parameters = int(tensor["parameters"])
                if encoding == ROW_NORMALIZED_ENCODING:
                    shape = [int(value) for value in tensor["shape"]]
                    rows = shape[0]
                    columns = parameters // rows
                    if (
                        parameters % rows
                        or columns % 128
                        or parameters % 4
                        or len(tensor["sections"]) != 2
                    ):
                        raise Qwen38Aligned2CacheError(
                            f"tensor {tensor['name']} cannot use aligned2"
                        )
                    code = reader.sections[tensor["sections"][0]]
                    scales = reader.sections[tensor["sections"][1]]
                    offset = align_up(offset, CACHE_ALIGNMENT)
                    size = parameters // 4
                    layout.append(
                        {
                            "name": tensor["name"],
                            "shape": shape,
                            "rows": rows,
                            "columns": columns,
                            "parameters": parameters,
                            "offset": offset,
                            "bytes": size,
                            "source_code_section": code["id"],
                            "source_code_sha256": code["sha256"],
                            "source_scale_section": scales["id"],
                            "source_scale_sha256": scales["sha256"],
                            "source_scale_bytes": scales["bytes"],
                        }
                    )
                    offset += size
                    ternary_parameters += parameters
                    row_scale_bytes += int(scales["bytes"])
                elif isinstance(encoding, str) and encoding.startswith(
                    "retained_exact_"
                ):
                    if len(tensor["sections"]) != 1:
                        raise Qwen38Aligned2CacheError(
                            f"retained tensor {tensor['name']} has invalid sections"
                        )
                    section = reader.sections[tensor["sections"][0]]
                    retained_exact_bytes += int(section["bytes"])
                    retained_exact_parameters += parameters
                else:
                    raise Qwen38Aligned2CacheError(
                        f"tensor {tensor['name']} has unsupported encoding {encoding!r}"
                    )
            if ternary_parameters + retained_exact_parameters != source["parameters"]:
                raise Qwen38Aligned2CacheError(
                    "aligned2 source tensor parameter ledger is incomplete"
                )
            cache_payload_bytes = sum(record["bytes"] for record in layout)
            cache_file_bytes = offset
            runtime_bytes = cache_file_bytes + row_scale_bytes + retained_exact_bytes
            ledger = {
                "tensor_count": len(layout),
                "ternary_parameters": ternary_parameters,
                "retained_exact_parameters": retained_exact_parameters,
                "cache_payload_bytes": cache_payload_bytes,
                "cache_file_bytes": cache_file_bytes,
                "cache_alignment_padding_bytes": cache_file_bytes
                - cache_payload_bytes,
                "row_scale_bytes_shared_from_canonical_container": row_scale_bytes,
                "retained_exact_bytes_shared_from_canonical_container": (
                    retained_exact_bytes
                ),
                "active_runtime_weight_bytes": runtime_bytes,
                "active_runtime_bits_per_model_parameter": (
                    runtime_bytes * 8 / source["parameters"]
                ),
                "combined_canonical_and_cache_disk_bytes": (
                    source["bytes"] + cache_file_bytes
                ),
            }
        return source, layout, ledger

    def _identity(self, source: Mapping, layout: list[dict], ledger: Mapping) -> dict:
        return {
            "schema": "qwen38_27b_aligned2_cache_identity_v1",
            "source": dict(source),
            "cache_path": str(self.cache_path.resolve()),
            "alignment": CACHE_ALIGNMENT,
            "layout": layout,
            "ledger": dict(ledger),
        }

    def _new_state(self, identity: Mapping) -> dict:
        return {
            "schema": CACHE_STATE_SCHEMA,
            "identity_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
            "identity": dict(identity),
            "cache_initialized": False,
            "completed": [],
            "complete": False,
        }

    def _validate_state(self, state: dict, identity: Mapping) -> None:
        expected_sha256 = hashlib.sha256(canonical_json(identity)).hexdigest()
        if (
            state.get("schema") != CACHE_STATE_SCHEMA
            or state.get("identity_sha256") != expected_sha256
            or state.get("identity") != identity
            or not isinstance(state.get("cache_initialized"), bool)
            or not isinstance(state.get("completed"), list)
            or not isinstance(state.get("complete"), bool)
        ):
            raise Qwen38Aligned2CacheError(
                "aligned2 checkpoint differs from the source or fixed layout"
            )

    def _initialize_cache(self, state: dict, cache_bytes: int) -> dict:
        if state["cache_initialized"]:
            if (
                not self.cache_path.is_file()
                or self.cache_path.stat().st_size != cache_bytes
            ):
                raise Qwen38Aligned2CacheError(
                    "initialized aligned2 cache file has the wrong size"
                )
            return state
        if state["completed"]:
            raise Qwen38Aligned2CacheError(
                "uninitialized aligned2 cache checkpoint has completed tensors"
            )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if self.cache_path.exists() else "x+b"
        with self.cache_path.open(mode) as handle:
            if handle.seek(0, os.SEEK_END) not in (0, cache_bytes):
                raise Qwen38Aligned2CacheError(
                    "uninitialized aligned2 cache file has an unexpected size"
                )
            descriptor = handle.fileno()
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(descriptor, 0, cache_bytes)
            else:
                os.ftruncate(descriptor, cache_bytes)
            handle.flush()
            os.fsync(descriptor)
        _fsync_directory(self.cache_path.parent)
        state["cache_initialized"] = True
        atomic_bytes(self.state_path, canonical_json(state))
        return state

    def _verify_completed(
        self, state: dict, layout: list[dict], handle: BinaryIO
    ) -> None:
        completed = state["completed"]
        if len(completed) > len(layout):
            raise Qwen38Aligned2CacheError("aligned2 checkpoint exceeds its layout")
        for index, record in enumerate(completed):
            expected = layout[index]
            if (
                not isinstance(record, dict)
                or record.get("name") != expected["name"]
                or record.get("offset") != expected["offset"]
                or record.get("bytes") != expected["bytes"]
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
            ):
                raise Qwen38Aligned2CacheError(
                    "aligned2 checkpoint is not a committed layout prefix"
                )
            observed = _sha256_region(
                handle, expected["offset"], expected["bytes"]
            )
            if observed != record["sha256"]:
                raise Qwen38Aligned2CacheError(
                    f"aligned2 checkpoint tensor {expected['name']} changed"
                )

    def _verify_padding(self, layout: list[dict], handle: BinaryIO) -> None:
        previous_end = 0
        for record in layout:
            gap = record["offset"] - previous_end
            if gap:
                handle.seek(previous_end)
                if any(handle.read(gap)):
                    raise Qwen38Aligned2CacheError(
                        "aligned2 cache alignment padding is nonzero"
                    )
            previous_end = record["offset"] + record["bytes"]

    def verify_manifest(self, *, verify_full_cache: bool = True) -> dict:
        manifest = _load_canonical_json(self.manifest_path)
        source, layout, ledger = self._source_and_layout()
        identity = self._identity(source, layout, ledger)
        cache = manifest.get("cache", {})
        if (
            manifest.get("schema") != CACHE_MANIFEST_SCHEMA
            or manifest.get("identity_sha256")
            != hashlib.sha256(canonical_json(identity)).hexdigest()
            or manifest.get("identity") != identity
            or cache.get("path") != str(self.cache_path.resolve())
            or cache.get("bytes") != ledger["cache_file_bytes"]
            or not self.cache_path.is_file()
            or self.cache_path.stat().st_size != ledger["cache_file_bytes"]
            or len(manifest.get("tensors", [])) != len(layout)
        ):
            raise Qwen38Aligned2CacheError("aligned2 final manifest changed")
        with self.cache_path.open("rb") as handle:
            for expected, tensor in zip(layout, manifest["tensors"]):
                if any(
                    tensor.get(key) != expected[key]
                    for key in ("name", "offset", "bytes")
                ):
                    raise Qwen38Aligned2CacheError(
                        "aligned2 final tensor layout changed"
                    )
                observed = _sha256_region(
                    handle, expected["offset"], expected["bytes"]
                )
                if observed != tensor.get("sha256"):
                    raise Qwen38Aligned2CacheError(
                        f"aligned2 final tensor {expected['name']} changed"
                    )
            self._verify_padding(layout, handle)
        if verify_full_cache and _sha256_file(self.cache_path) != cache.get("sha256"):
            raise Qwen38Aligned2CacheError("aligned2 final cache SHA-256 changed")
        return manifest

    def build(
        self,
        *,
        max_new_tensors: int | None = None,
        workers: int = 1,
    ) -> dict:
        """Resume the build, optionally stopping after a bounded new prefix."""
        if max_new_tensors is not None and max_new_tensors < 0:
            raise ValueError("max_new_tensors must be non-negative")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if not native_block203_transcode_available():
            raise Qwen38Aligned2CacheError(
                "native streaming block203 transcode is required"
            )
        if self.manifest_path.exists():
            return self.verify_manifest()

        source, layout, ledger = self._source_and_layout()
        identity = self._identity(source, layout, ledger)
        if self.state_path.exists():
            state = _load_canonical_json(self.state_path)
            self._validate_state(state, identity)
        else:
            if self.cache_path.exists():
                raise Qwen38Aligned2CacheError(
                    "aligned2 cache file exists without its checkpoint"
                )
            state = self._new_state(identity)
            atomic_bytes(self.state_path, canonical_json(state))
        state = self._initialize_cache(state, ledger["cache_file_bytes"])

        new_tensors = 0
        # Unbuffered I/O is required because pwrite bypasses Python's read
        # buffer and each committed region is immediately read back and hashed.
        with self.cache_path.open("r+b", buffering=0) as cache_handle:
            self._verify_completed(state, layout, cache_handle)
            with Qwen38CQ3RuntimeImage(self.container_path) as image:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    while len(state["completed"]) < len(layout):
                        remaining = len(layout) - len(state["completed"])
                        if max_new_tensors is not None:
                            allowed = max_new_tensors - new_tensors
                            if allowed <= 0:
                                return state
                            remaining = min(remaining, allowed)
                        batch_size = min(workers, remaining)
                        start = len(state["completed"])
                        records = layout[start : start + batch_size]
                        # Authentication uses the reader's seekable handle, so
                        # views are created serially before native transcodes
                        # run concurrently over independent mmap regions.
                        tensors = [
                            image.ternary_tensor(record["name"])
                            for record in records
                        ]
                        aligned_tensors = list(
                            executor.map(
                                lambda tensor: tensor.derive_aligned2(),
                                tensors,
                            )
                        )
                        for record, aligned in zip(records, aligned_tensors):
                            weights = memoryview(
                                aligned.packed_weights
                            ).cast("B")
                            if len(weights) != record["bytes"]:
                                raise Qwen38Aligned2CacheError(
                                    f"aligned2 tensor {record['name']} byte "
                                    "ledger changed"
                                )
                            digest = hashlib.sha256(weights).hexdigest()
                            _write_all_at(
                                cache_handle.fileno(),
                                record["offset"],
                                weights,
                            )
                            os.fsync(cache_handle.fileno())
                            state["completed"].append(
                                {
                                    "name": record["name"],
                                    "offset": record["offset"],
                                    "bytes": record["bytes"],
                                    "sha256": digest,
                                }
                            )
                            atomic_bytes(
                                self.state_path, canonical_json(state)
                            )
                            new_tensors += 1
            self._verify_completed(state, layout, cache_handle)
            self._verify_padding(layout, cache_handle)

        cache_sha256 = _sha256_file(self.cache_path)
        manifest = {
            "schema": CACHE_MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "identity_sha256": state["identity_sha256"],
            "identity": identity,
            "cache": {
                "path": str(self.cache_path.resolve()),
                "bytes": ledger["cache_file_bytes"],
                "sha256": cache_sha256,
                "full_int8_weight_matrix_materialized": False,
            },
            "tensors": list(state["completed"]),
            "complete": True,
            "claim_boundary": {
                "qwen38_27b_derived_weight_cache_complete": True,
                "canonical_container_remains_source_of_truth": True,
                "whole_model_graph_runnable": False,
                "capability_or_perplexity_proven": False,
                "end_to_end_tokens_per_second_proven": False,
                "scientific_breakthrough_proven": False,
            },
        }
        atomic_bytes(self.manifest_path, canonical_json(manifest))
        state["complete"] = True
        state["cache_sha256"] = cache_sha256
        state["manifest_sha256"] = _sha256_file(self.manifest_path)
        atomic_bytes(self.state_path, canonical_json(state))
        return manifest


class Qwen38Aligned2RuntimeImage:
    """Authenticated mmap execution over aligned weights and canonical scales."""

    def __init__(
        self,
        container_path: str | Path,
        cache_path: str | Path,
        manifest_path: str | Path,
        *,
        verify_source_container: bool = False,
        verify_full_cache: bool = False,
    ) -> None:
        self.container_path = Path(container_path)
        self.cache_path = Path(cache_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = _load_canonical_json(self.manifest_path)
        identity = self.manifest.get("identity")
        if not isinstance(identity, dict):
            raise Qwen38Aligned2CacheError("aligned2 runtime identity is absent")
        identity_sha256 = hashlib.sha256(canonical_json(identity)).hexdigest()
        cache = self.manifest.get("cache", {})
        if (
            self.manifest.get("schema") != CACHE_MANIFEST_SCHEMA
            or self.manifest.get("complete") is not True
            or self.manifest.get("identity_sha256") != identity_sha256
            or identity.get("schema")
            != "qwen38_27b_aligned2_cache_identity_v1"
            or identity.get("alignment") != CACHE_ALIGNMENT
            or identity.get("cache_path") != str(self.cache_path.resolve())
            or cache.get("path") != str(self.cache_path.resolve())
            or cache.get("full_int8_weight_matrix_materialized") is not False
        ):
            raise Qwen38Aligned2CacheError("aligned2 runtime manifest changed")
        source_identity = identity.get("source", {})
        if source_identity.get("path") != str(self.container_path.resolve()):
            raise Qwen38Aligned2CacheError("aligned2 runtime source path changed")
        if not self.cache_path.is_file():
            raise Qwen38Aligned2CacheError("aligned2 runtime cache is absent")
        cache_bytes = int(cache.get("bytes", -1))
        if (
            cache_bytes < 0
            or self.cache_path.stat().st_size != cache_bytes
            or identity.get("ledger", {}).get("cache_file_bytes")
            != cache_bytes
        ):
            raise Qwen38Aligned2CacheError("aligned2 runtime cache size changed")

        self.source = Qwen38CQ3RuntimeImage(self.container_path)
        try:
            source_reader = self.source._reader
            source_model = source_reader.directory["model"]
            if (
                source_identity.get("bytes") != source_reader.file_bytes
                or source_identity.get("root_sha256")
                != source_reader.header["root_sha256"]
                or source_identity.get("directory_sha256")
                != source_reader.header["directory_sha256"]
                or source_identity.get("model") != source_model
                or source_identity.get("parameters")
                != source_reader.header["parameter_count"]
                or source_model.get("id") != MODEL_ID
            ):
                raise Qwen38Aligned2CacheError(
                    "aligned2 runtime source identity changed"
                )
            if (
                verify_source_container
                and _sha256_file(self.container_path)
                != source_identity.get("sha256")
            ):
                raise Qwen38Aligned2CacheError(
                    "aligned2 runtime source SHA-256 changed"
                )
            layout = identity.get("layout")
            tensors = self.manifest.get("tensors")
            if (
                not isinstance(layout, list)
                or not isinstance(tensors, list)
                or len(layout) != len(tensors)
            ):
                raise Qwen38Aligned2CacheError(
                    "aligned2 runtime tensor inventory changed"
                )
            self.records: dict[str, dict] = {}
            previous_end = 0
            for layout_record, tensor_record in zip(layout, tensors):
                name = layout_record.get("name")
                offset = layout_record.get("offset")
                size = layout_record.get("bytes")
                if (
                    not isinstance(name, str)
                    or not name
                    or name in self.records
                    or not isinstance(offset, int)
                    or offset < previous_end
                    or offset % CACHE_ALIGNMENT
                    or not isinstance(size, int)
                    or size <= 0
                    or offset + size > cache_bytes
                    or any(
                        tensor_record.get(key) != layout_record.get(key)
                        for key in ("name", "offset", "bytes")
                    )
                    or not isinstance(tensor_record.get("sha256"), str)
                    or len(tensor_record["sha256"]) != 64
                ):
                    raise Qwen38Aligned2CacheError(
                        "aligned2 runtime tensor record is invalid"
                    )
                try:
                    source_tensor = self.source.tensors[name]
                    code_id, scale_id = source_tensor["sections"]
                    code = source_reader.sections[code_id]
                    scales = source_reader.sections[scale_id]
                except (KeyError, ValueError, TypeError, IndexError) as error:
                    raise Qwen38Aligned2CacheError(
                        f"aligned2 source tensor {name} is absent"
                    ) from error
                if (
                    source_tensor.get("encoding") != ROW_NORMALIZED_ENCODING
                    or code_id != layout_record.get("source_code_section")
                    or code["sha256"]
                    != layout_record.get("source_code_sha256")
                    or scale_id != layout_record.get("source_scale_section")
                    or scales["sha256"]
                    != layout_record.get("source_scale_sha256")
                    or scales["bytes"]
                    != layout_record.get("source_scale_bytes")
                ):
                    raise Qwen38Aligned2CacheError(
                        f"aligned2 source tensor {name} identity changed"
                    )
                self.records[name] = {
                    **layout_record,
                    "sha256": tensor_record["sha256"],
                }
                previous_end = offset + size
            if previous_end != cache_bytes:
                raise Qwen38Aligned2CacheError(
                    "aligned2 runtime cache envelope changed"
                )
            self._mapping: np.memmap | None = np.memmap(
                self.cache_path, dtype=np.uint8, mode="r"
            )
            self._verified_tensors: set[str] = set()
            self._closed = False
            if verify_full_cache and _sha256_file(self.cache_path) != cache.get(
                "sha256"
            ):
                raise Qwen38Aligned2CacheError(
                    "aligned2 runtime full cache SHA-256 changed"
                )
        except BaseException:
            self.source.close()
            raise

    def __enter__(self) -> "Qwen38Aligned2RuntimeImage":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.source.close()
            self._mapping = None
            self._closed = True

    def _require_open(self) -> np.memmap:
        if self._closed or self._mapping is None:
            raise RuntimeError("Qwen3.8 aligned2 runtime image is closed")
        return self._mapping

    def _verify_tensor(self, name: str) -> dict:
        mapping = self._require_open()
        try:
            record = self.records[name]
        except KeyError as error:
            raise KeyError(f"unknown Qwen3.8 aligned2 tensor {name!r}") from error
        if name not in self._verified_tensors:
            start = record["offset"]
            stop = start + record["bytes"]
            digest = hashlib.sha256(mapping[start:stop]).hexdigest()
            if digest != record["sha256"]:
                raise Qwen38Aligned2CacheError(
                    f"aligned2 runtime tensor {name} SHA-256 mismatch"
                )
            self._verified_tensors.add(name)
        return record

    def aligned_tensor(self, name: str) -> Qwen38CQ3AlignedTensor:
        """Authenticate and map aligned weights plus canonical row scales."""
        mapping = self._require_open()
        record = self._verify_tensor(name)
        scales = self.source.scale_tensor(name)
        shape = tuple(int(value) for value in record["shape"])
        if (
            scales.shape != shape
            or scales.rows != record["rows"]
            or scales.columns != record["columns"]
            or record["bytes"] != scales.rows * scales.columns // 4
        ):
            raise Qwen38Aligned2CacheError(
                f"aligned2 runtime tensor {name} dimensions changed"
            )
        start = record["offset"]
        weights = mapping[start : start + record["bytes"]].reshape(
            scales.rows, scales.columns // 4
        )
        weights.flags.writeable = False
        return Qwen38CQ3AlignedTensor(
            name=name,
            shape=shape,
            rows=scales.rows,
            columns=scales.columns,
            packed_weights=weights,
            packed_scale_indices=scales.packed_scale_indices,
            row_anchors=scales.row_anchors,
            ratio_codebook=scales.ratio_codebook,
        )

    def mlp(self, layer: int) -> Qwen38CQ3MLP:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be non-negative")
        prefix = f"model.language_model.layers.{layer}.mlp"
        return Qwen38CQ3MLP(
            gate=self.aligned_tensor(f"{prefix}.gate_proj.weight"),
            up=self.aligned_tensor(f"{prefix}.up_proj.weight"),
            down=self.aligned_tensor(f"{prefix}.down_proj.weight"),
        )

    def deltanet(self, layer: int) -> Qwen38CQ3DeltaNet:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("Qwen3.8-27B layer index must be non-negative")
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        conv = self.exact_tensor(f"{prefix}.conv1d.weight").as_f32()
        if conv.shape != (Qwen38CQ3DeltaNet.CONV_DIM, 1, 4):
            raise CQ3RuntimeError("Qwen3.8 DeltaNet conv1d shape changed")
        return Qwen38CQ3DeltaNet(
            qkv=self.aligned_tensor(f"{prefix}.in_proj_qkv.weight"),
            z=self.aligned_tensor(f"{prefix}.in_proj_z.weight"),
            a=self.aligned_tensor(f"{prefix}.in_proj_a.weight"),
            b=self.aligned_tensor(f"{prefix}.in_proj_b.weight"),
            out=self.aligned_tensor(f"{prefix}.out_proj.weight"),
            conv_weights=np.ascontiguousarray(conv[:, 0, :].T),
            a_log=np.ascontiguousarray(
                self.exact_tensor(f"{prefix}.A_log").as_f32()
            ),
            dt_bias=np.ascontiguousarray(
                self.exact_tensor(f"{prefix}.dt_bias").as_f32()
            ),
            norm_weights=np.ascontiguousarray(
                self.exact_tensor(f"{prefix}.norm.weight").as_f32()
            ),
        )

    def exact_tensor(self, name: str):
        return self.source.exact_tensor(name)

    def auxiliary_file(self, name: str) -> np.ndarray:
        return self.source.auxiliary_file(name)

    def verify_runtime_inputs(self) -> dict:
        """Verify cache weights and every canonical non-base3 runtime section."""
        for name in self.records:
            self._verify_tensor(name)
            self.source.scale_tensor(name)
        exact = 0
        for tensor in self.source.tensors.values():
            encoding = tensor.get("encoding")
            if isinstance(encoding, str) and encoding.startswith(
                "retained_exact_"
            ):
                self.source.exact_tensor(tensor["name"])
                exact += 1
        for name in self.source.auxiliary_files:
            self.source.auxiliary_file(name)
        return {
            "schema": "qwen38_27b_aligned2_runtime_verify_receipt_v1",
            "aligned_weight_tensors_verified": len(self._verified_tensors),
            "row_scale_tensors_verified": len(self.records),
            "retained_exact_tensors_verified": exact,
            "auxiliary_files_verified": len(self.source.auxiliary_files),
            "canonical_block203_weight_sections_read": False,
        }
