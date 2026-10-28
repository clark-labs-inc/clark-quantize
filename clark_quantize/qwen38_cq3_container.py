"""Atomic canonical container for Qwen3.8-27B ternary research artifacts.

The container is an offline/storage boundary.  ISA-specific GGUF compute
images are derived artifacts and are intentionally not embedded here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO, Iterable, Iterator, Mapping

from .base3_chunks import canonical_json


MAGIC = b"CQ3W27B1"
VERSION_MAJOR = 1
VERSION_MINOR = 0
HEADER_BYTES = 128
FOOTER_BYTES = 32
PAYLOAD_OFFSET = 4096
SECTION_ALIGNMENT = 64
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
MODEL_ID = "Qwen/Qwen3.8-27B"
DIRECTORY_SCHEMA = "clark_qwen38_cq3_directory_v1"
HEADER = struct.Struct("<8sHHIQQQQQQ32s32s")

if HEADER.size != HEADER_BYTES:
    raise AssertionError("Qwen3.8 CQ3 header must remain exactly 128 bytes")


class CQ3ContainerError(ValueError):
    """The container is malformed, non-canonical, or violates its ledger."""


def align_up(value: int, alignment: int = SECTION_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("value must be non-negative and alignment positive")
    return ((value + alignment - 1) // alignment) * alignment


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CQ3ContainerError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _zero_bytes(handle: BinaryIO, count: int) -> None:
    block = bytes(min(COPY_CHUNK_BYTES, count))
    remaining = count
    while remaining:
        take = min(len(block), remaining)
        handle.write(block[:take])
        remaining -= take


def _source_chunks(source: bytes | bytearray | memoryview | Path) -> Iterator[bytes]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        view = memoryview(source)
        for start in range(0, len(view), COPY_CHUNK_BYTES):
            yield bytes(view[start : start + COPY_CHUNK_BYTES])
        return
    path = Path(source)
    with path.open("rb") as handle:
        while True:
            content = handle.read(COPY_CHUNK_BYTES)
            if not content:
                return
            yield content


def _validate_tensor_inventory(
    tensors: list[dict], section_ids: set[str], parameter_count: int
) -> None:
    names: set[str] = set()
    observed_parameters = 0
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            raise CQ3ContainerError(f"tensor record {index} is not an object")
        name = tensor.get("name")
        shape = tensor.get("shape")
        sections = tensor.get("sections")
        if not isinstance(name, str) or not name or name in names:
            raise CQ3ContainerError(f"tensor record {index} has an invalid or duplicate name")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
        ):
            raise CQ3ContainerError(f"tensor {name} has an invalid shape")
        if (
            not isinstance(sections, list)
            or not sections
            or any(section not in section_ids for section in sections)
            or len(set(sections)) != len(sections)
        ):
            raise CQ3ContainerError(f"tensor {name} has invalid section references")
        parameters = 1
        for dimension in shape:
            parameters *= dimension
        if tensor.get("parameters") != parameters:
            raise CQ3ContainerError(f"tensor {name} parameter ledger does not match its shape")
        names.add(name)
        observed_parameters += parameters
    if observed_parameters != parameter_count:
        raise CQ3ContainerError(
            f"tensor inventory accounts {observed_parameters} parameters; "
            f"header requires {parameter_count}"
        )


def _validate_complete_section_references(
    tensors: list[dict], auxiliary_files: list[dict], sections: list[dict]
) -> None:
    by_id = {record["id"]: record for record in sections}
    references = [
        section_id
        for tensor in tensors
        for section_id in tensor["sections"]
    ]
    auxiliary_names: set[str] = set()
    for index, auxiliary in enumerate(auxiliary_files):
        if not isinstance(auxiliary, dict):
            raise CQ3ContainerError(f"auxiliary record {index} is not an object")
        name = auxiliary.get("name")
        section_id = auxiliary.get("section")
        if (
            not isinstance(name, str)
            or not name
            or name in auxiliary_names
            or section_id not in by_id
        ):
            raise CQ3ContainerError(f"auxiliary record {index} is invalid")
        section = by_id[section_id]
        if (
            auxiliary.get("bytes") != section["bytes"]
            or auxiliary.get("sha256") != section["sha256"]
        ):
            raise CQ3ContainerError(f"auxiliary file {name} differs from its section")
        auxiliary_names.add(name)
        references.append(section_id)
    if len(references) != len(set(references)) or set(references) != set(by_id):
        raise CQ3ContainerError(
            "every payload section must be referenced exactly once by a tensor or auxiliary file"
        )


class Qwen38CQ3Writer:
    """Stream sections into a temporary file and atomically publish on finalize."""

    def __init__(
        self,
        path: Path,
        *,
        model: Mapping[str, object],
        strategy: Mapping[str, object],
        parameter_count: int,
    ) -> None:
        self.path = Path(path)
        if model.get("id") != MODEL_ID or not isinstance(model.get("revision"), str):
            raise ValueError("the container requires the pinned Qwen3.8-27B identity")
        if not strategy.get("name"):
            raise ValueError("a named storage strategy is required")
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        self.model = dict(model)
        self.strategy = dict(strategy)
        self.parameter_count = int(parameter_count)
        self.sections: list[dict] = []
        self._section_ids: set[str] = set()
        self._finalized = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", dir=self.path.parent
        )
        self._temporary = Path(temporary_name)
        self._handle = os.fdopen(descriptor, "w+b")
        _zero_bytes(self._handle, PAYLOAD_OFFSET)

    def __enter__(self) -> "Qwen38CQ3Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._finalized:
            self.abort()

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self._temporary.unlink(missing_ok=True)

    def add_section(
        self,
        section_id: str,
        kind: str,
        source: bytes | bytearray | memoryview | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict:
        return self.add_section_parts(
            section_id, kind, (source,), metadata=metadata
        )

    def add_section_parts(
        self,
        section_id: str,
        kind: str,
        sources: Iterable[bytes | bytearray | memoryview | Path],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict:
        """Stream several sources into one authenticated aligned section."""
        if self._finalized or self._handle.closed:
            raise RuntimeError("container writer is closed")
        if not section_id or section_id in self._section_ids:
            raise ValueError("section IDs must be nonempty and unique")
        if not kind:
            raise ValueError("section kind must be nonempty")
        metadata_value = dict(metadata or {})
        canonical_json(metadata_value)
        offset = align_up(self._handle.tell())
        _zero_bytes(self._handle, offset - self._handle.tell())
        digest = hashlib.sha256()
        size = 0
        source_count = 0
        for source in sources:
            source_count += 1
            for content in _source_chunks(source):
                self._handle.write(content)
                digest.update(content)
                size += len(content)
        if source_count == 0:
            raise ValueError("a container section requires at least one source")
        record = {
            "id": section_id,
            "kind": kind,
            "offset": offset,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "metadata": metadata_value,
        }
        self.sections.append(record)
        self._section_ids.add(section_id)
        return dict(record)

    def finalize(
        self,
        *,
        tensors: Iterable[Mapping[str, object]],
        auxiliary_files: Iterable[Mapping[str, object]] = (),
    ) -> dict:
        if self._finalized or self._handle.closed:
            raise RuntimeError("container writer is closed")
        tensor_records = [dict(record) for record in tensors]
        auxiliary_records = [dict(record) for record in auxiliary_files]
        _validate_tensor_inventory(
            tensor_records, self._section_ids, self.parameter_count
        )
        canonical_json(auxiliary_records)
        _validate_complete_section_references(
            tensor_records, auxiliary_records, self.sections
        )
        directory_offset = align_up(self._handle.tell())
        _zero_bytes(self._handle, directory_offset - self._handle.tell())
        section_payload_bytes = sum(record["bytes"] for record in self.sections)
        directory = {
            "schema": DIRECTORY_SCHEMA,
            "model": self.model,
            "strategy": self.strategy,
            "parameter_count": self.parameter_count,
            "section_alignment": SECTION_ALIGNMENT,
            "sections": self.sections,
            "tensors": tensor_records,
            "auxiliary_files": auxiliary_records,
            "ledger": {
                "section_payload_bytes": section_payload_bytes,
                "payload_envelope_bytes": directory_offset - PAYLOAD_OFFSET,
                "alignment_padding_bytes": (
                    directory_offset - PAYLOAD_OFFSET - section_payload_bytes
                ),
            },
        }
        directory_content = canonical_json(directory)
        if len(directory_content) > MAX_DIRECTORY_BYTES:
            raise CQ3ContainerError("container directory exceeds the frozen size limit")
        self._handle.write(directory_content)
        directory_sha256 = hashlib.sha256(directory_content).digest()
        header_content = HEADER.pack(
            MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            HEADER_BYTES,
            PAYLOAD_OFFSET,
            directory_offset - PAYLOAD_OFFSET,
            directory_offset,
            len(directory_content),
            len(self.sections),
            self.parameter_count,
            directory_sha256,
            bytes(32),
        )
        self._handle.seek(0)
        self._handle.write(header_content)
        self._handle.seek(directory_offset + len(directory_content))
        root_sha256 = hashlib.sha256(header_content + directory_content).digest()
        self._handle.write(root_sha256)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        file_bytes = self._handle.tell()
        self._handle.close()
        os.replace(self._temporary, self.path)
        _fsync_directory(self.path.parent)
        self._finalized = True
        return {
            "schema": "clark_qwen38_cq3_write_receipt_v1",
            "path": str(self.path),
            "bytes": file_bytes,
            "directory_offset": directory_offset,
            "directory_bytes": len(directory_content),
            "directory_sha256": directory_sha256.hex(),
            "root_sha256": root_sha256.hex(),
            "sections": len(self.sections),
            "tensors": len(tensor_records),
            "parameters": self.parameter_count,
            "ledger": directory["ledger"],
        }


class Qwen38CQ3Reader:
    """Fail-closed reader for the canonical Qwen3.8-27B container."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = self.path.open("rb")
        self.file_bytes = self.path.stat().st_size
        header_content = self._handle.read(HEADER_BYTES)
        if len(header_content) != HEADER_BYTES:
            self.close()
            raise CQ3ContainerError("truncated CQ3 header")
        (
            magic,
            major,
            minor,
            header_bytes,
            payload_offset,
            payload_bytes,
            directory_offset,
            directory_bytes,
            section_count,
            parameter_count,
            directory_sha256,
            reserved,
        ) = HEADER.unpack(header_content)
        if (
            magic != MAGIC
            or (major, minor) != (VERSION_MAJOR, VERSION_MINOR)
            or header_bytes != HEADER_BYTES
            or payload_offset != PAYLOAD_OFFSET
            or payload_bytes != directory_offset - payload_offset
            or directory_offset % SECTION_ALIGNMENT
            or directory_bytes > MAX_DIRECTORY_BYTES
            or directory_offset + directory_bytes + FOOTER_BYTES != self.file_bytes
            or reserved != bytes(32)
        ):
            self.close()
            raise CQ3ContainerError("invalid CQ3 header contract")
        self._handle.seek(directory_offset)
        directory_content = self._handle.read(directory_bytes)
        footer = self._handle.read(FOOTER_BYTES)
        if (
            len(directory_content) != directory_bytes
            or hashlib.sha256(directory_content).digest() != directory_sha256
            or footer != hashlib.sha256(header_content + directory_content).digest()
        ):
            self.close()
            raise CQ3ContainerError("CQ3 directory or root checksum mismatch")
        try:
            directory = json.loads(directory_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.close()
            raise CQ3ContainerError("CQ3 directory is not valid JSON") from error
        if canonical_json(directory) != directory_content:
            self.close()
            raise CQ3ContainerError("CQ3 directory is not canonical JSON")
        if (
            directory.get("schema") != DIRECTORY_SCHEMA
            or directory.get("model", {}).get("id") != MODEL_ID
            or directory.get("parameter_count") != parameter_count
            or directory.get("section_alignment") != SECTION_ALIGNMENT
        ):
            self.close()
            raise CQ3ContainerError("CQ3 directory identity differs from its header")
        sections = directory.get("sections")
        if not isinstance(sections, list) or len(sections) != section_count:
            self.close()
            raise CQ3ContainerError("CQ3 section count differs from its header")
        self.header = {
            "payload_offset": payload_offset,
            "payload_bytes": payload_bytes,
            "directory_offset": directory_offset,
            "directory_bytes": directory_bytes,
            "section_count": section_count,
            "parameter_count": parameter_count,
            "directory_sha256": directory_sha256.hex(),
            "root_sha256": footer.hex(),
        }
        self.directory = directory
        self.sections: dict[str, dict] = {}
        self._verify_zero_range(HEADER_BYTES, payload_offset)
        previous_end = payload_offset
        for index, record in enumerate(sections):
            if not isinstance(record, dict):
                self.close()
                raise CQ3ContainerError(f"section record {index} is not an object")
            section_id = record.get("id")
            offset = record.get("offset")
            size = record.get("bytes")
            if (
                not isinstance(section_id, str)
                or not section_id
                or section_id in self.sections
                or not isinstance(record.get("kind"), str)
                or not record["kind"]
                or not isinstance(offset, int)
                or offset % SECTION_ALIGNMENT
                or not isinstance(size, int)
                or size < 0
                or offset < previous_end
                or offset + size > directory_offset
            ):
                self.close()
                raise CQ3ContainerError(f"section record {index} violates its bounds")
            _validate_sha256(record.get("sha256"), f"section {section_id}")
            self._verify_zero_range(previous_end, offset)
            self.sections[section_id] = record
            previous_end = offset + size
        self._verify_zero_range(previous_end, directory_offset)
        tensor_records = directory.get("tensors")
        if not isinstance(tensor_records, list):
            self.close()
            raise CQ3ContainerError("CQ3 tensor inventory is absent")
        _validate_tensor_inventory(
            tensor_records, set(self.sections), parameter_count
        )
        auxiliary_records = directory.get("auxiliary_files")
        if not isinstance(auxiliary_records, list):
            self.close()
            raise CQ3ContainerError("CQ3 auxiliary-file inventory is absent")
        _validate_complete_section_references(
            tensor_records, auxiliary_records, sections
        )
        ledger = directory.get("ledger", {})
        section_payload_bytes = sum(record["bytes"] for record in sections)
        expected_ledger = {
            "section_payload_bytes": section_payload_bytes,
            "payload_envelope_bytes": directory_offset - payload_offset,
            "alignment_padding_bytes": (
                directory_offset - payload_offset - section_payload_bytes
            ),
        }
        if ledger != expected_ledger:
            self.close()
            raise CQ3ContainerError("CQ3 byte ledger is inconsistent")

    def __enter__(self) -> "Qwen38CQ3Reader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _verify_zero_range(self, start: int, stop: int) -> None:
        self._handle.seek(start)
        remaining = stop - start
        while remaining:
            content = self._handle.read(min(COPY_CHUNK_BYTES, remaining))
            if len(content) == 0 or any(content):
                raise CQ3ContainerError("CQ3 alignment padding is non-canonical")
            remaining -= len(content)

    def iter_section(self, section_id: str) -> Iterator[bytes]:
        try:
            record = self.sections[section_id]
        except KeyError as error:
            raise KeyError(f"unknown CQ3 section {section_id!r}") from error
        self._handle.seek(record["offset"])
        remaining = record["bytes"]
        while remaining:
            content = self._handle.read(min(COPY_CHUNK_BYTES, remaining))
            if not content:
                raise CQ3ContainerError(f"section {section_id} is truncated")
            remaining -= len(content)
            yield content

    def read_section(self, section_id: str) -> bytes:
        return b"".join(self.iter_section(section_id))

    def verify_section(self, section_id: str) -> bool:
        digest = hashlib.sha256()
        for content in self.iter_section(section_id):
            digest.update(content)
        expected = self.sections[section_id]["sha256"]
        if digest.hexdigest() != expected:
            raise CQ3ContainerError(f"section {section_id} SHA-256 mismatch")
        return True

    def verify_all(self) -> dict:
        for section_id in self.sections:
            self.verify_section(section_id)
        return {
            "schema": "clark_qwen38_cq3_verify_receipt_v1",
            "path": str(self.path),
            "bytes": self.file_bytes,
            "root_sha256": self.header["root_sha256"],
            "sections_verified": len(self.sections),
            "parameters": self.header["parameter_count"],
        }
