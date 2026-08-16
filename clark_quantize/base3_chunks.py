"""Spot-safe, independently decodable chunk storage for exact ternary codes.

Each chunk is a complete :mod:`clark_quantize.base3` frame.  A durable partial
manifest is replaced atomically after every committed chunk, so an interrupted
writer can validate and reuse the prefix already on disk.  Resume is strict:
both the experiment identity and regenerated raw trit hashes must match.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np

from .base3 import frame_metadata, pack_base3, unpack_base3


SCHEMA = "clark_chunked_base3_v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, content: bytes) -> None:
    """Write, fsync, and atomically replace ``path`` in its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def encode_chunk(index: int, codes: np.ndarray, descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    """CPU-worker-friendly function returning a verified frame and ledger."""
    array = np.asarray(codes, dtype=np.int8).reshape(-1)
    raw = array.tobytes()
    frame = pack_base3(array)
    decoded = unpack_base3(frame)
    verified_roundtrip = bool(np.array_equal(decoded, array))
    if not verified_roundtrip:
        raise AssertionError(f"chunk {index} arithmetic round-trip changed source trits")
    return {
        "index": index,
        "descriptor": descriptor or {},
        "trits": int(array.size),
        "raw_sha256": sha256_bytes(raw),
        "frame_sha256": sha256_bytes(frame),
        "frame": frame,
        "frame_metadata": frame_metadata(frame),
        "verified_roundtrip": verified_roundtrip,
    }


def verify_chunk_file(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Independently hash and decode one committed chunk in a worker process."""
    frame = Path(path).read_bytes()
    frame_sha = sha256_bytes(frame)
    if frame_sha != record["frame_sha256"]:
        raise ValueError(f"chunk {record['index']} frame SHA-256 mismatch")
    decoded = unpack_base3(frame)
    raw_sha = sha256_bytes(decoded.tobytes())
    if decoded.size != record["trits"] or raw_sha != record["raw_sha256"]:
        raise ValueError(f"chunk {record['index']} independent decode mismatch")
    return {"index": record["index"], "trits": int(decoded.size), "raw_sha256": raw_sha}


class ChunkedBase3Store:
    """Append-only chunk directory with strict, deterministic resume."""

    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.chunk_dir = self.root / "chunks"
        self.partial_path = self.root / "manifest.partial.json"
        self.final_path = self.root / "manifest.json"
        self.identity = identity
        self.identity_sha256 = sha256_bytes(canonical_json(identity))
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            manifest = json.loads(self.final_path.read_text())
            self._validate_manifest_identity(manifest)
            self.manifest = manifest
            self.finalized = True
        elif self.partial_path.exists():
            manifest = json.loads(self.partial_path.read_text())
            self._validate_manifest_identity(manifest)
            self.manifest = manifest
            self.finalized = False
        else:
            self.manifest = {
                "schema": SCHEMA,
                "identity": identity,
                "identity_sha256": self.identity_sha256,
                "chunks": [],
                "complete": False,
            }
            self.finalized = False
            self._persist_partial()
        self._validate_contiguous_records()
        self._upgrade_verified_roundtrip_receipts()

    def _validate_manifest_identity(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema") != SCHEMA:
            raise ValueError("chunk manifest schema mismatch")
        if manifest.get("identity_sha256") != self.identity_sha256 or manifest.get("identity") != self.identity:
            raise ValueError("chunk manifest experiment identity mismatch")

    def _validate_contiguous_records(self) -> None:
        for expected, record in enumerate(self.manifest["chunks"]):
            if record.get("index") != expected:
                raise ValueError("chunk manifest is not a contiguous prefix")

    def _upgrade_verified_roundtrip_receipts(self) -> None:
        """Seal manifests written by the initial serial-verifying implementation."""
        changed = False
        for record in self.manifest["chunks"]:
            if "verified_roundtrip" not in record:
                self._validate_frame(record)
                record["verified_roundtrip"] = True
                changed = True
        if changed:
            target = self.final_path if self.finalized else self.partial_path
            atomic_bytes(target, canonical_json(self.manifest))

    @property
    def next_index(self) -> int:
        return len(self.manifest["chunks"])

    def chunk_path(self, index: int) -> Path:
        return self.chunk_dir / f"{index:08d}.cq3a"

    def _persist_partial(self) -> None:
        atomic_bytes(self.partial_path, canonical_json(self.manifest))

    def _validate_frame(self, record: dict[str, Any], expected_raw_sha256: str | None = None) -> np.ndarray:
        path = self.chunk_path(record["index"])
        frame = path.read_bytes()
        if sha256_bytes(frame) != record["frame_sha256"]:
            raise ValueError(f"chunk {record['index']} frame SHA-256 mismatch")
        decoded = unpack_base3(frame)
        raw_sha = sha256_bytes(decoded.tobytes())
        if raw_sha != record["raw_sha256"]:
            raise ValueError(f"chunk {record['index']} raw SHA-256 mismatch")
        if expected_raw_sha256 is not None and raw_sha != expected_raw_sha256:
            raise ValueError(f"chunk {record['index']} regenerated source mismatch")
        return decoded

    def commit_encoded(self, encoded: dict[str, Any]) -> str:
        """Commit the next encoded chunk, or validate an already committed one.

        Returns ``"written"`` for a new chunk and ``"reused"`` for a verified
        resume hit.
        """
        index = int(encoded["index"])
        record = {
            key: encoded[key]
            for key in (
                "index", "descriptor", "trits", "raw_sha256", "frame_sha256", "frame_metadata",
                "verified_roundtrip",
            )
        }
        if index < self.next_index:
            existing = self.manifest["chunks"][index]
            if existing != record:
                raise ValueError(f"chunk {index} resume metadata mismatch")
            self._validate_frame(existing, expected_raw_sha256=encoded["raw_sha256"])
            return "reused"
        if self.finalized:
            raise ValueError("cannot append to a finalized chunk store")
        if index != self.next_index:
            raise ValueError(f"chunk {index} is out of order; expected {self.next_index}")
        if encoded.get("verified_roundtrip") is not True:
            raise ValueError(f"chunk {index} was not round-trip verified by its encoder")
        frame = encoded["frame"]
        if sha256_bytes(frame) != record["frame_sha256"]:
            raise ValueError(f"chunk {index} in-memory frame SHA-256 mismatch")
        metadata = frame_metadata(frame)
        if metadata != record["frame_metadata"] or metadata["trits"] != record["trits"]:
            raise ValueError(f"chunk {index} frame metadata mismatch")
        atomic_bytes(self.chunk_path(index), frame)
        self.manifest["chunks"].append(record)
        self._persist_partial()
        return "written"

    def reuse_if_committed(
        self, index: int, codes: np.ndarray, descriptor: dict[str, Any] | None = None
    ) -> bool:
        """Validate regenerated codes against an existing prefix without encoding.

        This keeps restart cost at source regeneration plus hashing; the much
        slower arithmetic encoder runs only for chunks not yet committed.
        """
        if index >= self.next_index:
            return False
        array = np.asarray(codes, dtype=np.int8).reshape(-1)
        existing = self.manifest["chunks"][index]
        expected = {
            "descriptor": descriptor or {},
            "trits": int(array.size),
            "raw_sha256": sha256_bytes(array.tobytes()),
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"chunk {index} regenerated {key} mismatch")
        frame = self.chunk_path(index).read_bytes()
        if sha256_bytes(frame) != existing["frame_sha256"]:
            raise ValueError(f"chunk {index} frame SHA-256 mismatch")
        if existing.get("verified_roundtrip") is not True:
            raise ValueError(f"chunk {index} lacks a verified round-trip receipt")
        return True

    def validate_prefix(self, full_decode: bool = True) -> dict[str, int]:
        trits = payload_bits = frame_bytes = 0
        for record in self.manifest["chunks"]:
            path = self.chunk_path(record["index"])
            frame = path.read_bytes()
            if sha256_bytes(frame) != record["frame_sha256"]:
                raise ValueError(f"chunk {record['index']} frame SHA-256 mismatch")
            if full_decode:
                self._validate_frame(record)
            trits += int(record["trits"])
            payload_bits += int(record["frame_metadata"]["payload_bits"])
            frame_bytes += len(frame)
        return {"chunks": self.next_index, "trits": trits, "payload_bits": payload_bits, "frame_bytes": frame_bytes}

    def finalize(
        self,
        expected_chunks: int,
        expected_trits: int,
        source_sha256: str,
        extra: dict[str, Any] | None = None,
        full_decode: bool = True,
    ) -> dict[str, Any]:
        if self.next_index != expected_chunks:
            raise ValueError(f"cannot finalize {self.next_index} chunks; expected {expected_chunks}")
        if not all(record.get("verified_roundtrip") is True for record in self.manifest["chunks"]):
            raise ValueError("cannot finalize chunks without encoder round-trip receipts")
        ledger = self.validate_prefix(full_decode=full_decode)
        if ledger["trits"] != expected_trits:
            raise ValueError(f"cannot finalize {ledger['trits']} trits; expected {expected_trits}")
        self.manifest["complete"] = True
        self.manifest["source_sha256"] = source_sha256
        self.manifest["ledger"] = ledger
        self.manifest["extra"] = extra or {}
        atomic_bytes(self.final_path, canonical_json(self.manifest))
        self.partial_path.unlink(missing_ok=True)
        _fsync_directory(self.root)
        self.finalized = True
        return self.manifest

    def decoded_chunks(self) -> Iterator[np.ndarray]:
        for record in self.manifest["chunks"]:
            yield self._validate_frame(record)
