"""Restartable, hash-chained storage for exact Qwen static ternary frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base3_chunks import atomic_bytes, canonical_json


SCHEMA = "qwen38_static_chunk_store_v1"
RECEIPT_SCHEMA = "qwen38_static_chunk_receipt_v1"
ZERO_CHAIN = "0" * 64


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _chain(previous: str, core: dict[str, Any]) -> str:
    if len(previous) != 64:
        raise ValueError("static chunk chain digest has the wrong length")
    return hashlib.sha256(bytes.fromhex(previous) + canonical_json(core)).hexdigest()


class StaticChunkStore:
    """Commit exact frames in source order with strict, durable resume checks."""

    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.chunk_dir = self.root / "chunks"
        self.receipt_dir = self.root / "receipts"
        self.progress_path = self.root / "progress.json"
        self.final_path = self.root / "manifest.json"
        self.identity = identity
        self.identity_sha256 = sha256_bytes(canonical_json(identity))
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            self.progress = json.loads(self.final_path.read_text())
            self.finalized = True
        elif self.progress_path.exists():
            self.progress = json.loads(self.progress_path.read_text())
            self.finalized = False
        else:
            self.progress = {
                "schema": SCHEMA,
                "identity": identity,
                "identity_sha256": self.identity_sha256,
                "next_index": 0,
                "chain_sha256": ZERO_CHAIN,
                "totals": {
                    "trits": 0,
                    "source_frame_bytes": 0,
                    "static_frame_bytes": 0,
                    "counts": [0, 0, 0],
                },
                "complete": False,
            }
            self.finalized = False
            self._persist_progress()
        if (
            self.progress.get("schema") != SCHEMA
            or self.progress.get("identity") != identity
            or self.progress.get("identity_sha256") != self.identity_sha256
        ):
            raise ValueError("static chunk store identity changed")

    @property
    def next_index(self) -> int:
        return int(self.progress["next_index"])

    def chunk_path(self, index: int) -> Path:
        return self.chunk_dir / f"{index:08d}.cq3s"

    def receipt_path(self, index: int) -> Path:
        return self.receipt_dir / f"{index:08d}.json"

    def _persist_progress(self) -> None:
        atomic_bytes(self.progress_path, canonical_json(self.progress))

    def _validate_receipt(self, index: int, previous: str, *, hash_frame: bool) -> tuple[dict, str]:
        receipt_path = self.receipt_path(index)
        frame_path = self.chunk_path(index)
        if not receipt_path.is_file() or not frame_path.is_file():
            raise ValueError(f"static chunk {index} is incomplete")
        receipt = json.loads(receipt_path.read_text())
        core = receipt.get("core")
        chain = receipt.get("chain_sha256")
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or not isinstance(core, dict)
            or core.get("index") != index
            or core.get("previous_chain_sha256") != previous
            or chain != _chain(previous, core)
            or frame_path.stat().st_size != core.get("static_frame_bytes")
        ):
            raise ValueError(f"static chunk {index} receipt or chain changed")
        if hash_frame and sha256_bytes(frame_path.read_bytes()) != core.get("static_frame_sha256"):
            raise ValueError(f"static chunk {index} frame digest changed")
        return receipt, chain

    def validate_prefix(self, *, hash_frames: bool = True) -> dict[str, Any]:
        previous = ZERO_CHAIN
        totals = {
            "trits": 0,
            "source_frame_bytes": 0,
            "static_frame_bytes": 0,
            "counts": [0, 0, 0],
        }
        for index in range(self.next_index):
            receipt, previous = self._validate_receipt(index, previous, hash_frame=hash_frames)
            core = receipt["core"]
            totals["trits"] += int(core["trits"])
            totals["source_frame_bytes"] += int(core["source_frame_bytes"])
            totals["static_frame_bytes"] += int(core["static_frame_bytes"])
            totals["counts"] = [
                left + int(right) for left, right in zip(totals["counts"], core["counts"])
            ]
        if previous != self.progress["chain_sha256"] or totals != self.progress["totals"]:
            raise ValueError("static chunk progress aggregate or terminal chain changed")
        return totals

    def commit(self, encoded: dict[str, Any]) -> str:
        index = int(encoded["index"])
        if index < self.next_index:
            receipt, _ = self._validate_receipt(index, ZERO_CHAIN if index == 0 else json.loads(
                self.receipt_path(index - 1).read_text()
            )["chain_sha256"], hash_frame=True)
            supplied = encoded["frame"]
            if receipt["core"]["static_frame_sha256"] != sha256_bytes(supplied):
                raise ValueError(f"static chunk {index} regenerated frame changed")
            return "reused"
        if self.finalized:
            raise ValueError("cannot append to a finalized static chunk store")
        if index != self.next_index:
            raise ValueError(f"static chunk {index} is out of order; expected {self.next_index}")
        frame = bytes(encoded["frame"])
        counts = [int(value) for value in encoded["counts"]]
        if len(counts) != 3 or sum(counts) != int(encoded["trits"]):
            raise ValueError("static chunk counts do not match its trit count")
        frame_sha256 = sha256_bytes(frame)
        if frame_sha256 != encoded["static_frame_sha256"]:
            raise ValueError("static chunk in-memory digest changed")
        previous = self.progress["chain_sha256"]
        core = {
            "index": index,
            "tensor": encoded["tensor"],
            "trits": int(encoded["trits"]),
            "counts": counts,
            "raw_sha256": encoded["raw_sha256"],
            "source_frame_bytes": int(encoded["source_frame_bytes"]),
            "source_frame_sha256": encoded["source_frame_sha256"],
            "static_frame_bytes": len(frame),
            "static_frame_sha256": frame_sha256,
            "static_payload_bits": int(encoded["static_payload_bits"]),
            "frequencies": [int(value) for value in encoded["frequencies"]],
            "encode_seconds": float(encoded["encode_seconds"]),
            "decode_seconds": float(encoded["decode_seconds"]),
            "previous_chain_sha256": previous,
        }
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "core": core,
            "chain_sha256": _chain(previous, core),
        }
        atomic_bytes(self.chunk_path(index), frame)
        atomic_bytes(self.receipt_path(index), canonical_json(receipt))
        totals = self.progress["totals"]
        totals["trits"] += core["trits"]
        totals["source_frame_bytes"] += core["source_frame_bytes"]
        totals["static_frame_bytes"] += core["static_frame_bytes"]
        totals["counts"] = [
            left + right for left, right in zip(totals["counts"], counts)
        ]
        self.progress["next_index"] = index + 1
        self.progress["chain_sha256"] = receipt["chain_sha256"]
        self._persist_progress()
        return "written"

    def finalize(self, *, expected_chunks: int, expected_trits: int) -> dict[str, Any]:
        if self.next_index != expected_chunks:
            raise ValueError(f"static chunk store has {self.next_index}; expected {expected_chunks}")
        totals = self.validate_prefix(hash_frames=True)
        if totals["trits"] != expected_trits:
            raise ValueError("static chunk store trit denominator changed")
        final = dict(self.progress)
        final["complete"] = True
        atomic_bytes(self.final_path, canonical_json(final))
        self.progress = final
        self.finalized = True
        return final
