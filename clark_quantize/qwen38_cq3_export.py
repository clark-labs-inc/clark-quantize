"""Restartable exact-code spooling for the complete Qwen3.8-27B CQ3 image."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator

import numpy as np
import torch

from .base3 import frame_metadata, unpack_base3
from .base3_block203 import matrix_payload_bits, matrix_payload_bytes
from .ternary_cpu import pack_block203_rows, unpack_block203_rows


@dataclass(frozen=True)
class PackedRowBatch:
    payload: bytes
    payload_bits: int
    rows: int
    columns: int
    source_codes_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_verified_rows(codes: np.ndarray) -> PackedRowBatch:
    values = np.asarray(codes, dtype=np.int8)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("Qwen3.8 code batches require two positive dimensions")
    rows, columns = values.shape
    packed = pack_block203_rows(values)
    rebuilt = unpack_block203_rows(packed, rows, columns)
    if not np.array_equal(rebuilt, values):
        raise AssertionError("Qwen3.8 block-203 batch round-trip changed ternary codes")
    payload = packed.tobytes()
    expected_bytes = matrix_payload_bytes(rows, columns)
    if len(payload) != expected_bytes:
        raise AssertionError("Qwen3.8 block-203 batch byte count changed")
    return PackedRowBatch(
        payload=payload,
        payload_bits=matrix_payload_bits(rows, columns),
        rows=rows,
        columns=columns,
        source_codes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )


def transcode_mapped_record(root: str | Path, record: dict) -> PackedRowBatch:
    """Authenticate and transcode one historical 27B arithmetic-code frame."""
    artifact = Path(root)
    index = int(record["index"])
    descriptor = record["descriptor"]
    frame = (artifact / "codes.cq3d" / "chunks" / f"{index:08d}.cq3a").read_bytes()
    if hashlib.sha256(frame).hexdigest() != record["frame_sha256"]:
        raise ValueError(f"mapped Qwen3.8 chunk {index} frame SHA-256 mismatch")
    if frame_metadata(frame) != record["frame_metadata"]:
        raise ValueError(f"mapped Qwen3.8 chunk {index} frame metadata mismatch")
    codes = unpack_base3(frame)
    if hashlib.sha256(codes.tobytes()).hexdigest() != record["raw_sha256"]:
        raise ValueError(f"mapped Qwen3.8 chunk {index} raw-code SHA-256 mismatch")
    rows = int(descriptor["row_stop"]) - int(descriptor["row_start"])
    columns = int(descriptor["tensor_shape"][1])
    if codes.size != rows * columns:
        raise ValueError(f"mapped Qwen3.8 chunk {index} code shape mismatch")
    return pack_verified_rows(codes.reshape(rows, columns))


def _batch_pairs(batches: Iterable[PackedRowBatch]) -> Iterator[tuple[PackedRowBatch, bool]]:
    iterator = iter(batches)
    try:
        current = next(iterator)
    except StopIteration as error:
        raise ValueError("at least one packed row batch is required") from error
    for following in iterator:
        yield current, False
        current = following
    yield current, True


def atomic_code_spool(
    path: Path,
    batches: Iterable[PackedRowBatch],
    *,
    expected_rows: int,
    expected_columns: int,
) -> dict:
    """Concatenate batch bitstreams and atomically publish one tensor payload.

    Every non-final batch must end on a byte boundary.  Runners use 128-row
    batches, so this holds for every possible per-row bit width while allowing
    only the final tensor byte to contain canonical zero padding.
    """
    if expected_rows <= 0 or expected_columns <= 0:
        raise ValueError("expected tensor dimensions must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    rows = bits = size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for batch, final in _batch_pairs(batches):
                if batch.columns != expected_columns or batch.rows <= 0:
                    raise ValueError("packed Qwen3.8 batch dimensions changed")
                if not final and batch.payload_bits % 8:
                    raise ValueError("non-final packed Qwen3.8 batch is not byte-aligned")
                expected_batch_bytes = (batch.payload_bits + 7) // 8
                if len(batch.payload) != expected_batch_bytes:
                    raise ValueError("packed Qwen3.8 batch length changed")
                handle.write(batch.payload)
                digest.update(batch.payload)
                rows += batch.rows
                bits += batch.payload_bits
                size += len(batch.payload)
            if rows != expected_rows or bits != matrix_payload_bits(rows, expected_columns):
                raise ValueError("packed Qwen3.8 tensor row or bit coverage is incomplete")
            if size != matrix_payload_bytes(rows, expected_columns):
                raise ValueError("packed Qwen3.8 tensor byte envelope is not canonical")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "columns": expected_columns,
        "trits": rows * expected_columns,
        "payload_bits": bits,
        "exact_batch_roundtrip": True,
    }


def exact_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return the checkpoint tensor's exact host-endian byte representation."""
    if sys_byteorder() != "little":
        raise RuntimeError("Qwen3.8 CQ3 exact tensors require a little-endian host")
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def sys_byteorder() -> str:
    # Isolated for an inexpensive failure-path unit test.
    import sys

    return sys.byteorder
