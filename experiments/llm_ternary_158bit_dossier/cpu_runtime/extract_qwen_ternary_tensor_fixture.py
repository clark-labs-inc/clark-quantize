#!/usr/bin/env python3
"""Extract one integrity-checked Qwen ternary matrix for cross-ISA benchmarking."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor_codes(
    reader: QwenBase3ArtifactReader,
    name: str,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if name not in reader.records_by_tensor:
        raise KeyError(name)
    code_batches: list[np.ndarray] = []
    scale_batches: list[np.ndarray] = []
    record_indices: list[int] = []
    loaded_rows = 0
    for record in reader.records_by_tensor[name]:
        codes, scales = reader.raw_batch(record)
        take = min(rows - loaded_rows, codes.shape[0])
        code_batches.append(codes[:take])
        scale_batches.append(scales[:take])
        record_indices.append(int(record["index"]))
        loaded_rows += take
        if loaded_rows == rows:
            break
    codes = np.ascontiguousarray(np.concatenate(code_batches), dtype=np.int8)
    scales = np.ascontiguousarray(np.concatenate(scale_batches), dtype=np.float16)
    expected_scales = math.ceil(columns / reader.group_size)
    if codes.shape != (rows, columns) or scales.shape != (rows, expected_scales):
        raise ValueError("artifact tensor shape differs from requested fixture")
    return codes, scales, record_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--columns", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0:
        parser.error("--rows and --columns must be positive")
    if not args.out.is_absolute() or not args.receipt.is_absolute():
        parser.error("--out and --receipt must be absolute")

    parent = json.loads(args.pack_receipt.read_text())
    if (
        parent.get("promotion", {}).get("passed") is not True
        or parent.get("identity", {}).get("model_revision") != args.model_revision
    ):
        raise ValueError("fixture parent is not the promoted exact pack receipt")
    reader = QwenBase3ArtifactReader(args.artifact)
    identity = reader.manifest["identity"]
    if (
        identity.get("model_revision") != args.model_revision
        or reader.group_size != 128
    ):
        raise ValueError("artifact model revision or group contract differs")

    codes, scales, record_indices = load_tensor_codes(
        reader, args.tensor, args.rows, args.columns
    )
    codes_sha256 = hashlib.sha256(codes.tobytes()).hexdigest()
    scales_sha256 = hashlib.sha256(scales.tobytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.out.name}.", dir=args.out.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, codes=codes, scales=scales)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as stored:
            loaded_codes = np.ascontiguousarray(stored["codes"], dtype=np.int8)
            loaded_scales = np.ascontiguousarray(stored["scales"], dtype=np.float16)
        if (
            loaded_codes.shape != codes.shape
            or loaded_scales.shape != scales.shape
            or hashlib.sha256(loaded_codes.tobytes()).hexdigest() != codes_sha256
            or hashlib.sha256(loaded_scales.tobytes()).hexdigest() != scales_sha256
        ):
            raise RuntimeError("fixture verification after serialization failed")
        os.replace(temporary, args.out)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    receipt = {
        "schema": "clark_qwen_ternary_tensor_fixture_v1",
        "model_revision": args.model_revision,
        "artifact": {
            "root": str(args.artifact.resolve()),
            "manifest_sha256": file_sha256(reader.manifest_path),
            "pack_receipt": str(args.pack_receipt.resolve()),
            "pack_receipt_sha256": file_sha256(args.pack_receipt),
        },
        "tensor": {
            "name": args.tensor,
            "shape": [args.rows, args.columns],
            "group_size": reader.group_size,
            "artifact_record_indices": record_indices,
            "codes_dtype": "int8",
            "codes_sha256": codes_sha256,
            "scales_dtype": "float16",
            "scales_sha256": scales_sha256,
        },
        "fixture": {
            "path": str(args.out.resolve()),
            "bytes": args.out.stat().st_size,
            "sha256": file_sha256(args.out),
        },
        "extractor_sha256": file_sha256(Path(__file__)),
    }
    atomic_bytes(
        args.receipt,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
