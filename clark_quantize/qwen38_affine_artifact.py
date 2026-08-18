"""Authenticated packed artifacts for the Qwen3.8 affine-base3 block gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base3_chunks import atomic_bytes, canonical_json
from .qwen38_affine_butterfly import AffineTernary
from .qwen38_affine_codebook import (
    CompressedAffineTernary,
    SignedRowNormalizedCodebook,
    reconstruct_signed_row_normalized,
)
from .qwen38_row_scale_codebook import (
    RowNormalizedScaleCodebook,
    reconstruct_row_normalized_scales,
)
from .scale_codebook import pack_nibbles, unpack_nibbles
from .ternary_cpu import (
    block203_matrix_size,
    pack_block203_rows,
    unpack_block203_rows,
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_blob(root: Path, relative: Path, content: bytes) -> dict[str, Any]:
    path = root / relative
    atomic_bytes(path, content)
    return {
        "path": relative.as_posix(),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def _read_blob(root: Path, descriptor: dict[str, Any]) -> bytes:
    path = root / str(descriptor["path"])
    content = path.read_bytes()
    if (
        len(content) != int(descriptor["bytes"])
        or sha256_bytes(content) != descriptor["sha256"]
    ):
        raise ValueError(f"affine artifact blob failed authentication: {path}")
    return content


def write_affine_tensor(
    root: Path,
    name: str,
    compressed: CompressedAffineTernary,
) -> dict[str, Any]:
    """Write and immediately round-trip one actual packed tensor payload."""
    payload = compressed.payload
    rows, columns = payload.codes.shape
    key = hashlib.sha256(name.encode()).hexdigest()[:20]
    directory = Path("tensors") / key
    codes = payload.codes.detach().cpu().numpy().astype(np.int8, copy=False)
    packed_codes = pack_block203_rows(codes)
    rebuilt_codes = unpack_block203_rows(packed_codes, rows, columns)
    if not np.array_equal(rebuilt_codes, codes):
        raise AssertionError(f"base3 round-trip changed affine codes for {name}")
    expected_bits, expected_bytes = block203_matrix_size(rows, columns)
    if packed_codes.nbytes != expected_bytes:
        raise AssertionError(f"base3 byte ledger changed for {name}")

    result: dict[str, Any] = {
        "schema": "qwen38_affine_base3_tensor_v1",
        "tensor": name,
        "shape": [rows, columns],
        "group_size": payload.group_size,
        "codes": {
            **_write_blob(root, directory / "codes.block203", packed_codes.tobytes()),
            "payload_bits": expected_bits,
            "unpacked_sha256": sha256_bytes(codes.tobytes()),
        },
    }
    for kind, encoding in (("alpha", compressed.alpha), ("mu", compressed.mu)):
        packed_indices = pack_nibbles(encoding.indices)
        restored = unpack_nibbles(packed_indices, encoding.indices.size).reshape(
            encoding.indices.shape
        )
        if not np.array_equal(restored, encoding.indices):
            raise AssertionError(f"nibble round-trip changed {kind} indices for {name}")
        result[kind] = {
            "indices": _write_blob(
                root, directory / f"{kind}.indices4", packed_indices
            ),
            "anchors": _write_blob(
                root,
                directory / f"{kind}.anchors.f16",
                encoding.anchors.astype("<f2", copy=False).tobytes(),
            ),
            "codebook": _write_blob(
                root,
                directory / f"{kind}.codebook.f16",
                encoding.values.astype("<f2", copy=False).tobytes(),
            ),
            "unpacked_indices_sha256": sha256_bytes(encoding.indices.tobytes()),
        }
    result["storage_bytes"] = sum(
        result[kind][part]["bytes"]
        for kind, parts in (
            ("codes", (None,)),
            ("alpha", ("indices", "anchors", "codebook")),
            ("mu", ("indices", "anchors", "codebook")),
        )
        for part in parts
        if kind != "codes"
    ) + result["codes"]["bytes"]
    descriptor_path = directory / "descriptor.json"
    atomic_bytes(root / descriptor_path, canonical_json(result))
    result["descriptor"] = {
        "path": descriptor_path.as_posix(),
        "bytes": (root / descriptor_path).stat().st_size,
        "sha256": sha256_file(root / descriptor_path),
    }
    return result


def load_compressed_affine_tensor(
    root: Path, descriptor: dict[str, Any]
) -> CompressedAffineTernary:
    if descriptor.get("schema") != "qwen38_affine_base3_tensor_v1":
        raise ValueError("affine tensor schema changed")
    rows, columns = (int(value) for value in descriptor["shape"])
    codes_blob = _read_blob(root, descriptor["codes"])
    codes_array = unpack_block203_rows(
        np.frombuffer(codes_blob, dtype=np.uint8), rows, columns
    )
    if sha256_bytes(codes_array.tobytes()) != descriptor["codes"]["unpacked_sha256"]:
        raise ValueError("affine unpacked codes failed authentication")

    decoded: dict[str, np.ndarray] = {}
    encodings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    groups = (columns + int(descriptor["group_size"]) - 1) // int(
        descriptor["group_size"]
    )
    for kind in ("alpha", "mu"):
        section = descriptor[kind]
        packed = _read_blob(root, section["indices"])
        indices = unpack_nibbles(packed, rows * groups).reshape(rows, groups)
        if sha256_bytes(indices.tobytes()) != section["unpacked_indices_sha256"]:
            raise ValueError(f"affine {kind} indices failed authentication")
        anchors = np.frombuffer(_read_blob(root, section["anchors"]), dtype="<f2")
        codebook = np.frombuffer(_read_blob(root, section["codebook"]), dtype="<f2")
        if anchors.size != rows or codebook.size != 16:
            raise ValueError(f"affine {kind} dimensions changed")
        decoded[kind] = (
            reconstruct_row_normalized_scales(anchors, codebook, indices)
            if kind == "alpha"
            else reconstruct_signed_row_normalized(anchors, codebook, indices)
        )
        encodings[kind] = (anchors.copy(), codebook.copy(), indices.copy())

    payload = AffineTernary(
        codes=torch.from_numpy(codes_array.copy()),
        mu=torch.from_numpy(decoded["mu"].copy()),
        alpha=torch.from_numpy(decoded["alpha"].copy()),
        group_size=int(descriptor["group_size"]),
    )
    if (
        set(payload.codes.unique().tolist()) - {-1, 0, 1}
        or not bool(torch.isfinite(payload.mu).all())
        or not bool(torch.isfinite(payload.alpha).all())
        or bool((payload.alpha < 0).any())
    ):
        raise ValueError("decoded affine tensor is non-finite")
    alpha_anchors, alpha_values, alpha_indices = encodings["alpha"]
    mu_anchors, mu_values, mu_indices = encodings["mu"]
    return CompressedAffineTernary(
        payload=payload,
        alpha=RowNormalizedScaleCodebook(
            anchors=alpha_anchors,
            values=alpha_values,
            indices=alpha_indices,
            reconstructed=decoded["alpha"],
        ),
        mu=SignedRowNormalizedCodebook(
            anchors=mu_anchors,
            values=mu_values,
            indices=mu_indices,
            reconstructed=decoded["mu"],
        ),
    )


def load_affine_tensor(root: Path, descriptor: dict[str, Any]) -> AffineTernary:
    return load_compressed_affine_tensor(root, descriptor).payload


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> None:
    """Authenticate the standalone descriptor written beside its payload."""
    record = descriptor["descriptor"]
    content = _read_blob(root, record)
    stored = json.loads(content)
    expected = {key: value for key, value in descriptor.items() if key != "descriptor"}
    if stored != expected:
        raise ValueError("affine standalone descriptor diverged")
