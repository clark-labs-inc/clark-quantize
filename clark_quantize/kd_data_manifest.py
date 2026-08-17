"""Immutable provenance manifest for Qwen ternary-KD token arms."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


KD_DATA_SCHEMA = "clark_qwen35_kd_arms_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def seal_kd_arms(
    root: Path,
    *,
    arms: Sequence[str],
    model: str,
    revision: str,
    probe_tokens: int,
    output: Path,
) -> dict:
    if sys.byteorder != "little":
        raise RuntimeError("KD arm format currently requires a little-endian host")
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir() or output.parent != root:
        raise ValueError("KD manifest must be written directly inside its absolute data root")
    if not arms or len(set(arms)) != len(arms) or probe_tokens <= 0:
        raise ValueError("KD arm names must be unique and probe_tokens must be positive")
    records = {}
    for arm in arms:
        if not arm or Path(arm).name != arm:
            raise ValueError(f"unsafe KD arm name: {arm!r}")
        path = root / f"{arm}.bin"
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError(f"missing or unsafe KD arm file: {path}")
        size = path.stat().st_size
        if size % 4:
            raise ValueError(f"KD arm byte length is not uint32-aligned: {path}")
        tokens = size // 4
        if tokens <= probe_tokens:
            raise ValueError(f"KD arm has no training tokens after its probe prefix: {path}")
        records[arm] = {
            "path": path.name,
            "bytes": size,
            "tokens": tokens,
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema": KD_DATA_SCHEMA,
        "model": model,
        "revision": revision,
        "dtype": "uint32_le",
        "probe_tokens_per_arm": probe_tokens,
        "arms": records,
    }
    _atomic_json(output, manifest)
    return manifest


def verify_kd_arms(
    manifest_path: Path,
    *,
    root: Path,
    arms: Sequence[str],
    model: str,
    revision: str,
    probe_tokens: int,
    rehash: bool = True,
) -> dict:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.parent != root or not manifest_path.is_file():
        raise ValueError("KD manifest must be a file directly inside its data root")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("arms")
    if (
        manifest.get("schema") != KD_DATA_SCHEMA
        or manifest.get("model") != model
        or manifest.get("revision") != revision
        or manifest.get("dtype") != "uint32_le"
        or manifest.get("probe_tokens_per_arm") != probe_tokens
        or not isinstance(records, dict)
        or sorted(records) != sorted(arms)
    ):
        raise ValueError("KD data manifest identity mismatch")
    for arm in arms:
        record = records[arm]
        path = root / f"{arm}.bin"
        if (
            record.get("path") != path.name
            or path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != root
            or path.stat().st_size != record.get("bytes")
            or record.get("tokens") != record.get("bytes") // 4
        ):
            raise ValueError(f"KD arm identity mismatch: {arm}")
        if rehash and file_sha256(path) != record.get("sha256"):
            raise ValueError(f"KD arm SHA-256 mismatch: {arm}")
    return manifest
