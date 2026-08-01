"""Canonical artifact identity checks shared by final Swin2SR evaluators."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(candidate: dict) -> None:
    materialization = Path(candidate["materialization"])
    hard_state = Path(candidate["hard_state"])
    runtime_state = Path(candidate["runtime_state"])
    portable = Path(candidate["portable_onnx"])
    portable_ledger = Path(candidate["portable_ledger"])
    runtime_package = Path(candidate["runtime_package"])
    if file_sha256(materialization) != candidate["materialization_sha256"]:
        raise ValueError("selected materialization hash mismatch")
    if file_sha256(hard_state) != candidate["hard_state_sha256"]:
        raise ValueError("selected hard-state hash mismatch")
    if file_sha256(runtime_state) != candidate["runtime_state_sha256"]:
        raise ValueError("selected runtime-state hash mismatch")
    if file_sha256(portable) != candidate["portable_onnx_sha256"]:
        raise ValueError("selected portable ONNX hash mismatch")
    if portable.stat().st_size != int(candidate["portable_onnx_bytes"]):
        raise ValueError("selected portable ONNX size mismatch")
    if candidate["runtime_package_compression"] != "gzip":
        raise ValueError("selected runtime package is not deterministic gzip")
    if file_sha256(runtime_package) != candidate["runtime_package_sha256"]:
        raise ValueError("selected runtime package hash mismatch")
    if runtime_package.stat().st_size != int(candidate["runtime_package_bytes"]):
        raise ValueError("selected runtime package size mismatch")
    if gzip.decompress(runtime_package.read_bytes()) != portable.read_bytes():
        raise ValueError(
            "selected runtime package does not expand to portable ONNX"
        )
    if file_sha256(portable_ledger) != candidate["portable_ledger_sha256"]:
        raise ValueError("selected portable ledger hash mismatch")
    manifest = json.loads(materialization.read_text())
    if (
        manifest.get("format")
        != "backdropkit-swin2sr-surface-materialization-v1"
        or not manifest.get("all_target_surfaces_materialized_hard")
    ):
        raise ValueError("selected materialization is not an all-hard identity")
    dense_record = manifest.get("dense_state", {})
    runtime_record = manifest.get("runtime_state", {})
    if (
        Path(dense_record.get("path", "")).resolve() != hard_state.resolve()
        or dense_record.get("file_sha256") != candidate["hard_state_sha256"]
    ):
        raise ValueError("materialization dense-state identity mismatch")
    if (
        Path(runtime_record.get("path", "")).resolve() != runtime_state.resolve()
        or runtime_record.get("file_sha256")
        != candidate["runtime_state_sha256"]
    ):
        raise ValueError("materialization runtime-state identity mismatch")
    ledger = json.loads(portable_ledger.read_text())
    expected_ledger = {
        "format": "clark-portable-onnx-ternary-v2",
        "materialization_manifest_sha256": candidate[
            "materialization_sha256"
        ],
        "hard_state_sha256": candidate["hard_state_sha256"],
        "runtime_state_file_sha256": candidate["runtime_state_sha256"],
        "model_sha256": candidate["portable_onnx_sha256"],
        "runtime_package_sha256": candidate["runtime_package_sha256"],
        "runtime_package_compression": candidate[
            "runtime_package_compression"
        ],
        "scope": "all_matrix_and_conv",
    }
    if any(ledger.get(key) != value for key, value in expected_ledger.items()):
        raise ValueError(
            "portable ledger does not match canonical materialization"
        )
