#!/usr/bin/env python3
"""Pack passed v45 block residuals into one mmap-ready authenticated image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, BinaryIO, Mapping, Optional


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRAINING_SCHEMA = "qwen38_27b_all_layer_mixed_training_receipt_v44"
RESIDUAL_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_receipt_v45"
LAYER_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_layer_v45"
PAYLOAD_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_payload_v45"
RECEIPT_SCHEMA = "qwen38_27b_residual_runtime_bundle_v56"
MAGIC = b"Q38R16I1"
VERSION = 1
HEADER_BYTES = 4096
TABLE_OFFSET = HEADER_BYTES
TABLE_ENTRY_BYTES = 64
LAYER_COUNT = 64
INPUT_FEATURES = 5120
OUTPUT_FEATURES = 5120
RANK = 16
PAYLOAD_BYTES = 174_144
TABLE_BYTES = LAYER_COUNT * TABLE_ENTRY_BYTES
PAYLOAD_OFFSET = TABLE_OFFSET + TABLE_BYTES
ROOT_OFFSET = 188
ROOT_BYTES = 32
PAYLOAD_FORMAT = (
    "left_INT8_per_column_scale_FP16_then_right_INT8_per_row_"
    "scale_FP16_then_bias_FP16"
)
TABLE_STRUCT = struct.Struct("<IIQQ32sQ")


@dataclass(frozen=True)
class LayerPayload:
    layer: int
    active: bool
    path: Optional[Path]
    sha256: Optional[str]


@dataclass(frozen=True)
class ImageSummary:
    bytes: int
    root_sha256: str
    active_layers: list[int]
    bypass_layers: list[int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_segment(stream: BinaryIO, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = size
    while remaining:
        block = stream.read(min(1024 * 1024, remaining))
        if not block:
            raise EOFError("unexpected EOF while hashing residual payload")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def sha256_with_zeroed_root(path: Path) -> str:
    digest = hashlib.sha256()
    position = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            mutable = bytearray(block)
            start = max(ROOT_OFFSET - position, 0)
            stop = min(ROOT_OFFSET + ROOT_BYTES - position, len(mutable))
            if start < stop:
                mutable[start:stop] = b"\0" * (stop - start)
            digest.update(mutable)
            position += len(block)
    return digest.hexdigest()


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def safe_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("v45 artifact reference must be relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("v45 artifact reference escapes its root") from error
    return resolved


def collect_payloads(
    training_receipt_path: Path,
    residual_receipt_path: Path,
    artifact_root: Path,
) -> tuple[list[LayerPayload], dict[str, Any], dict[str, Any]]:
    training_sha = sha256_file(training_receipt_path)
    training = json.loads(training_receipt_path.read_text())
    residual = json.loads(residual_receipt_path.read_text())
    layers = residual.get("layers", [])
    gate = residual.get("gate", {})
    if (
        training.get("schema") != TRAINING_SCHEMA
        or training.get("model") != MODEL
        or training.get("revision") != REVISION
        or training.get("gate", {}).get("passed") is not True
        or training.get("complete") is not True
        or training.get("terminal") is not True
        or residual.get("schema") != RESIDUAL_SCHEMA
        or residual.get("model") != MODEL
        or residual.get("revision") != REVISION
        or residual.get("training_receipt_sha256") != training_sha
        or gate.get("passed") is not True
        or residual.get("complete") is not True
        or residual.get("terminal") is not True
        or len(layers) != LAYER_COUNT
        or [item.get("layer") for item in layers] != list(range(LAYER_COUNT))
    ):
        raise ValueError("v44/v45 parent receipt identity or denominator changed")

    records: list[LayerPayload] = []
    active_count = 0
    total_payload_bytes = 0
    for layer, record in enumerate(layers):
        if (
            record.get("schema") != LAYER_SCHEMA
            or record.get("model") != MODEL
            or record.get("revision") != REVISION
            or record.get("repository_commit") != residual.get("repository_commit")
            or record.get("preregistration_sha256")
            != residual.get("preregistration_sha256")
            or record.get("training_receipt_sha256") != training_sha
            or record.get("complete") is not True
            or record.get("terminal") is not True
        ):
            raise ValueError(f"v45 layer {layer} identity changed")
        mode = record.get("mode")
        if mode == "exact_bypass":
            if (
                record.get("descriptor") is not None
                or record.get("payload") is not None
                or record.get("persistent_payload_bytes") != 0
                or record.get("bypass_tensor_exact") is not True
            ):
                raise ValueError(f"v45 bypass layer {layer} has payload state")
            records.append(LayerPayload(layer, False, None, None))
            continue
        if mode != "rank16_int8_residual":
            raise ValueError(f"v45 layer {layer} mode changed")
        descriptor_path = safe_child(artifact_root, str(record.get("descriptor", "")))
        payload_path = safe_child(artifact_root, str(record.get("payload", "")))
        if not descriptor_path.is_file() or not payload_path.is_file():
            raise FileNotFoundError(f"v45 layer {layer} residual artifact missing")
        descriptor = json.loads(descriptor_path.read_text())
        payload = descriptor.get("payload", {})
        if (
            descriptor.get("schema") != PAYLOAD_SCHEMA
            or descriptor.get("model") != MODEL
            or descriptor.get("revision") != REVISION
            or descriptor.get("layer") != layer
            or descriptor.get("rank") != RANK
            or descriptor.get("input_features") != INPUT_FEATURES
            or descriptor.get("output_features") != OUTPUT_FEATURES
            or descriptor.get("format") != PAYLOAD_FORMAT
            or descriptor.get("preregistration_sha256")
            != residual.get("preregistration_sha256")
            or descriptor.get("training_receipt_sha256") != training_sha
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
            or record.get("persistent_payload_bytes") != PAYLOAD_BYTES
            or payload.get("bytes") != PAYLOAD_BYTES
            or payload_path.stat().st_size != PAYLOAD_BYTES
            or payload.get("sha256") != sha256_file(payload_path)
        ):
            raise ValueError(f"v45 layer {layer} residual payload changed")
        records.append(LayerPayload(layer, True, payload_path, payload["sha256"]))
        active_count += 1
        total_payload_bytes += PAYLOAD_BYTES

    if (
        gate.get("observed_layers") != LAYER_COUNT
        or gate.get("active_residual_layers") != active_count
        or gate.get("exact_bypass_layers") != LAYER_COUNT - active_count
        or gate.get("all_layers_terminal") is not True
        or gate.get("all_active_layers_calibration_not_regressed") is not True
        or gate.get("all_active_layers_validation_not_regressed") is not True
        or gate.get("all_bypass_layers_exact") is not True
        or gate.get("residual_payload_within_maximum") is not True
        or gate.get("total_residual_payload_bytes") != total_payload_bytes
    ):
        raise ValueError("v45 terminal residual gate changed")
    return records, training, residual


def build_header(
    *,
    active_count: int,
    file_bytes: int,
    training_sha256: str,
    residual_sha256: str,
    preregistration_sha256: str,
) -> bytearray:
    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    struct.pack_into(
        "<IIIIIIIIQQQI",
        header,
        8,
        VERSION,
        HEADER_BYTES,
        LAYER_COUNT,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        RANK,
        PAYLOAD_BYTES,
        TABLE_ENTRY_BYTES,
        TABLE_OFFSET,
        PAYLOAD_OFFSET,
        file_bytes,
        active_count,
    )
    header[72:92] = bytes.fromhex(REVISION)
    header[92:124] = bytes.fromhex(training_sha256)
    header[124:156] = bytes.fromhex(residual_sha256)
    header[156:188] = bytes.fromhex(preregistration_sha256)
    return header


def write_bundle(
    records: list[LayerPayload],
    output: Path,
    *,
    training_sha256: str,
    residual_sha256: str,
    preregistration_sha256: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite residual bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    active_count = sum(record.active for record in records)
    file_bytes = PAYLOAD_OFFSET + active_count * PAYLOAD_BYTES
    if shutil.disk_usage(output.parent).free < file_bytes + 1024**3:
        raise RuntimeError("insufficient disk for v56 residual bundle")
    header = build_header(
        active_count=active_count,
        file_bytes=file_bytes,
        training_sha256=training_sha256,
        residual_sha256=residual_sha256,
        preregistration_sha256=preregistration_sha256,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as target:
            target.write(header)
            offset = PAYLOAD_OFFSET
            for record in records:
                payload_sha = bytes.fromhex(record.sha256) if record.active else bytes(32)
                target.write(
                    TABLE_STRUCT.pack(
                        1 if record.active else 0,
                        record.layer,
                        offset if record.active else 0,
                        PAYLOAD_BYTES if record.active else 0,
                        payload_sha,
                        0,
                    )
                )
                if record.active:
                    offset += PAYLOAD_BYTES
            if target.tell() != PAYLOAD_OFFSET:
                raise AssertionError("v56 table alignment changed")
            for record in records:
                if not record.active:
                    continue
                assert record.path is not None
                with record.path.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != file_bytes:
            raise ValueError("v56 residual bundle byte count changed")
        root = bytes.fromhex(sha256_with_zeroed_root(temporary))
        with temporary.open("r+b") as target:
            target.seek(ROOT_OFFSET)
            target.write(root)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_bundle(
    path: Path,
    *,
    training_sha256: str,
    residual_sha256: str,
    preregistration_sha256: str,
) -> ImageSummary:
    with path.open("rb") as stream:
        header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:8] != MAGIC:
            raise ValueError("v56 residual bundle magic changed")
        fields = struct.unpack_from("<IIIIIIIIQQQI", header, 8)
        (
            version,
            header_bytes,
            layer_count,
            input_features,
            output_features,
            rank,
            payload_bytes,
            table_entry_bytes,
            table_offset,
            payload_offset,
            file_bytes,
            active_count,
        ) = fields
        root = header[ROOT_OFFSET : ROOT_OFFSET + ROOT_BYTES].hex()
        if (
            version != VERSION
            or header_bytes != HEADER_BYTES
            or layer_count != LAYER_COUNT
            or input_features != INPUT_FEATURES
            or output_features != OUTPUT_FEATURES
            or rank != RANK
            or payload_bytes != PAYLOAD_BYTES
            or table_entry_bytes != TABLE_ENTRY_BYTES
            or table_offset != TABLE_OFFSET
            or payload_offset != PAYLOAD_OFFSET
            or file_bytes != path.stat().st_size
            or header[72:92].hex() != REVISION
            or header[92:124].hex() != training_sha256
            or header[124:156].hex() != residual_sha256
            or header[156:188].hex() != preregistration_sha256
            or root != sha256_with_zeroed_root(path)
        ):
            raise ValueError("v56 residual bundle header authentication failed")
        active: list[int] = []
        bypass: list[int] = []
        expected_offset = PAYLOAD_OFFSET
        stream.seek(TABLE_OFFSET)
        entries = [stream.read(TABLE_ENTRY_BYTES) for _ in range(LAYER_COUNT)]
        for layer, entry in enumerate(entries):
            if len(entry) != TABLE_ENTRY_BYTES:
                raise EOFError("v56 residual table truncated")
            enabled, observed_layer, offset, size, payload_sha, reserved = (
                TABLE_STRUCT.unpack(entry)
            )
            if observed_layer != layer or reserved != 0:
                raise ValueError("v56 residual table ordering changed")
            if enabled == 0:
                if offset != 0 or size != 0 or payload_sha != bytes(32):
                    raise ValueError("v56 bypass table entry has payload state")
                bypass.append(layer)
            elif enabled == 1:
                if (
                    offset != expected_offset
                    or size != PAYLOAD_BYTES
                    or hash_segment(stream, offset, size) != payload_sha.hex()
                ):
                    raise ValueError("v56 active payload authentication failed")
                active.append(layer)
                expected_offset += PAYLOAD_BYTES
            else:
                raise ValueError("v56 residual table active flag changed")
        if len(active) != active_count or expected_offset != file_bytes:
            raise ValueError("v56 active residual denominator changed")
    return ImageSummary(path.stat().st_size, root, active, bypass)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--residual-receipt", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    training_sha = sha256_file(args.training_receipt)
    residual_sha = sha256_file(args.residual_receipt)
    records, _, residual = collect_payloads(
        args.training_receipt, args.residual_receipt, args.artifact_root
    )
    preregistration_sha = str(residual["preregistration_sha256"])
    repository_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if args.output.exists() or args.receipt.exists():
        if not args.output.is_file() or not args.receipt.is_file():
            raise RuntimeError("partial v56 residual materialization")
        existing = json.loads(args.receipt.read_text())
        summary = validate_bundle(
            args.output,
            training_sha256=training_sha,
            residual_sha256=residual_sha,
            preregistration_sha256=preregistration_sha,
        )
        if (
            existing.get("schema") != RECEIPT_SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("training_receipt_sha256") != training_sha
            or existing.get("residual_receipt_sha256") != residual_sha
            or existing.get("bundle", {}).get("sha256") != sha256_file(args.output)
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing v56 residual receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    write_bundle(
        records,
        args.output,
        training_sha256=training_sha,
        residual_sha256=residual_sha,
        preregistration_sha256=preregistration_sha,
    )
    summary = validate_bundle(
        args.output,
        training_sha256=training_sha,
        residual_sha256=residual_sha,
        preregistration_sha256=preregistration_sha,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL,
        "revision": REVISION,
        "repository_commit": repository_commit,
        "training_receipt_sha256": training_sha,
        "residual_receipt_sha256": residual_sha,
        "preregistration_sha256": preregistration_sha,
        "bundle": {
            "path": str(args.output),
            "bytes": summary.bytes,
            "sha256": sha256_file(args.output),
            "root_sha256": summary.root_sha256,
        },
        "active_layers": summary.active_layers,
        "bypass_layers": summary.bypass_layers,
        "active_layer_count": len(summary.active_layers),
        "bypass_layer_count": len(summary.bypass_layers),
        "persistent_dense_residual_bytes": 0,
        "rank_scratch_bytes_per_task": 64,
        "checks": {
            "all_64_layers_classified": (
                len(summary.active_layers) + len(summary.bypass_layers) == 64
            ),
            "active_count_matches_v45": (
                len(summary.active_layers)
                == residual["gate"]["active_residual_layers"]
            ),
            "payload_bytes_match_v45": (
                len(summary.active_layers) * PAYLOAD_BYTES
                == residual["gate"]["total_residual_payload_bytes"]
            ),
            "embedded_root_authenticated": (
                summary.root_sha256 == sha256_with_zeroed_root(args.output)
            ),
        },
        "claim_boundary": (
            "Exact compact v45 residual packaging only; llama.cpp placement, "
            "whole-model quality, throughput, memory, energy, novelty, and "
            "scientific-breakthrough gates remain unmeasured."
        ),
        "complete": True,
        "terminal": True,
    }
    if not all(receipt["checks"].values()):
        raise AssertionError("v56 residual bundle terminal gate failed")
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
