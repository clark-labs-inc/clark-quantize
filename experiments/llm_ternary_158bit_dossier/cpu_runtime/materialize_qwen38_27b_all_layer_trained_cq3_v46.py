#!/usr/bin/env python3
"""Overlay all 192 trained Qwen3.8-27B CQ3 tensors into a GGUF template.

The structural template must already contain CQ3_0 slots for every MLP tensor.
This tool does not re-quantize weights: it authenticates and copies the exact
v44 trained bytes into those slots while proving that every non-target byte is
unchanged.  AFF3 attention tensors and post-block residuals are intentionally
outside this materialization stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRAINING_SCHEMA = "qwen38_27b_all_layer_mixed_training_receipt_v44"
LAYER_SCHEMA = "qwen38_27b_all_layer_mixed_trained_layer_v44"
TENSOR_SCHEMA = "qwen38_27b_calibration_ranked_mixed_trained_cq3_tensor_v1"
SCOPE_SCHEMA = "gguf_selective_conversion_verification_v1"
GGML_TYPE_CQ3_0 = 43
CQ3_BLOCK_WEIGHTS = 256
CQ3_BLOCK_BYTES = 68
TARGET_SHAPES = {
    "ffn_gate": (17408, 5120),
    "ffn_up": (17408, 5120),
    "ffn_down": (5120, 17408),
}


@dataclass(frozen=True)
class PayloadRecord:
    tensor: str
    layer: int
    shape: tuple[int, int]
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TensorSlot:
    tensor_type: int
    shape: tuple[int, int]
    offset: int
    bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def expected_payload_bytes(shape: tuple[int, int]) -> int:
    rows, columns = shape
    if rows <= 0 or columns <= 0 or columns % CQ3_BLOCK_WEIGHTS:
        raise ValueError("CQ3_0 shape is not block aligned")
    return rows * (columns // CQ3_BLOCK_WEIGHTS) * CQ3_BLOCK_BYTES


def expected_targets(
    layer: int,
    *,
    target_shapes: Mapping[str, tuple[int, int]] = TARGET_SHAPES,
) -> dict[str, tuple[int, int]]:
    return {
        f"blk.{layer}.{kind}.weight": tuple(shape)
        for kind, shape in target_shapes.items()
    }


def _safe_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("artifact reference must be relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("artifact reference escapes its root") from error
    return resolved


def collect_training_payloads(
    receipt_path: Path,
    artifact_root: Path,
    *,
    required_layers: int = 64,
    target_shapes: Mapping[str, tuple[int, int]] = TARGET_SHAPES,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> list[PayloadRecord]:
    """Authenticate the complete v44 CQ3 inventory without decoding it."""
    receipt = json.loads(receipt_path.read_text())
    layers = receipt.get("layers", [])
    if (
        receipt.get("schema") != TRAINING_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("gate", {}).get("passed") is not True
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or len(layers) != required_layers
        or [item.get("layer") for item in layers] != list(range(required_layers))
    ):
        raise ValueError("v44 all-layer training receipt is not promoted")

    records: list[PayloadRecord] = []
    seen: set[str] = set()
    for layer_index, layer in enumerate(layers):
        if (
            layer.get("schema") != LAYER_SCHEMA
            or layer.get("model") != MODEL_ID
            or layer.get("revision") != MODEL_REVISION
            or layer.get("repository_commit") != receipt.get("repository_commit")
            or layer.get("preregistration_sha256")
            != receipt.get("preregistration_sha256")
            or layer.get("direct_receipt_sha256")
            != receipt.get("direct_receipt_sha256")
            or layer.get("passed") is not True
            or layer.get("complete") is not True
            or layer.get("terminal") is not True
        ):
            raise ValueError(f"v44 layer {layer_index} is not promoted")

        expected = expected_targets(layer_index, target_shapes=target_shapes)
        descriptor_references = layer.get("cq3_descriptors", [])
        training_by_name = {
            item.get("gguf_tensor"): item for item in layer.get("cq3_training", [])
        }
        if (
            len(descriptor_references) != len(expected)
            or len(training_by_name) != len(expected)
            or set(training_by_name) != set(expected)
        ):
            raise ValueError(f"v44 layer {layer_index} CQ3 denominator changed")

        observed: set[str] = set()
        for reference in descriptor_references:
            descriptor_path = _safe_child(artifact_root, str(reference))
            if not descriptor_path.is_file() or descriptor_path.suffix != ".json":
                raise FileNotFoundError(descriptor_path)
            descriptor = json.loads(descriptor_path.read_text())
            tensor = descriptor.get("gguf_tensor")
            if tensor not in expected or tensor in observed or tensor in seen:
                raise ValueError(f"unexpected or duplicate CQ3 tensor: {tensor}")
            observed.add(tensor)
            seen.add(tensor)
            shape = tuple(int(value) for value in descriptor.get("shape", []))
            payload_path = descriptor_path.with_suffix(".cq3")
            payload = descriptor.get("payload", {})
            training_payload = training_by_name[tensor].get("payload", {})
            expected_bytes = expected_payload_bytes(shape)
            if (
                descriptor.get("schema") != TENSOR_SCHEMA
                or descriptor.get("repository_commit")
                != receipt.get("repository_commit")
                or descriptor.get("preregistration_sha256")
                != receipt.get("preregistration_sha256")
                or descriptor.get("training_amendment_sha256")
                != receipt.get("preregistration_sha256")
                or descriptor.get("training_cache_sha256")
                != receipt.get("full_context_receipt_sha256")
                or descriptor.get("layer") != layer_index
                or shape != expected[tensor]
                or descriptor.get("group_size") != 128
                or descriptor.get("row_shard") != 128
                or payload.get("bytes") != expected_bytes
                or training_payload.get("bytes") != expected_bytes
                or training_payload.get("sha256") != payload.get("sha256")
                or descriptor.get("all_pack_unpack_mappings_exact") is not True
                or descriptor.get("complete") is not True
                or descriptor.get("terminal") is not True
                or not payload_path.is_file()
                or payload_path.stat().st_size != expected_bytes
                or file_hasher(payload_path) != payload.get("sha256")
            ):
                raise ValueError(f"v44 CQ3 payload authentication failed: {tensor}")
            records.append(
                PayloadRecord(
                    tensor=tensor,
                    layer=layer_index,
                    shape=shape,
                    path=payload_path,
                    bytes=expected_bytes,
                    sha256=str(payload["sha256"]),
                )
            )
        if observed != set(expected):
            raise ValueError(f"v44 layer {layer_index} CQ3 target set changed")

    expected_count = required_layers * len(target_shapes)
    if len(records) != expected_count or len(seen) != expected_count:
        raise ValueError("v44 complete CQ3 tensor denominator changed")
    return sorted(records, key=lambda item: item.tensor)


def validate_template_scope(
    template: Path,
    scope_receipt_path: Path,
    payloads: list[PayloadRecord],
    *,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    scope = json.loads(scope_receipt_path.read_text())
    candidate = scope.get("models", {}).get("candidate", {})
    names = sorted(item.tensor for item in payloads)
    checks = scope.get("checks", {})
    if (
        scope.get("schema") != SCOPE_SCHEMA
        or scope.get("valid") is not True
        or scope.get("target_count") != len(names)
        or sorted(scope.get("target_names", [])) != names
        or scope.get("expected_types", {}).get("candidate") != GGML_TYPE_CQ3_0
        or checks.get("all_non_target_types_sizes_and_payloads_identical") is not True
        or checks.get("candidate_target_types_valid") is not True
        or checks.get("exact_target_count") is not True
        or not template.is_file()
        or candidate.get("bytes") != template.stat().st_size
        or candidate.get("sha256") != file_hasher(template)
    ):
        raise ValueError("all-MLP CQ3 structural GGUF template is not authenticated")
    return scope


def validate_tensor_slots(
    slots: Mapping[str, TensorSlot], payloads: list[PayloadRecord]
) -> list[tuple[int, int, PayloadRecord]]:
    expected_names = {item.tensor for item in payloads}
    if set(slots) != expected_names:
        raise ValueError("GGUF CQ3 slot denominator changed")
    ranges: list[tuple[int, int, PayloadRecord]] = []
    for payload in payloads:
        slot = slots[payload.tensor]
        if (
            slot.tensor_type != GGML_TYPE_CQ3_0
            or slot.shape != payload.shape
            or slot.bytes != payload.bytes
            or slot.offset < 0
        ):
            raise ValueError(f"GGUF CQ3 slot does not fit payload: {payload.tensor}")
        ranges.append((slot.offset, slot.offset + slot.bytes, payload))
    ranges.sort(key=lambda item: item[0])
    if any(right[0] < left[1] for left, right in pairwise(ranges)):
        raise ValueError("GGUF CQ3 target ranges overlap")
    return ranges


def _hash_segment(handle: BinaryIO, digest: Any, start: int, stop: int) -> None:
    handle.seek(start)
    remaining = stop - start
    while remaining:
        block = handle.read(min(16 * 1024 * 1024, remaining))
        if not block:
            raise EOFError("unexpected EOF while hashing GGUF non-target bytes")
        digest.update(block)
        remaining -= len(block)


def sha256_excluding(path: Path, ranges: list[tuple[int, int]]) -> str:
    normalized = sorted(ranges)
    if any(start < 0 or stop <= start for start, stop in normalized):
        raise ValueError("invalid exclusion range")
    if any(right[0] < left[1] for left, right in pairwise(normalized)):
        raise ValueError("overlapping exclusion ranges")
    if normalized and normalized[-1][1] > path.stat().st_size:
        raise ValueError("exclusion range extends beyond the file")
    digest = hashlib.sha256()
    cursor = 0
    with path.open("rb") as handle:
        for start, stop in normalized:
            _hash_segment(handle, digest, cursor, start)
            cursor = stop
        _hash_segment(handle, digest, cursor, path.stat().st_size)
    return digest.hexdigest()


def atomic_overlay(
    template: Path,
    output: Path,
    ranges: list[tuple[int, int, PayloadRecord]],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    required = template.stat().st_size + 2 * 1024**3
    available = shutil.disk_usage(output.parent).free
    if available < required:
        raise RuntimeError(
            f"materialization requires {required} free bytes, found {available}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with template.open("rb") as source, os.fdopen(descriptor, "w+b") as target:
            shutil.copyfileobj(source, target, length=64 * 1024 * 1024)
            for start, stop, payload in ranges:
                target.seek(start)
                with payload.path.open("rb") as payload_handle:
                    shutil.copyfileobj(payload_handle, target, length=16 * 1024 * 1024)
                if target.tell() != stop:
                    raise ValueError(
                        f"CQ3 overlay ended at wrong offset: {payload.tensor}"
                    )
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != template.stat().st_size:
            raise ValueError("materialized GGUF byte count changed")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _slots_from_gguf(reader: Any, names: set[str]) -> dict[str, TensorSlot]:
    slots: dict[str, TensorSlot] = {}
    for tensor in reader.tensors:
        if tensor.name not in names:
            continue
        slots[tensor.name] = TensorSlot(
            tensor_type=int(tensor.tensor_type),
            shape=tuple(reversed([int(value) for value in tensor.shape])),
            offset=int(tensor.data_offset),
            bytes=int(tensor.n_bytes),
        )
    return slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--template-scope-receipt", required=True, type=Path)
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payloads = collect_training_payloads(args.training_receipt, args.artifact_root)
    scope = validate_template_scope(
        args.template, args.template_scope_receipt, payloads
    )
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(
            "set PYTHONPATH to the pinned llama.cpp gguf-py directory"
        ) from error

    reader = GGUFReader(str(args.template), mode="r")
    slots = _slots_from_gguf(reader, {item.tensor for item in payloads})
    ranges = validate_tensor_slots(slots, payloads)
    del reader
    byte_ranges = [(start, stop) for start, stop, _ in ranges]
    template_non_target_sha = sha256_excluding(args.template, byte_ranges)
    atomic_overlay(args.template, args.output, ranges)
    output_non_target_sha = sha256_excluding(args.output, byte_ranges)

    output_reader = GGUFReader(str(args.output), mode="r")
    output_slots = _slots_from_gguf(output_reader, {item.tensor for item in payloads})
    validate_tensor_slots(output_slots, payloads)
    output_by_name = {tensor.name: tensor for tensor in output_reader.tensors}
    target_results: dict[str, dict[str, Any]] = {}
    for start, stop, payload in ranges:
        observed = hashlib.sha256(
            memoryview(output_by_name[payload.tensor].data).cast("B")
        ).hexdigest()
        target_results[payload.tensor] = {
            "layer": payload.layer,
            "shape": list(payload.shape),
            "offset": start,
            "bytes": stop - start,
            "trained_payload_path": str(payload.path),
            "expected_sha256": payload.sha256,
            "observed_sha256": observed,
            "match": observed == payload.sha256,
        }
    del output_reader

    checks = {
        "training_gate_passed": True,
        "template_scope_valid": scope["valid"] is True,
        "exact_target_count": len(target_results) == 192,
        "same_complete_file_bytes": (
            args.output.stat().st_size == args.template.stat().st_size
        ),
        "all_non_target_file_bytes_identical": (
            template_non_target_sha == output_non_target_sha
        ),
        "all_trained_target_payloads_exact": all(
            item["match"] for item in target_results.values()
        ),
    }
    receipt = {
        "schema": "qwen38_27b_all_layer_trained_cq3_gguf_materialization_v46",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "training_receipt": {
            "path": str(args.training_receipt),
            "sha256": sha256_file(args.training_receipt),
        },
        "artifact_root": str(args.artifact_root),
        "template": {
            "path": str(args.template),
            "bytes": args.template.stat().st_size,
            "sha256": sha256_file(args.template),
            "scope_receipt": str(args.template_scope_receipt),
            "scope_receipt_sha256": sha256_file(args.template_scope_receipt),
        },
        "candidate": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "non_target_file_bytes_sha256": {
            "template": template_non_target_sha,
            "candidate": output_non_target_sha,
        },
        "targets": target_results,
        "checks": checks,
        "valid": all(checks.values()),
        "claim_boundary": (
            "exact runnable all-MLP trained-CQ3 GGUF assembly only; AFF3, "
            "post-block residuals, whole-model quality, throughput, memory, "
            "energy, novelty, and breakthrough remain unmeasured"
        ),
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
