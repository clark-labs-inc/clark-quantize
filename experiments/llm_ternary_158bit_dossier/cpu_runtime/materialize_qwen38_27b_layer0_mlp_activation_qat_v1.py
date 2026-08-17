#!/usr/bin/env python3
"""Materialize the trained Qwen3.8-27B layer-0 CQ3 payloads into a GGUF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGETS = {
    "blk.0.ffn_down.weight": (5120, 17408),
    "blk.0.ffn_gate.weight": (17408, 5120),
    "blk.0.ffn_up.weight": (17408, 5120),
}
GGML_TYPE_CQ3_0 = 43


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_segment(handle: BinaryIO, digest: Any, start: int, stop: int) -> None:
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
    if any(right[0] < left[1] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("overlapping exclusion ranges")
    digest = hashlib.sha256()
    cursor = 0
    with path.open("rb") as handle:
        for start, stop in normalized:
            hash_segment(handle, digest, cursor, start)
            cursor = stop
        hash_segment(handle, digest, cursor, path.stat().st_size)
    return digest.hexdigest()


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", type=Path,
        default=Path("/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0.gguf"),
    )
    parser.add_argument(
        "--template-receipt", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-v1/conversion-receipt.json"),
    )
    parser.add_argument(
        "--training-receipt", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-v1.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0-activation-QAT-v1.gguf"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-materialization-v1.json"),
    )
    args = parser.parse_args()

    template_receipt = json.loads(args.template_receipt.read_text())
    training = json.loads(args.training_receipt.read_text())
    if (
        template_receipt.get("valid") is not True
        or template_receipt.get("candidate", {}).get("sha256") != sha256_file(args.template)
        or training.get("model_id") != MODEL_ID
        or training.get("model_revision") != MODEL_REVISION
        or training.get("numeric_promotion_pass") is not True
        or training.get("complete") is not True
    ):
        raise ValueError("template or Qwen3.8-27B training receipt is not promoted")

    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError("set PYTHONPATH to the pinned llama.cpp gguf-py directory") from error
    reader = GGUFReader(str(args.template), mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if not set(TARGETS) <= set(tensors):
        raise ValueError("structural GGUF template lacks one or more target tensors")
    payloads = {
        item["gguf_tensor"]: Path(item["payload"]["path"])
        for item in training["tensors"]
    }
    if set(payloads) != set(TARGETS):
        raise ValueError("training receipt target set mismatch")

    replacements: list[tuple[int, int, Path, str]] = []
    for name, hf_shape in TARGETS.items():
        tensor = tensors[name]
        payload = payloads[name]
        receipt_entry = next(item for item in training["tensors"] if item["gguf_tensor"] == name)
        logical_hf_shape = tuple(reversed([int(value) for value in tensor.shape]))
        if (
            int(tensor.tensor_type) != GGML_TYPE_CQ3_0
            or logical_hf_shape != hf_shape
            or tensor.n_bytes != payload.stat().st_size
            or receipt_entry["payload"]["bytes"] != payload.stat().st_size
            or receipt_entry["payload"]["sha256"] != sha256_file(payload)
        ):
            raise ValueError(f"trained payload does not fit GGUF target {name}")
        start = int(tensor.data_offset)
        replacements.append((start, start + int(tensor.n_bytes), payload, name))
    del reader

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {args.output}")
    available = shutil.disk_usage(args.output.parent).free
    required = args.template.stat().st_size + 2 * 1024**3
    if available < required:
        raise RuntimeError(f"materialization requires {required} free bytes, found {available}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    temporary = Path(temporary_name)
    try:
        with args.template.open("rb") as source, os.fdopen(descriptor, "w+b") as destination:
            shutil.copyfileobj(source, destination, length=64 * 1024 * 1024)
            for start, stop, payload, _ in replacements:
                destination.seek(start)
                with payload.open("rb") as payload_handle:
                    shutil.copyfileobj(payload_handle, destination, length=16 * 1024 * 1024)
                if destination.tell() != stop:
                    raise ValueError("trained payload write ended at the wrong GGUF offset")
            destination.flush()
            os.fsync(destination.fileno())
        if temporary.stat().st_size != args.template.stat().st_size:
            raise ValueError("materialized GGUF byte count changed")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    ranges = [(start, stop) for start, stop, _, _ in replacements]
    template_non_target = sha256_excluding(args.template, ranges)
    output_non_target = sha256_excluding(args.output, ranges)
    output_reader = GGUFReader(str(args.output), mode="r")
    output_tensors = {tensor.name: tensor for tensor in output_reader.tensors}
    observed_targets = {}
    for start, stop, payload, name in replacements:
        tensor = output_tensors[name]
        observed = hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()
        expected = sha256_file(payload)
        observed_targets[name] = {
            "offset": start,
            "bytes": stop - start,
            "payload_path": str(payload),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "match": observed == expected,
        }
    del output_reader
    checks = {
        "same_complete_file_bytes": args.output.stat().st_size == args.template.stat().st_size,
        "all_non_target_file_bytes_identical": template_non_target == output_non_target,
        "all_target_payloads_match_training": all(item["match"] for item in observed_targets.values()),
        "exact_target_count": len(observed_targets) == 3,
    }
    receipt = {
        "schema": "qwen38_27b_layer0_mlp_activation_qat_materialization_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "template": {
            "path": str(args.template),
            "bytes": args.template.stat().st_size,
            "sha256": sha256_file(args.template),
            "receipt": str(args.template_receipt),
            "receipt_sha256": sha256_file(args.template_receipt),
        },
        "training_receipt": {
            "path": str(args.training_receipt),
            "sha256": sha256_file(args.training_receipt),
        },
        "candidate": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "non_target_file_bytes_sha256": {
            "template": template_non_target,
            "candidate": output_non_target,
        },
        "targets": observed_targets,
        "checks": checks,
        "valid": all(checks.values()),
        "claim_boundary": "byte-scope-authenticated Qwen3.8-27B GGUF materialization; capability and throughput remain unmeasured",
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
