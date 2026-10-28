#!/usr/bin/env python3
"""Verify that a GGUF conversion changed exactly a preregistered tensor scope."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def inventory(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(
            "gguf-py is required; set PYTHONPATH to llama.cpp/gguf-py"
        ) from error

    file_sha256 = sha256_file(path)
    reader = GGUFReader(str(path), mode="r")
    tensors: dict[str, dict[str, Any]] = {}
    for index, tensor in enumerate(reader.tensors, start=1):
        if tensor.name in tensors:
            raise RuntimeError(f"duplicate GGUF tensor: {tensor.name}")
        payload = memoryview(tensor.data).cast("B")
        tensors[tensor.name] = {
            "shape": [int(value) for value in tensor.shape],
            "type": int(tensor.tensor_type),
            "payload_bytes": int(payload.nbytes),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        if index % 64 == 0 or index == len(reader.tensors):
            print(
                json.dumps(
                    {
                        "model": str(path),
                        "tensors_hashed": index,
                        "total": len(reader.tensors),
                    }
                ),
                flush=True,
            )
    del reader
    gc.collect()
    return file_sha256, tensors


def aggregate_digest(tensors: dict[str, dict[str, Any]], names: list[str]) -> str:
    return canonical_digest(
        [{"name": name, **tensors[name]} for name in names]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--target-regex", required=True)
    parser.add_argument("--expected-target-count", required=True, type=int)
    parser.add_argument("--baseline-target-type", required=True, type=int)
    parser.add_argument("--candidate-target-type", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} GGUF not found: {path}")

    model_sha256: dict[str, str] = {}
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        print(json.dumps({"inventory_started": label, "path": str(path)}), flush=True)
        model_sha256[label], inventories[label] = inventory(path)

    baseline_names = set(inventories["baseline"])
    candidate_names = set(inventories["candidate"])
    common_names = sorted(baseline_names & candidate_names)
    pattern = re.compile(args.target_regex)
    target_names = [name for name in common_names if pattern.fullmatch(name)]
    target_set = set(target_names)
    non_target_names = [name for name in common_names if name not in target_set]

    same_names = baseline_names == candidate_names
    same_shapes = same_names and all(
        inventories["baseline"][name]["shape"]
        == inventories["candidate"][name]["shape"]
        for name in common_names
    )
    non_targets_identical = same_names and all(
        inventories["baseline"][name] == inventories["candidate"][name]
        for name in non_target_names
    )
    baseline_target_types_valid = len(target_names) == args.expected_target_count and all(
        inventories["baseline"][name]["type"] == args.baseline_target_type
        for name in target_names
    )
    candidate_target_types_valid = len(target_names) == args.expected_target_count and all(
        inventories["candidate"][name]["type"] == args.candidate_target_type
        for name in target_names
    )
    target_payloads_changed = len(target_names) == args.expected_target_count and all(
        inventories["baseline"][name]["payload_sha256"]
        != inventories["candidate"][name]["payload_sha256"]
        for name in target_names
    )
    checks = {
        "same_tensor_names": same_names,
        "same_logical_shapes": same_shapes,
        "exact_target_count": len(target_names) == args.expected_target_count,
        "baseline_target_types_valid": baseline_target_types_valid,
        "candidate_target_types_valid": candidate_target_types_valid,
        "all_target_payloads_changed": target_payloads_changed,
        "all_non_target_types_sizes_and_payloads_identical": non_targets_identical,
    }

    receipt: dict[str, Any] = {
        "schema": "gguf_selective_conversion_verification_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "baseline": {
                "path": str(baseline),
                "bytes": baseline.stat().st_size,
                "sha256": model_sha256["baseline"],
            },
            "candidate": {
                "path": str(candidate),
                "bytes": candidate.stat().st_size,
                "sha256": model_sha256["candidate"],
            },
        },
        "target_regex": args.target_regex,
        "expected_target_count": args.expected_target_count,
        "expected_types": {
            "baseline": args.baseline_target_type,
            "candidate": args.candidate_target_type,
        },
        "tensor_count": len(common_names),
        "target_count": len(target_names),
        "non_target_count": len(non_target_names),
        "target_names": target_names,
        "checks": checks,
        "aggregate_non_target_digest": {
            label: aggregate_digest(inventories[label], non_target_names)
            for label in inventories
        },
        "targets": {
            name: {label: inventories[label][name] for label in inventories}
            for name in target_names
        },
    }
    receipt["valid"] = all(checks.values())
    receipt["receipt_content_sha256"] = canonical_digest(receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(receipt, indent=2), flush=True)
    if not receipt["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
