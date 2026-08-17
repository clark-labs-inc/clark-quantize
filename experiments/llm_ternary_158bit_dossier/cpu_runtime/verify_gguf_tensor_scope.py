#!/usr/bin/env python3
"""Prove that a selective GGUF quantization changed only preregistered tensors."""

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
                    {"model": str(path), "tensors_hashed": index, "total": len(reader.tensors)}
                ),
                flush=True,
            )
    del reader
    gc.collect()
    return file_sha256, tensors


def aggregate_payload_digest(
    tensors: dict[str, dict[str, Any]], names: list[str]
) -> str:
    return canonical_digest(
        [
            {
                "name": name,
                "shape": tensors[name]["shape"],
                "type": tensors[name]["type"],
                "payload_bytes": tensors[name]["payload_bytes"],
                "payload_sha256": tensors[name]["payload_sha256"],
            }
            for name in names
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--target-regex", required=True)
    parser.add_argument("--baseline-target-type", required=True, type=int)
    parser.add_argument("--packed-target-type", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    model_paths = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
        "control": args.control.resolve(),
    }
    for label, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} GGUF not found: {path}")

    model_sha256: dict[str, str] = {}
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in model_paths.items():
        print(json.dumps({"inventory_started": label, "path": str(path)}), flush=True)
        model_sha256[label], inventories[label] = inventory(path)

    names = {label: set(value) for label, value in inventories.items()}
    common_names = sorted(set.intersection(*names.values()))
    target_pattern = re.compile(args.target_regex)
    target_names = [name for name in common_names if target_pattern.fullmatch(name)]
    non_target_names = [name for name in common_names if name not in set(target_names)]

    same_names = names["baseline"] == names["candidate"] == names["control"]
    same_shapes = same_names and all(
        inventories["baseline"][name]["shape"]
        == inventories["candidate"][name]["shape"]
        == inventories["control"][name]["shape"]
        for name in common_names
    )
    non_target_identical = same_names and all(
        inventories["baseline"][name] == inventories["candidate"][name]
        == inventories["control"][name]
        for name in non_target_names
    )
    baseline_types_valid = bool(target_names) and all(
        inventories["baseline"][name]["type"] == args.baseline_target_type
        for name in target_names
    )
    packed_types_valid = bool(target_names) and all(
        inventories[label][name]["type"] == args.packed_target_type
        for label in ("candidate", "control")
        for name in target_names
    )
    target_candidate_control_differences = [
        name
        for name in target_names
        if inventories["candidate"][name]["payload_sha256"]
        != inventories["control"][name]["payload_sha256"]
    ]
    checks = {
        "same_tensor_names": same_names,
        "same_logical_shapes": same_shapes,
        "exactly_one_or_more_targets": bool(target_names),
        "baseline_target_types_valid": baseline_types_valid,
        "candidate_and_control_target_types_valid": packed_types_valid,
        "all_non_target_types_sizes_and_payloads_identical": non_target_identical,
        "candidate_differs_from_control_on_at_least_one_target": bool(
            target_candidate_control_differences
        ),
    }

    receipt: dict[str, Any] = {
        "schema": "gguf_tensor_scope_verification_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            label: {"path": str(model_paths[label]), "sha256": model_sha256[label]}
            for label in model_paths
        },
        "target_regex": args.target_regex,
        "expected_types": {
            "baseline_target": args.baseline_target_type,
            "candidate_and_control_target": args.packed_target_type,
        },
        "tensor_count": len(common_names),
        "target_count": len(target_names),
        "non_target_count": len(non_target_names),
        "target_names": target_names,
        "target_candidate_control_differences": target_candidate_control_differences,
        "checks": checks,
        "aggregate_non_target_digest": {
            label: aggregate_payload_digest(inventories[label], non_target_names)
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
