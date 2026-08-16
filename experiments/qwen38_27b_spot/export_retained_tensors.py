#!/usr/bin/env python3
"""Export and independently verify every non-ternary Qwen tensor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json, sha256_bytes


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_shards(model_dir: Path, shard_names: set[str], revision: str) -> dict[str, Any]:
    rows = []
    for name in sorted(shard_names):
        metadata_path = model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        lines = metadata_path.read_text().splitlines()
        if len(lines) < 2 or lines[0] != revision or len(lines[1]) != 64:
            raise ValueError(f"invalid pinned download metadata for {name}")
        expected = lines[1].lower()
        actual = file_sha256(model_dir / name)
        if actual != expected:
            raise ValueError(f"source shard SHA-256 mismatch for {name}")
        rows.append({
            "name": name,
            "bytes": (model_dir / name).stat().st_size,
            "sha256": actual,
        })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "shards": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": rows,
    }


def export_retained(
    model_dir: Path,
    mapped_names: set[str],
    output: Path,
    *,
    model_revision: str,
) -> tuple[dict[str, Any], str]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    unknown = sorted(mapped_names - set(weight_map))
    if unknown:
        raise ValueError(f"mapped tensors absent from checkpoint index: {unknown[:8]}")
    retained_names = sorted(set(weight_map) - mapped_names)
    if not retained_names:
        raise ValueError("checkpoint has no retained tensors")

    by_shard: dict[str, list[str]] = {}
    for name in retained_names:
        by_shard.setdefault(weight_map[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name).contiguous()
                tensors[name] = tensor
                entries.append({
                    "name": name,
                    "source_shard": shard,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "parameters": tensor.numel(),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "sha256": tensor_sha256(tensor),
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        save_file(tensors, temporary, metadata={
            "schema": "clark_qwen38_retained_tensors_v1",
            "model_revision": model_revision,
        })
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    del tensors
    expected = {entry["name"]: entry for entry in entries}
    with safe_open(output, framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        if actual_names != set(expected):
            raise ValueError("retained export tensor-name set mismatch")
        for name in sorted(actual_names):
            tensor = handle.get_tensor(name)
            entry = expected[name]
            if (
                list(tensor.shape) != entry["shape"]
                or str(tensor.dtype) != entry["dtype"]
                or tensor_sha256(tensor) != entry["sha256"]
            ):
                raise ValueError(f"retained export verification failed for {name}")
    return {
        "schema": "clark_qwen38_retained_tensor_manifest_v1",
        "model_revision": model_revision,
        "tensor_count": len(entries),
        "parameters": sum(entry["parameters"] for entry in entries),
        "tensor_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }, file_sha256(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--pack-receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal-retained.safetensors"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal-retained.json"),
    )
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-retained-export.json"))
    args = parser.parse_args()

    started = time.time()
    preflight = json.loads(args.preflight.read_text())
    pack = json.loads(args.pack_receipt.read_text())
    if not pack.get("promotion", {}).get("passed"):
        raise ValueError("full base-3 pack did not pass promotion")
    revision = pack["identity"]["model_revision"]
    if revision != preflight["download_receipt"]["revision"]:
        raise ValueError("pack/preflight model revision mismatch")
    index_content = (args.model_dir / "model.safetensors.index.json").read_bytes()
    if sha256_bytes(index_content) != pack["identity"]["index_sha256"]:
        raise ValueError("checkpoint index SHA-256 does not match pack identity")
    mapped_names = {row["name"] for row in pack["identity"]["mapped_sources"]}
    index = json.loads(index_content)
    source_integrity = verify_source_shards(
        args.model_dir, set(index["weight_map"].values()), revision
    )
    if (
        source_integrity["shards"] != preflight["download_receipt"]["shards"]
        or source_integrity["bytes"] != preflight["download_receipt"]["bytes"]
    ):
        raise ValueError("source shard inventory does not match download receipt")
    retained, output_sha256 = export_retained(
        args.model_dir, mapped_names, args.output, model_revision=revision
    )
    expected_parameters = int(preflight["parameter_ledger"]["retained_high_precision"])
    expected_bytes = expected_parameters * 2
    pack_ledger = pack["complete_model_ledger"]
    ledger_consistent = (
        pack_ledger.get("retained_high_precision_bytes") == expected_bytes
        and pack_ledger.get("complete_payload_bytes")
        == pack_ledger.get("ternary_code_frame_bytes", 0)
        + pack_ledger.get("fp16_scale_bytes", 0)
        + expected_bytes
        and abs(
            pack_ledger.get("payload_effective_bits_per_total_parameter", float("inf"))
            - pack_ledger.get("complete_payload_bytes", 0) * 8
            / int(preflight["parameter_ledger"]["total"])
        ) <= 1e-12
    )
    exact = (
        retained["parameters"] == expected_parameters
        and retained["tensor_bytes"] == expected_bytes
        and all(entry["dtype"] == "torch.bfloat16" for entry in retained["entries"])
        and ledger_consistent
    )
    atomic_bytes(args.manifest, canonical_json(retained))
    receipt = {
        "schema": "clark_qwen38_retained_export_receipt_v1",
        "hypothesis": "Every parameter excluded from ternary coding can be exported exactly into one authenticated BF16 bundle.",
        "model_revision": revision,
        "mapped_tensor_count": len(mapped_names),
        "source_integrity": source_integrity,
        "retained": {
            "path": str(args.output),
            "file_sha256": output_sha256,
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_bytes(args.manifest.read_bytes()),
            "tensor_count": retained["tensor_count"],
            "parameters": retained["parameters"],
            "tensor_bytes": retained["tensor_bytes"],
        },
        "complete_parameter_bundle": {
            "ternary_parameters": int(preflight["parameter_ledger"]["ternary_payload"]),
            "retained_parameters": retained["parameters"],
            "total_parameters": int(preflight["parameter_ledger"]["total"]),
            "ternary_code_frame_bytes": pack_ledger["ternary_code_frame_bytes"],
            "fp16_scale_bytes": pack_ledger["fp16_scale_bytes"],
            "retained_tensor_bytes": retained["tensor_bytes"],
            "complete_payload_bytes": pack_ledger["complete_payload_bytes"],
            "effective_bits_per_total_parameter": pack_ledger[
                "payload_effective_bits_per_total_parameter"
            ],
        },
        "elapsed_seconds": time.time() - started,
        "promotion": {
            "passed": exact,
            "scope": "all retained checkpoint tensors; configuration and tokenizer remain in the pinned source directory",
        },
        "stop_condition": "All retained tensors or first index, dtype, shape, hash, byte-ledger, or atomic-write failure.",
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
