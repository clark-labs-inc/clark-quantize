#!/usr/bin/env python3
"""Authenticate every Qwen3.8-27B BF16 shard before whole-model conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def authenticate_source(
    *,
    model_dir: Path,
    parent_receipt_path: Path,
    download_receipt_path: Path,
    tree_manifest_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    preregistration = load_json(preregistration_path)
    expected = preregistration["source_authentication"]
    model = preregistration["model"]
    revision = preregistration["revision"]

    require_sha256(
        parent_receipt_path,
        expected["parent_receipt_sha256"],
        "parent-model receipt",
    )
    require_sha256(
        download_receipt_path,
        expected["download_receipt_sha256"],
        "download receipt",
    )
    require_sha256(
        tree_manifest_path,
        expected["hf_tree_manifest_sha256"],
        "Hugging Face tree manifest",
    )

    parent_receipt = load_json(parent_receipt_path)
    matching = [
        row for row in parent_receipt.get("models", []) if row.get("repo_id") == model
    ]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {model} parent-model receipt")
    parent = matching[0]
    if parent.get("complete") is not True or parent.get("revision") != revision:
        raise ValueError("parent-model receipt is incomplete or names another revision")

    download_receipt = load_json(download_receipt_path)
    if (
        download_receipt.get("model_id") != model
        or download_receipt.get("revision") != revision
        or download_receipt.get("schema") != "qwen38_model_download_v1"
    ):
        raise ValueError("download receipt does not bind the frozen model revision")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_sha = require_sha256(config_path, expected["config_sha256"], "config")
    index_sha = require_sha256(index_path, expected["model_index_sha256"], "index")
    if parent.get("config_sha256") != config_sha or parent.get("index_sha256") != index_sha:
        raise ValueError("parent-model receipt disagrees with authenticated config/index")
    if download_receipt.get("index_sha256") != index_sha:
        raise ValueError("download receipt disagrees with authenticated model index")

    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight map")
    shard_names = sorted(set(weight_map.values()))
    if not shard_names or not all(isinstance(name, str) for name in shard_names):
        raise ValueError("model index has invalid shard names")
    if len(shard_names) != int(expected["shards"]):
        raise ValueError("model index shard count differs from the frozen contract")

    tree = load_json(tree_manifest_path)
    tree_files = tree.get("files")
    if tree.get("format_version") != 1 or not isinstance(tree_files, dict):
        raise ValueError("unsupported Hugging Face tree manifest")

    shard_receipts: list[dict[str, Any]] = []
    total_bytes = 0
    for name in shard_names:
        descriptor = tree_files.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"tree manifest has no descriptor for {name}")
        expected_digest = descriptor.get("lfs_sha256")
        expected_bytes = descriptor.get("lfs_size")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"tree manifest has no LFS SHA-256 for {name}")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"tree manifest has no valid LFS byte count for {name}")
        path = model_dir / name
        observed_bytes = path.stat().st_size
        if observed_bytes != expected_bytes:
            raise ValueError(
                f"{name} byte count mismatch: expected {expected_bytes}, "
                f"observed {observed_bytes}"
            )
        observed_digest = require_sha256(path, expected_digest, name)
        total_bytes += observed_bytes
        shard_receipts.append(
            {"name": name, "bytes": observed_bytes, "sha256": observed_digest}
        )

    expected_total = int(expected["safetensors_bytes"])
    index_total = int(index.get("metadata", {}).get("total_size", -1))
    if total_bytes != expected_total:
        raise ValueError(
            f"safetensor byte total mismatch: expected {expected_total}, "
            f"observed {total_bytes}"
        )
    if int(parent.get("safetensors_bytes", -1)) != total_bytes:
        raise ValueError("parent-model receipt has another safetensor byte total")
    if int(download_receipt.get("bytes", -1)) != total_bytes:
        raise ValueError("download receipt has another safetensor byte total")
    if int(parent.get("index_total_size", -1)) != index_total:
        raise ValueError("parent-model receipt has another index tensor-byte total")
    if int(parent.get("shards", -1)) != len(shard_names):
        raise ValueError("parent-model receipt has another shard count")
    if int(download_receipt.get("shards", -1)) != len(shard_names):
        raise ValueError("download receipt has another shard count")

    return {
        "schema": "qwen38_27b_hf_source_authentication_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "revision": revision,
        "model_directory": str(model_dir.resolve()),
        "config_sha256": config_sha,
        "model_index_sha256": index_sha,
        "hf_tree_manifest_sha256": expected["hf_tree_manifest_sha256"],
        "shards": shard_receipts,
        "shard_count": len(shard_receipts),
        "safetensors_bytes": total_bytes,
        "index_tensor_bytes": index_total,
        "complete": True,
        "valid": True,
        "terminal": True,
        "claim_boundary": (
            "source-byte authentication only; no conversion, quality, storage, "
            "throughput, or breakthrough claim"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--tree-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = authenticate_source(
        model_dir=args.model_dir,
        parent_receipt_path=args.parent_receipt,
        download_receipt_path=args.download_receipt,
        tree_manifest_path=args.tree_manifest,
        preregistration_path=args.preregistration,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)


if __name__ == "__main__":
    main()
