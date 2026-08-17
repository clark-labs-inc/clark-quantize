"""Materialize a small safetensors weight patch as a complete HF model tree.

Unchanged top-level model files are hardlinked. Only shards containing patched
tensors are rewritten, and the source model is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def materialize(model_dir: Path, patch_path: Path, out_dir: Path, receipt_path: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output model: {out_dir}")
    partial_dir = out_dir.with_name(f".{out_dir.name}.partial")
    if partial_dir.exists():
        raise FileExistsError(f"partial output already exists: {partial_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    patch = load_file(str(patch_path), device="cpu")
    unknown = sorted(set(patch) - set(weight_map))
    if unknown:
        raise KeyError(f"patch contains {len(unknown)} tensors absent from model index: {unknown[:8]}")
    if not patch:
        raise ValueError("weight patch is empty")

    patched_by_shard: dict[str, list[str]] = {}
    for tensor_name in sorted(patch):
        patched_by_shard.setdefault(weight_map[tensor_name], []).append(tensor_name)

    source_hashes_before = {
        shard: sha256_file(model_dir / shard)
        for shard in sorted(patched_by_shard)
    }
    partial_dir.mkdir(parents=True)
    hardlinked: list[str] = []
    rewritten: list[dict[str, Any]] = []

    for source in sorted(model_dir.iterdir()):
        if not source.is_file() or source.name in patched_by_shard:
            continue
        destination = partial_dir / source.name
        os.link(source, destination)
        hardlinked.append(source.name)

    for shard, tensor_names in sorted(patched_by_shard.items()):
        source = model_dir / shard
        with safe_open(source, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        for tensor_name in tensor_names:
            replacement = patch[tensor_name]
            original = tensors[tensor_name]
            if replacement.shape != original.shape or replacement.dtype != original.dtype:
                raise ValueError(
                    f"patch mismatch for {tensor_name}: "
                    f"expected {tuple(original.shape)} {original.dtype}, "
                    f"got {tuple(replacement.shape)} {replacement.dtype}"
                )
            tensors[tensor_name] = replacement.contiguous()
        destination = partial_dir / shard
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, destination)
        rewritten.append(
            {
                "shard": shard,
                "patched_tensors": tensor_names,
                "size_bytes": destination.stat().st_size,
            }
        )
        del tensors

    os.replace(partial_dir, out_dir)

    for tensor_name, expected in patch.items():
        shard = weight_map[tensor_name]
        with safe_open(out_dir / shard, framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(tensor_name)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    source_hashes_after = {
        shard: sha256_file(model_dir / shard)
        for shard in sorted(patched_by_shard)
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source model shard changed during materialization")

    receipt = {
        "schema": "clark_hf_weight_patch_materialization_v1",
        "model_dir": str(model_dir),
        "model_index_sha256": sha256_file(index_path),
        "patch": str(patch_path),
        "patch_sha256": sha256_file(patch_path),
        "patch_tensor_count": len(patch),
        "output_dir": str(out_dir),
        "hardlinked_files": hardlinked,
        "rewritten_shards": rewritten,
        "source_rewritten_shard_sha256": source_hashes_before,
        "output_rewritten_shard_sha256": {
            row["shard"]: sha256_file(out_dir / row["shard"])
            for row in rewritten
        },
        "verification": "all patched tensors exact; source rewritten-shard hashes unchanged",
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    receipt = materialize(args.model_dir, args.patch, args.out_dir, args.receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
