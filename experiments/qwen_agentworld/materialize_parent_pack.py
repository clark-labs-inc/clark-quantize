"""Materialize a Clark Air parent-pack manifest as a normal HF checkpoint.

This is an evaluation bridge, not a storage format. The packed artifact remains
the source of truth for bytes-on-disk; this script reconstructs those packed
groups into ordinary safetensors shards so servers such as vLLM can evaluate the
behavior of the compressed weights without a custom low-bit runtime loader.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify_parent_pack_manifest import (
    apply_weight_patch,
    load_weight_patch,
    reconstruct_group,
)


DEFAULT_EXCLUDE = {
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.pt",
    "*.pth",
    "__pycache__",
}


def should_copy(path: Path, excludes: set[str]) -> bool:
    name = path.name
    return not any(fnmatch.fnmatch(name, pattern) for pattern in excludes)


def copy_model_sidecars(model_dir: Path, out_dir: Path, excludes: set[str]) -> list[str]:
    copied: list[str] = []
    for src in sorted(model_dir.iterdir()):
        if not should_copy(src, excludes):
            continue
        dst = out_dir / src.name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(*excludes),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(src, dst)
        copied.append(src.name)
    return copied


def load_index(model_dir: Path) -> dict[str, Any]:
    candidates = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one *.safetensors.index.json in {model_dir}, "
            f"found {[p.name for p in candidates]}"
        )
    return json.loads(candidates[0].read_text())


def groups_by_tensor(manifest: dict[str, Any]) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for parent in manifest["packed"]:
        tensor = str(parent["tensor"])
        for group in parent["groups"]:
            rows[tensor].append((parent, group))
    return dict(rows)


def artifact_cache_get(cache: dict[str, Any], artifact_path: str) -> Any:
    artifact = cache.get(artifact_path)
    if artifact is None:
        artifact = np.load(artifact_path)
        cache[artifact_path] = artifact
    return artifact


def apply_groups_to_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    groups: list[tuple[dict[str, Any], dict[str, Any]]],
    artifact_cache: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    out = tensor.contiguous().clone()
    rows: list[dict[str, Any]] = []
    for parent, group in groups:
        artifact = artifact_cache_get(artifact_cache, str(parent["artifact"]))
        reconstructed = reconstruct_group(artifact, group).to(dtype=out.dtype)
        if "expert" in group:
            expert = int(group["expert"])
            out[expert].copy_(reconstructed)
        else:
            if tuple(reconstructed.shape) != tuple(out.shape):
                raise ValueError(
                    f"group {group['group']} reconstructed shape {tuple(reconstructed.shape)} "
                    f"does not match tensor {tensor_name} shape {tuple(out.shape)}"
                )
            out.copy_(reconstructed)
        rows.append(
            {
                "group": group.get("group"),
                "rate": group.get("rate"),
                "expert": group.get("expert"),
                "artifact": parent.get("artifact"),
                "relative_rmse": group.get("relative_rmse"),
            }
        )
    return out, rows


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--weight-patch", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional sidecar filename pattern to skip while copying.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    weight_patch_path = args.weight_patch
    if weight_patch_path is None and manifest.get("weight_patch"):
        weight_patch_path = Path(str(manifest["weight_patch"]))
    weight_patch = load_weight_patch(weight_patch_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    excludes = set(DEFAULT_EXCLUDE) | set(args.exclude)
    copied_sidecars = copy_model_sidecars(args.model_dir, args.out_dir, excludes)

    index = load_index(args.model_dir)
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard in weight_map.items():
        by_shard[str(shard)].append(tensor_name)

    packed_groups = groups_by_tensor(manifest)
    artifact_cache: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    patched_count = 0
    materialized_group_count = 0
    total_size = 0

    try:
        for shard, tensor_names in sorted(by_shard.items()):
            shard_tensors: dict[str, torch.Tensor] = {}
            with safe_open(args.model_dir / shard, framework="pt", device="cpu") as source:
                for tensor_name in sorted(tensor_names):
                    original = source.get_tensor(tensor_name)
                    tensor, patch_key = apply_weight_patch(original, tensor_name, weight_patch)
                    if patch_key is not None:
                        patched_count += 1
                    materialized_rows: list[dict[str, Any]] = []
                    if tensor_name in packed_groups:
                        tensor, materialized_rows = apply_groups_to_tensor(
                            tensor,
                            tensor_name,
                            packed_groups[tensor_name],
                            artifact_cache,
                        )
                        materialized_group_count += len(materialized_rows)
                    tensor = tensor.contiguous()
                    shard_tensors[tensor_name] = tensor
                    total_size += tensor_nbytes(tensor)
                    report_rows.append(
                        {
                            "tensor": tensor_name,
                            "shard": shard,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "bytes": tensor_nbytes(tensor),
                            "weight_patch_key": patch_key,
                            "materialized_groups": materialized_rows,
                        }
                    )
            save_file(shard_tensors, args.out_dir / shard, metadata={"format": "pt"})
            print(json.dumps({"saved_shard": shard, "tensors": len(shard_tensors)}))
    finally:
        for artifact in artifact_cache.values():
            artifact.close()

    out_index = {
        "metadata": dict(index.get("metadata") or {}),
        "weight_map": weight_map,
    }
    out_index["metadata"]["total_size"] = total_size
    index_name = sorted(args.model_dir.glob("*.safetensors.index.json"))[0].name
    (args.out_dir / index_name).write_text(json.dumps(out_index, indent=2) + "\n")

    summary = manifest.get("summary") or {}
    report = {
        "manifest": str(args.manifest),
        "model_dir": str(args.model_dir),
        "out_dir": str(args.out_dir),
        "weight_patch": str(weight_patch_path) if weight_patch_path else None,
        "copied_sidecars": copied_sidecars,
        "shards": len(by_shard),
        "tensors": len(weight_map),
        "patched_tensors": patched_count,
        "materialized_groups": materialized_group_count,
        "total_size": total_size,
        "source_artifact_bytes": summary.get("artifact_bytes", manifest.get("artifact_bytes")),
        "source_bytes": summary.get("source_bytes", manifest.get("source_bytes")),
        "source_compression": summary.get("compression", manifest.get("compression")),
        "source_by_rate": summary.get("by_rate"),
        "rows": report_rows,
    }
    report_path = args.report or (args.out_dir / "clark_air_materialize_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "report": str(report_path),
                "tensors": len(weight_map),
                "patched_tensors": patched_count,
                "materialized_groups": materialized_group_count,
                "total_size": total_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
