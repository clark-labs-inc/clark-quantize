#!/usr/bin/env python3
"""Validate and summarize paired Qwen teacher and MLP activation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_source_rows(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest["content_sha256"]
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_digest(body) != claimed:
        raise ValueError("split manifest content hash mismatch")
    source_path = manifest_path.parent / manifest["source"]["path"]
    if sha256_file(source_path) != manifest["source"]["sha256"]:
        raise ValueError("calibration source SHA-256 mismatch")
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    return manifest, rows


def summarize(
    teacher_path: Path,
    activation_path: Path,
    split_manifest_path: Path,
    *,
    layer: int,
    positions_per_row: int,
) -> dict[str, Any]:
    if positions_per_row <= 0:
        raise ValueError("positions_per_row must be positive")
    manifest, source_rows = load_source_rows(split_manifest_path)
    teacher = torch.load(teacher_path, map_location="cpu")
    activation = torch.load(activation_path, map_location="cpu")
    if teacher.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError("unsupported teacher cache format")
    if activation.get("format") != "qwen_mlp_activation_cache_v1":
        raise ValueError("unsupported activation cache format")
    teacher_rows = teacher["rows"]
    activation_rows = activation["rows"]
    if len(teacher_rows) != len(source_rows) or len(activation_rows) != len(source_rows):
        raise ValueError("cache row count does not match frozen source")

    for index, source in enumerate(source_rows):
        expected_id = source["id"]
        expected_task = source["source"]
        teacher_row = teacher_rows[index]
        activation_row = activation_rows[index]
        if (
            teacher_row.get("index") != index
            or teacher_row.get("id") != expected_id
            or teacher_row.get("task") != expected_task
        ):
            raise ValueError(f"teacher row {index} does not match frozen source")
        if (
            activation_row.get("id") != expected_id
            or activation_row.get("task") != expected_task
        ):
            raise ValueError(f"activation row {index} does not match frozen source")
        generated = int(teacher_row["generated_tokens"])
        if teacher_row["topk_alignment_matches"] != generated:
            raise ValueError(f"teacher row {index} fails causal top-k alignment")
        if activation_row["positions"] != min(positions_per_row, generated):
            raise ValueError(f"activation row {index} position count is inconsistent")

    layer_payload = activation["layers"][layer]
    expected_positions = sum(int(row["positions"]) for row in activation_rows)
    x = layer_payload["x"]
    y = layer_payload["y"]
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != expected_positions:
        raise ValueError("activation tensor shape does not match row metadata")

    split_summary: dict[str, Any] = {}
    for name in ("train", "validation_selection", "independent_test"):
        indices = manifest[name]["indices"]
        split_summary[name] = {
            "rows": len(indices),
            "positions": sum(int(activation_rows[index]["positions"]) for index in indices),
            "ids": [activation_rows[index]["id"] for index in indices],
        }

    total_generated = sum(int(row["generated_tokens"]) for row in teacher_rows)
    total_greedy = sum(int(row["greedy_alignment_matches"]) for row in teacher_rows)
    total_topk = sum(int(row["topk_alignment_matches"]) for row in teacher_rows)
    result = {
        "schema": "qwen_cpu_cache_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "teacher_cache": str(teacher_path.resolve()),
        "teacher_cache_sha256": sha256_file(teacher_path),
        "activation_cache": str(activation_path.resolve()),
        "activation_cache_sha256": sha256_file(activation_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_content_sha256": manifest["content_sha256"],
        "rows": len(source_rows),
        "task_counts": manifest["source"]["task_counts"],
        "total_generated_tokens": total_generated,
        "greedy_alignment_matches": total_greedy,
        "greedy_alignment_fraction": total_greedy / total_generated,
        "topk_alignment_matches": total_topk,
        "topk_alignment_total": total_generated,
        "layer": layer,
        "positions_per_row": positions_per_row,
        "activation_shape": list(x.shape),
        "activation_dtype": str(x.dtype),
        "splits": split_summary,
    }
    result["report_content_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", required=True, type=Path)
    parser.add_argument("--activation-cache", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--positions-per-row", type=int, default=24)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(
        args.teacher_cache,
        args.activation_cache,
        args.split_manifest,
        layer=args.layer,
        positions_per_row=args.positions_per_row,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
