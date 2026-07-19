"""Merge Qwen teacher-cache shards or partials into one trainer cache."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


SUPPORTED_FORMATS = {
    "qwen_agentworld_teacher_behavior_v1",
    "qwen_onpolicy_survival_cache_v1",
}


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return list(dict.fromkeys(paths))


def row_tensor_shape(row: dict[str, Any]) -> tuple[int, int]:
    values = row.get("top_values")
    if not isinstance(values, torch.Tensor):
        values = row.get("teacher_top_logprobs")
    if not isinstance(values, torch.Tensor):
        raise ValueError("row missing top_values or teacher_top_logprobs")
    if values.ndim != 2:
        raise ValueError(f"teacher values must be rank-2, got {tuple(values.shape)}")
    return int(values.shape[0]), int(values.shape[1])


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    positions, top_k = row_tensor_shape(row)
    return {
        "task": row.get("task"),
        "id": row.get("id"),
        "tokens": int(row.get("tokens") or positions),
        "positions": positions,
        "top_k": top_k,
        "meta": row.get("source_meta", {}),
    }


def load_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"cache is not a dict: {path}")
    if cache.get("format") not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported cache format in {path}: {cache.get('format')!r}")
    rows = cache.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"cache rows is not a list: {path}")
    return cache


def compatible_header(base: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    for key in (
        "format",
        "model_dir",
        "max_length",
        "top_k",
        "teacher_values_kind",
        "position_alignment",
    ):
        left = base.get(key)
        right = candidate.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"incompatible {key} in {path}: {right!r} != {left!r}")


def merge(paths: list[Path], *, keep_duplicates: bool) -> dict[str, Any]:
    if not paths:
        raise ValueError("no input cache paths")
    caches = [load_cache(path) for path in paths]
    base = {key: value for key, value in caches[0].items() if key != "rows"}
    for path, cache in zip(paths[1:], caches[1:], strict=True):
        compatible_header(base, cache, path)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids = 0
    task_counts: Counter[str] = Counter()
    for cache in caches:
        for row in cache["rows"]:
            row_id = str(row.get("id") or "")
            if row_id and row_id in seen_ids:
                duplicate_ids += 1
                if not keep_duplicates:
                    continue
            if row_id:
                seen_ids.add(row_id)
            row = dict(row)
            row["index"] = len(rows)
            rows.append(row)
            task_counts[str(row.get("task") or "unknown")] += 1

    base["rows"] = rows
    base["failures"] = sum(int(cache.get("failures") or 0) for cache in caches)
    base["merged_from"] = [str(path) for path in paths]
    base["merge_duplicate_ids"] = duplicate_ids
    base["merge_task_counts"] = dict(task_counts)
    return base


def write_meta(cache: dict[str, Any], out: Path) -> Path:
    meta_path = out.with_suffix(out.suffix + ".json")
    meta = {
        key: value
        for key, value in cache.items()
        if key != "rows"
    }
    meta["rows"] = [row_meta(row) for row in cache["rows"]]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="input .pt/.pt.partial paths or globs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep-duplicates", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    paths = expand_patterns(args.patterns)
    if not paths:
        raise SystemExit("no cache files matched")
    cache = merge(paths, keep_duplicates=args.keep_duplicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    meta_path = write_meta(cache, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "meta": str(meta_path),
                "inputs": len(paths),
                "rows": len(cache["rows"]),
                "failures": cache.get("failures", 0),
                "duplicate_ids": cache.get("merge_duplicate_ids", 0),
                "tasks": cache.get("merge_task_counts", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
