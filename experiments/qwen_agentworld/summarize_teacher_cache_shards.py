"""Summarize sharded Qwen teacher-cache metadata without loading tensor shards."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return list(dict.fromkeys(paths))


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    frac = index - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def load_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata: {path}: {exc}") from exc


def shard_summary(path: Path) -> dict[str, Any]:
    meta = load_meta(path)
    pt_path = Path(str(path).removesuffix(".json"))
    rows = meta.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"metadata rows is not a list: {path}")

    modalities: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    reasoning: Counter[str] = Counter()
    token_values: list[int] = []
    position_values: list[int] = []
    topk_values: list[int] = []
    missing: Counter[str] = Counter()
    duplicate_ids = 0
    seen_ids: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            missing["row_dict"] += 1
            continue
        row_meta = row.get("meta") or {}
        modality = row_meta.get("modality") or "unknown"
        modalities[str(modality)] += 1
        tasks[str(row.get("task") or "unknown")] += 1
        reasoning_key = f"reasoning={bool(row_meta.get('reasoning', False))},think={bool(row_meta.get('think_tokens', False))}"
        reasoning[reasoning_key] += 1
        for key, dest in (("tokens", token_values), ("positions", position_values), ("top_k", topk_values)):
            value = row.get(key)
            if isinstance(value, int):
                dest.append(value)
            else:
                missing[key] += 1
        row_id = row.get("id")
        if not row_id:
            missing["id"] += 1
        elif row_id in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(str(row_id))

    return {
        "meta_path": str(path),
        "pt_path": str(pt_path),
        "pt_exists": pt_path.exists(),
        "pt_bytes": pt_path.stat().st_size if pt_path.exists() else 0,
        "rows": len(rows),
        "failures": int(meta.get("failures") or 0),
        "format": meta.get("format"),
        "builder": meta.get("builder"),
        "max_length": meta.get("max_length"),
        "last_n": meta.get("last_n"),
        "top_k": meta.get("top_k"),
        "modalities": dict(modalities),
        "tasks": dict(tasks),
        "reasoning": dict(reasoning),
        "tokens": {
            "min": min(token_values) if token_values else None,
            "mean": mean(token_values) if token_values else None,
            "p50": percentile(token_values, 0.50),
            "p90": percentile(token_values, 0.90),
            "max": max(token_values) if token_values else None,
        },
        "positions": {
            "min": min(position_values) if position_values else None,
            "mean": mean(position_values) if position_values else None,
            "max": max(position_values) if position_values else None,
        },
        "topk_values": sorted(set(topk_values)),
        "missing": dict(missing),
        "duplicate_ids": duplicate_ids,
    }


def combine(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = sum(item["rows"] for item in summaries)
    total_failures = sum(item["failures"] for item in summaries)
    total_bytes = sum(item["pt_bytes"] for item in summaries)
    modalities: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    reasoning: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    bad_shards: list[str] = []
    formats = Counter()
    builders = Counter()
    topk_values = set()

    for item in summaries:
        modalities.update(item["modalities"])
        tasks.update(item["tasks"])
        reasoning.update(item["reasoning"])
        missing.update(item["missing"])
        formats[str(item.get("format"))] += 1
        builders[str(item.get("builder"))] += 1
        topk_values.update(item["topk_values"])
        if not item["pt_exists"] or item["rows"] <= 0 or item["format"] != "qwen_agentworld_teacher_behavior_v1":
            bad_shards.append(item["meta_path"])

    rows_per_gb = total_rows / (total_bytes / 1_000_000_000) if total_bytes else None
    projected_gb_100k = 100_000 / rows_per_gb if rows_per_gb else None
    return {
        "shards": len(summaries),
        "rows": total_rows,
        "failures": total_failures,
        "bytes": total_bytes,
        "gb": total_bytes / 1_000_000_000,
        "rows_per_gb": rows_per_gb,
        "projected_gb_100k": projected_gb_100k,
        "modalities": dict(modalities),
        "tasks": dict(tasks),
        "reasoning": dict(reasoning),
        "formats": dict(formats),
        "builders": dict(builders),
        "topk_values": sorted(topk_values),
        "missing": dict(missing),
        "bad_shards": bad_shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="metadata JSON globs, for example '*shard*.pt.json'")
    parser.add_argument("--out", type=Path, help="optional JSON report path")
    parser.add_argument("--print-shards", action="store_true")
    args = parser.parse_args()

    paths = expand_patterns(args.patterns)
    if not paths:
        raise SystemExit("no metadata files matched")
    summaries = [shard_summary(path) for path in paths]
    report = {
        "patterns": args.patterns,
        "summary": combine(summaries),
        "shards": summaries,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    compact = report["summary"] if not args.print_shards else report
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
