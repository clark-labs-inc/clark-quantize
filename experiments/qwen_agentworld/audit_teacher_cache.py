"""Audit Qwen teacher-cache shape for QAT sweep design.

This script intentionally does not load the model. It inspects row counts,
supervised positions, task/source balance, and whether a sequential one-row
trainer will actually fill the requested token budget per optimizer step.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_lengths(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def task_name(row: dict[str, Any]) -> str:
    task = row.get("task")
    if task:
        return str(task)
    source = str(row.get("source") or "unknown")
    return source.split(":", 1)[0]


def audit_cache(
    path: Path,
    *,
    max_train_tokens: int,
    arm_steps: int,
    shuffle_rows: bool,
    row_seed: int,
) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu")
    rows = list(cache.get("rows") or [])
    selected = [row for row in rows if int(row.get("tokens") or 0) <= max_train_tokens]
    if shuffle_rows:
        random.Random(row_seed).shuffle(selected)
    touched = selected[: min(arm_steps, len(selected))]
    eval_tail = selected[min(arm_steps, len(selected)) :]

    token_values = [int(row.get("tokens") or 0) for row in rows]
    keep_values = [int(row.get("keep") or row.get("tokens") or 0) for row in rows]
    selected_tokens = [int(row.get("tokens") or 0) for row in selected]
    selected_keep = [int(row.get("keep") or row.get("tokens") or 0) for row in selected]

    task_counts: Counter[str] = Counter()
    task_tokens: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        name = task_name(row)
        task_counts[name] += 1
        task_tokens[name] += int(row.get("tokens") or 0)

    selected_task_counts: Counter[str] = Counter()
    eval_tail_task_counts: Counter[str] = Counter()
    for row in selected:
        selected_task_counts[task_name(row)] += 1
    for row in eval_tail:
        eval_tail_task_counts[task_name(row)] += 1

    avg_selected_tokens = mean(selected_tokens) if selected_tokens else None
    return {
        "cache_path": str(path),
        "format": cache.get("format"),
        "row_count": len(rows),
        "token_lengths": summarize_lengths(token_values),
        "keep_lengths": summarize_lengths(keep_values),
        "task_counts": dict(task_counts),
        "task_tokens": dict(task_tokens),
        "trainer_assumption": {
            "one_cache_row_per_optimizer_step": True,
            "does_not_pack_rows_to_max_train_tokens": True,
            "max_train_tokens": max_train_tokens,
            "shuffle_rows": shuffle_rows,
            "row_seed": row_seed,
            "selected_rows": len(selected),
            "selected_token_lengths": summarize_lengths(selected_tokens),
            "selected_keep_lengths": summarize_lengths(selected_keep),
            "avg_token_fill_ratio": (
                avg_selected_tokens / max_train_tokens
                if avg_selected_tokens is not None and max_train_tokens > 0
                else None
            ),
        },
        "arm_coverage": {
            "arm_steps": arm_steps,
            "rows_touched_by_one_arm": len(touched),
            "tokens_seen_by_one_arm": sum(int(row.get("tokens") or 0) for row in touched),
            "rows_unreached_after_one_arm": len(eval_tail),
            "eval_tail_tokens": sum(int(row.get("tokens") or 0) for row in eval_tail),
            "selected_task_counts": dict(selected_task_counts),
            "eval_tail_task_counts": dict(eval_tail_task_counts),
            "tail_is_clean_for_same_order_arm": arm_steps < len(selected),
        },
        "multimodal_markers": {
            "has_any_image_field": any(
                any(key in row for key in ("image", "images", "image_path", "image_paths", "pixel_values"))
                for row in rows
            ),
            "has_any_video_field": any(
                any(key in row for key in ("video", "videos", "video_path", "video_paths", "frames"))
                for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--max-train-tokens", type=int, default=8192)
    parser.add_argument("--arm-steps", type=int, default=2000)
    parser.add_argument("--shuffle-rows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--row-seed", type=int, default=17)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_cache(
                args.cache,
                max_train_tokens=args.max_train_tokens,
                arm_steps=args.arm_steps,
                shuffle_rows=args.shuffle_rows,
                row_seed=args.row_seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
