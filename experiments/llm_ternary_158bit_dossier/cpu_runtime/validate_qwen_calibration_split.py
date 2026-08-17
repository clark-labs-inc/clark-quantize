#!/usr/bin/env python3
"""Validate a frozen Qwen calibration source and train/select/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("calibration source is empty")
    return rows


def validate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest["content_sha256"]
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_digest(body) != claimed:
        raise ValueError("split manifest content hash mismatch")

    source_path = manifest_path.parent / manifest["source"]["path"]
    if sha256_file(source_path) != manifest["source"]["sha256"]:
        raise ValueError("calibration source SHA-256 mismatch")
    rows = load_jsonl(source_path)
    if len(rows) != manifest["source"]["rows"]:
        raise ValueError("calibration source row count mismatch")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise ValueError("every calibration row must have a nonempty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("calibration row ids are not unique")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError("every calibration row must have nonempty text")

    observed_task_counts: dict[str, int] = {}
    for row in rows:
        task = row.get("source")
        if not isinstance(task, str) or not task:
            raise ValueError("every calibration row must have a source task")
        observed_task_counts[task] = observed_task_counts.get(task, 0) + 1
    if observed_task_counts != manifest["source"]["task_counts"]:
        raise ValueError("calibration source task counts mismatch")

    split_names = ("train", "validation_selection", "independent_test")
    expected_per_task = {"train": 4, "validation_selection": 2, "independent_test": 2}
    all_indices: list[int] = []
    split_summary: dict[str, Any] = {}
    for split_name in split_names:
        split = manifest[split_name]
        indices = split["indices"]
        if len(indices) != split["rows"] or len(set(indices)) != len(indices):
            raise ValueError(f"{split_name} row count or uniqueness mismatch")
        if any(not isinstance(index, int) or not 0 <= index < len(rows) for index in indices):
            raise ValueError(f"{split_name} contains an invalid row index")
        selected_ids = [ids[index] for index in indices]
        if selected_ids != split["ids"]:
            raise ValueError(f"{split_name} ids do not match source indices")
        if split["maximum_positions_after_cache"] != 24 * len(indices):
            raise ValueError(f"{split_name} maximum cached-position count mismatch")
        task_counts = {task: 0 for task in observed_task_counts}
        for index in indices:
            task_counts[rows[index]["source"]] += 1
        if any(value != expected_per_task[split_name] for value in task_counts.values()):
            raise ValueError(f"{split_name} is not task balanced")
        all_indices.extend(indices)
        split_summary[split_name] = {
            "rows": len(indices),
            "task_counts": task_counts,
            "ids_sha256": canonical_digest(selected_ids),
        }
    if sorted(all_indices) != list(range(len(rows))):
        raise ValueError("splits are not disjoint and exhaustive")

    evaluation_strings: set[str] = set()
    evaluation_hashes: dict[str, str] = {}
    for key in ("cute_quick_manifest", "math_code_manifest"):
        eval_path = manifest_path.parent / manifest["external_evaluation_exclusion"][key]
        evaluation_hashes[key] = sha256_file(eval_path)
        evaluation_strings.update(
            normalize_text(value)
            for value in nested_strings(json.loads(eval_path.read_text()))
            if value.strip()
        )
    overlapping_ids = [
        row["id"] for row in rows if normalize_text(row["text"]) in evaluation_strings
    ]
    if overlapping_ids:
        raise ValueError(f"calibration prompts overlap frozen evaluation: {overlapping_ids}")

    result = {
        "schema": "qwen_calibration_split_validation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_content_sha256": claimed,
        "source": str(source_path.resolve()),
        "source_sha256": manifest["source"]["sha256"],
        "rows": len(rows),
        "task_counts": observed_task_counts,
        "splits": split_summary,
        "evaluation_manifest_sha256": evaluation_hashes,
        "exact_prompt_overlap_ids": overlapping_ids,
    }
    result["report_content_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
