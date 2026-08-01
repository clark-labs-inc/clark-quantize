#!/usr/bin/env python3
"""Pin and index rights-approved screen recordings without fetching video blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "markov-ai/computer-use-large"
REVISION = "b50aeccec6d24a56ed4f8fbb9f5b2a16846b46a9"
LICENSE = "cc-by-4.0"
METADATA_FILES = (
    "data/autocad/metadata.jsonl",
    "data/autocad_2/metadata.jsonl",
    "data/blender/metadata.jsonl",
    "data/blender_2/metadata.jsonl",
    "data/excel/metadata.jsonl",
    "data/photoshop/metadata.jsonl",
    "data/photoshop_2/metadata.jsonl",
    "data/salesforce/metadata.jsonl",
    "data/vscode/metadata.jsonl",
)
SPLIT_BY_CATEGORY = {
    "salesforce": "validation",
    "vscode": "test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_priority(repo_path: str) -> str:
    return hashlib.sha256(f"{REVISION}:{repo_path}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-hours", type=float, default=1000.0)
    args = parser.parse_args()
    if args.training_hours <= 0:
        raise ValueError("training hours must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    source_files = []
    quarantined_rows = []
    for metadata_name in METADATA_FILES:
        local = Path(hf_hub_download(
            REPO_ID,
            metadata_name,
            repo_type="dataset",
            revision=REVISION,
        ))
        source_files.append({
            "path": metadata_name,
            "bytes": local.stat().st_size,
            "sha256": sha256(local),
        })
        source_directory = str(Path(metadata_name).parent)
        with local.open() as handle:
            for line_number, line in enumerate(handle, 1):
                source = json.loads(line)
                missing_core = [
                    key for key in ("file_name", "category")
                    if key not in source
                ]
                if missing_core:
                    quarantined_rows.append({
                        "source_metadata": metadata_name,
                        "source_line": line_number,
                        "missing": missing_core,
                    })
                    continue
                category = str(source["category"])
                repo_path = f"{source_directory}/{source['file_name']}"
                complete = "trimmed_duration" in source and "num_segments" in source
                rows.append({
                    "repo_id": REPO_ID,
                    "revision": REVISION,
                    "license": LICENSE,
                    "repo_path": repo_path,
                    "category": category,
                    "duration_seconds": (
                        float(source["trimmed_duration"])
                        if complete else None
                    ),
                    "segments": int(source["num_segments"]) if complete else None,
                    "metadata_complete": complete,
                    "split": SPLIT_BY_CATEGORY.get(category, "train"),
                    "source_metadata": metadata_name,
                    "source_line": line_number,
                    "priority": stable_priority(repo_path),
                })

    train = sorted(
        (
            row for row in rows
            if row["split"] == "train" and row["metadata_complete"]
        ),
        key=lambda row: (row["category"], row["priority"]),
    )
    categories = sorted({row["category"] for row in train})
    target_per_category = args.training_hours * 3600 / len(categories)
    selected = set()
    selected_seconds = defaultdict(float)
    for category in categories:
        candidates = [row for row in train if row["category"] == category]
        for row in candidates:
            if selected_seconds[category] >= target_per_category:
                break
            selected.add(row["repo_path"])
            selected_seconds[category] += row["duration_seconds"]

    index_path = args.out / "markov-screen-recording-index.jsonl"
    with index_path.open("w") as handle:
        for row in sorted(rows, key=lambda item: item["repo_path"]):
            handle.write(json.dumps({
                **row,
                "selected_for_initial_ingest": row["repo_path"] in selected,
            }, sort_keys=True) + "\n")

    counts = Counter(row["split"] for row in rows)
    seconds = Counter()
    for row in rows:
        if row["metadata_complete"]:
            seconds[row["split"]] += row["duration_seconds"]
    receipt = {
        "format": "backdropkit-video-source-index-v1",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "metadata_files": source_files,
        "index": {
            "path": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256(index_path),
        },
        "rows": len(rows),
        "rows_quarantined_for_missing_identity": len(quarantined_rows),
        "quarantined_identity_rows": quarantined_rows,
        "rows_with_complete_duration": sum(
            row["metadata_complete"] for row in rows
        ),
        "rows_quarantined_for_incomplete_metadata": sum(
            not row["metadata_complete"] for row in rows
        ),
        "split_rows": dict(sorted(counts.items())),
        "split_hours": {
            name: value / 3600
            for name, value in sorted(seconds.items())
        },
        "initial_ingest": {
            "target_hours": args.training_hours,
            "selected_videos": len(selected),
            "selected_hours": sum(selected_seconds.values()) / 3600,
            "selected_hours_by_category": {
                name: value / 3600
                for name, value in sorted(selected_seconds.items())
            },
            "selection": "stable SHA-256 priority, equal target hours per training category",
        },
    }
    receipt_path = args.out / "markov-screen-recording-index-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
