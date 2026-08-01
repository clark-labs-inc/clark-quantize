#!/usr/bin/env python3
"""Materialize a pinned, held-out real-UI evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


RICO_REPO = "bevaya/RICO-Screen2Words"
RICO_REVISION = "fcd10c2cda2538ed71b1028984675fa29854283b"
RICO_LICENSE = "CC-BY-4.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache/clark-quantize/swin2sr-rico-heldout-v1",
    )
    parser.add_argument("--per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=(
            Path.home()
            / ".cache/clark-quantize/swin2sr-hf-100k-v1/manifest.jsonl"
        ),
    )
    return parser.parse_args()


def split_files(split: str) -> list[str]:
    files = HfApi().list_repo_files(
        RICO_REPO,
        repo_type="dataset",
        revision=RICO_REVISION,
    )
    matches = sorted(name for name in files if name.startswith(f"data/{split}-"))
    if not matches:
        raise ValueError(f"no {split} files found at pinned RICO revision")
    return matches


def reservoir(split: str, limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected: list[dict] = []
    seen = 0
    for filename in split_files(split):
        path = hf_hub_download(
            repo_id=RICO_REPO,
            repo_type="dataset",
            filename=filename,
            revision=RICO_REVISION,
        )
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["screenId", "image"]):
            for row_index, record in enumerate(batch.to_pylist()):
                image = record["image"]
                data = image["bytes"] if image else None
                if not data:
                    continue
                candidate = {
                    "screen_id": str(record["screenId"]),
                    "source_file": filename,
                    "source_row": row_offset + row_index,
                    "data": data,
                }
                seen += 1
                if len(selected) < limit:
                    selected.append(candidate)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < limit:
                        selected[replacement] = candidate
            row_offset += len(batch)
    if len(selected) != limit:
        raise ValueError(f"{split} provided {len(selected)} images, expected {limit}")
    return selected


def training_hashes(path: Path) -> set[str]:
    if not path.exists():
        raise ValueError(f"training manifest does not exist: {path}")
    return {
        json.loads(line)["sha256"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def write_image(root: Path, split: str, candidate: dict, excluded: set[str]) -> dict:
    data = candidate.pop("data")
    digest = hashlib.sha256(data).hexdigest()
    if digest in excluded:
        raise ValueError(f"held-out {split} image overlaps training hash {digest}")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image_format = (image.format or "PNG").lower()
    extension = {"jpeg": "jpg", "png": "png", "webp": "webp"}.get(image_format)
    if extension is None:
        raise ValueError(f"unsupported held-out image format: {image_format}")
    destination = root / split / f"{digest[:20]}.{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "source": "rico_screen2words_heldout",
        "split": split,
        "repo": RICO_REPO,
        "revision": RICO_REVISION,
        "license": RICO_LICENSE,
        "record_id": candidate["screen_id"],
        "source_file": candidate["source_file"],
        "source_row": candidate["source_row"],
        "sha256": digest,
        "width": width,
        "height": height,
        "local_path": str(destination.relative_to(root)),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    excluded = training_hashes(args.training_manifest)
    records = []
    for split_index, split in enumerate(("val", "test")):
        selected = reservoir(split, args.per_split, args.seed + split_index)
        records.extend(
            write_image(args.out, split, candidate, excluded)
            for candidate in selected
        )
    records.sort(key=lambda record: (record["split"], record["record_id"]))
    with (args.out / "manifest.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "images": len(records),
        "counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("val", "test")
        },
        "repo": RICO_REPO,
        "revision": RICO_REVISION,
        "license": RICO_LICENSE,
        "seed": args.seed,
        "training_manifest": str(args.training_manifest),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
