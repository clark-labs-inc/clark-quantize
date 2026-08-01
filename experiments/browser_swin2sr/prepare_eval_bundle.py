#!/usr/bin/env python3
"""Materialize immutable low/high PNG pairs for cross-runtime Swin2SR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_stratified import (
    deterministic_crop,
    load_records,
    select_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bundle(
    manifests: list[Path],
    out: Path,
    crop_size: int,
    max_samples: int | None,
    seed: int,
) -> dict:
    if crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    records = select_records(load_records(manifests), max_samples, seed)
    low_dir = out / "low"
    high_dir = out / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    seen = set()
    for record in records:
        source_sha256 = record.get("sha256") or file_sha256(Path(record["path"]))
        if source_sha256 in seen:
            raise ValueError(f"duplicate source sha256: {source_sha256}")
        seen.add(source_sha256)
        low, high = deterministic_crop(record, crop_size)
        low_path = low_dir / f"{source_sha256}.png"
        high_path = high_dir / f"{source_sha256}.png"
        low.save(low_path, format="PNG", optimize=False)
        high.save(high_path, format="PNG", optimize=False)
        output_records.append({
            "source_sha256": source_sha256,
            "source": record.get("source", "unknown"),
            "split": record.get("split", "unknown"),
            "language": record.get("language"),
            "low_path": str(low_path.relative_to(out)),
            "low_sha256": file_sha256(low_path),
            "high_path": str(high_path.relative_to(out)),
            "high_sha256": file_sha256(high_path),
        })
    manifest_path = out / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in output_records)
    )
    metadata = {
        "format": "backdropkit-swin2sr-eval-v1",
        "crop_size": crop_size,
        "low_size": crop_size // 2,
        "seed": seed,
        "samples": len(output_records),
        "source_manifests": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in manifests
        ],
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (out / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = prepare_bundle(
        args.manifest,
        args.out,
        args.crop_size,
        args.max_samples,
        args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
