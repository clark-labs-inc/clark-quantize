#!/usr/bin/env python3
"""Validate and seal a generated multilingual Swin2SR training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCE_FILES = (
    "generate_synthetic_text_data.py",
    "text_renderer.py",
    "validate_training_corpus.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON"
                ) from error
    return records


def heldout_hashes(bundle_paths: list[Path]) -> tuple[set[str], list[dict]]:
    hashes = set()
    receipts = []
    for bundle_path in bundle_paths:
        bundle = json.loads(bundle_path.read_text())
        manifest = Path(bundle["manifest"])
        if file_sha256(manifest) != bundle["manifest_sha256"]:
            raise ValueError(f"held-out bundle manifest hash mismatch: {bundle_path}")
        records = jsonl(manifest)
        source_hashes = {
            record.get("source_sha256")
            for record in records
        }
        if None in source_hashes:
            raise ValueError(f"held-out bundle has a missing source hash: {bundle_path}")
        hashes.update(source_hashes)
        receipts.append({
            "bundle": str(bundle_path),
            "bundle_sha256": file_sha256(bundle_path),
            "manifest": str(manifest),
            "manifest_sha256": bundle["manifest_sha256"],
            "samples": len(records),
        })
    return hashes, receipts


def validate(
    root: Path,
    expected_images: int,
    expected_languages: int,
    expected_per_language: int,
    exclude_bundles: list[Path],
) -> dict:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError("training corpus is missing summary.json or manifest.jsonl")
    summary = json.loads(summary_path.read_text())
    records = jsonl(manifest_path)
    expectations = {
        "images": expected_images,
        "languages": expected_languages,
        "images_per_language": expected_per_language,
    }
    for key, expected in expectations.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError(
                f"training summary {key}={summary.get(key)!r}, expected {expected}"
            )
    if len(records) != expected_images:
        raise ValueError(
            f"training manifest has {len(records)} records, expected {expected_images}"
        )
    if summary.get("text_renderer") != "pango-view":
        raise ValueError("training corpus did not use the glyph-safe Pango renderer")
    if summary.get("reject_unknown_glyphs") is not True:
        raise ValueError("training corpus did not reject unknown glyphs")
    allowed_resolutions = {
        tuple(resolution)
        for resolution in summary.get("resolutions", [])
    }
    languages = Counter()
    scripts = Counter()
    seen_hashes = set()
    seen_paths = set()
    total_bytes = 0
    for index, record in enumerate(records):
        language = record.get("language")
        digest = record.get("sha256")
        local_path = record.get("local_path")
        if not language or not digest or not local_path:
            raise ValueError(f"training record {index} is missing identity fields")
        image = root / local_path
        if not image.is_file():
            raise ValueError(f"training image is missing: {image}")
        if local_path in seen_paths:
            raise ValueError(f"duplicate training path: {local_path}")
        if digest in seen_hashes:
            raise ValueError(f"duplicate training image hash: {digest}")
        if file_sha256(image) != digest:
            raise ValueError(f"training image hash mismatch: {image}")
        resolution = (int(record["width"]), int(record["height"]))
        if resolution not in allowed_resolutions:
            raise ValueError(f"unexpected training resolution: {resolution}")
        if record.get("hf_source") != summary.get("source"):
            raise ValueError(f"training source mismatch in record {index}")
        if record.get("hf_revision") != summary.get("source_revision"):
            raise ValueError(f"training revision mismatch in record {index}")
        languages[language] += 1
        scripts[language.rsplit("_", 1)[-1]] += 1
        seen_paths.add(local_path)
        seen_hashes.add(digest)
        total_bytes += image.stat().st_size
    if len(languages) != expected_languages:
        raise ValueError(
            f"training corpus has {len(languages)} languages, "
            f"expected {expected_languages}"
        )
    unequal = {
        language: count
        for language, count in languages.items()
        if count != expected_per_language
    }
    if unequal:
        raise ValueError(f"training languages are not equally represented: {unequal}")
    excluded, heldout = heldout_hashes(exclude_bundles)
    overlap = sorted(seen_hashes & excluded)
    if overlap:
        raise ValueError(
            f"training corpus overlaps held-out data by {len(overlap)} hashes"
        )
    font_assets = summary.get("font_assets", {})
    font_path = Path(font_assets.get("font", ""))
    font_hash = font_assets.get("font_sha256")
    config_path = Path(font_assets.get("config", ""))
    if (
        not font_path.is_file()
        or not font_hash
        or file_sha256(font_path) != font_hash
    ):
        raise ValueError("pinned font asset is missing or changed")
    if not config_path.is_file():
        raise ValueError("pinned font configuration is missing")
    source_root = Path(__file__).resolve().parent
    return {
        "format": "backdropkit-swin2sr-training-corpus-receipt-v1",
        "root": str(root),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "images": len(records),
        "image_bytes": total_bytes,
        "unique_image_hashes": len(seen_hashes),
        "languages": len(languages),
        "images_per_language_min": min(languages.values()),
        "images_per_language_max": max(languages.values()),
        "scripts": len(scripts),
        "script_counts": dict(sorted(scripts.items())),
        "heldout_bundles": heldout,
        "heldout_hash_overlap": 0,
        "font": str(font_path),
        "font_sha256": font_hash,
        "font_config": str(config_path),
        "font_config_sha256": file_sha256(config_path),
        "source_files": {
            name: file_sha256(source_root / name)
            for name in SOURCE_FILES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--expected-languages", type=int, required=True)
    parser.add_argument("--expected-per-language", type=int, required=True)
    parser.add_argument("--exclude-bundle", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.root,
        args.expected_images,
        args.expected_languages,
        args.expected_per_language,
        args.exclude_bundle,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
