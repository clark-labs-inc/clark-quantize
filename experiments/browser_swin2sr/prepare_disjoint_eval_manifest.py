#!/usr/bin/env python3
"""Build a hash-disjoint, language-balanced evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-language", type=int, required=True)
    parser.add_argument("--expected-languages", type=int, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_disjoint_manifest(
    training_manifests: list[Path],
    candidate_manifests: list[Path],
    out: Path,
    per_language: int,
    expected_languages: int,
) -> dict:
    excluded = {
        record["sha256"]
        for manifest in training_manifests
        for record in read_records(manifest)
    }
    selected = []
    selected_hashes = set()
    language_counts = Counter()
    rejected_training = 0
    rejected_duplicate = 0
    for manifest in candidate_manifests:
        for record in read_records(manifest):
            digest = record.get("sha256")
            language = record.get("language")
            if not digest or not language:
                raise ValueError(f"{manifest} contains a record without sha256/language")
            if digest in excluded:
                rejected_training += 1
                continue
            if digest in selected_hashes:
                rejected_duplicate += 1
                continue
            if language_counts[language] >= per_language:
                continue
            source = manifest.parent / record["local_path"]
            if not source.is_file():
                raise ValueError(f"candidate image is missing: {source}")
            if file_sha256(source) != digest:
                raise ValueError(f"candidate image hash mismatch: {source}")
            normalized = dict(record)
            normalized["local_path"] = str(source.resolve())
            selected.append(normalized)
            selected_hashes.add(digest)
            language_counts[language] += 1
    if len(language_counts) != expected_languages:
        raise ValueError(
            f"expected {expected_languages} languages, found {len(language_counts)}"
        )
    incomplete = {
        language: count
        for language, count in language_counts.items()
        if count != per_language
    }
    if incomplete:
        raise ValueError(f"incomplete language counts: {incomplete}")
    selected.sort(key=lambda row: (row["language"], row["record_id"], row["sha256"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in selected
        )
    )
    receipt = {
        "format": "backdropkit-swin2sr-disjoint-eval-v1",
        "manifest": str(out),
        "manifest_sha256": file_sha256(out),
        "samples": len(selected),
        "languages": len(language_counts),
        "samples_per_language": per_language,
        "training_manifests": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in training_manifests
        ],
        "candidate_manifests": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in candidate_manifests
        ],
        "rejected_training_overlap": rejected_training,
        "rejected_candidate_duplicate": rejected_duplicate,
    }
    out.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    args = parse_args()
    receipt = build_disjoint_manifest(
        args.training_manifest,
        args.candidate_manifest,
        args.out,
        args.per_language,
        args.expected_languages,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
