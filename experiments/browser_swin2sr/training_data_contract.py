"""Immutable training-data identities for Swin2SR studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprints(roots: list[Path]) -> list[dict]:
    fingerprints = []
    for root in roots:
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.jsonl"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise ValueError(
                f"training root lacks summary.json or manifest.jsonl: {root}"
            )
        summary = json.loads(summary_path.read_text())
        fingerprints.append({
            "root": str(root),
            "summary": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "images": summary.get("images"),
            "source": summary.get("source"),
            "source_revision": summary.get("source_revision"),
        })
    return fingerprints


def validation_receipts(
    paths: list[Path],
    datasets: list[dict],
) -> list[dict]:
    roots = {
        Path(dataset["root"]).resolve(): dataset
        for dataset in datasets
    }
    receipts = []
    for path in paths:
        receipt = json.loads(path.read_text())
        if (
            receipt.get("format")
            != "backdropkit-swin2sr-training-corpus-receipt-v1"
        ):
            raise ValueError(f"unsupported training-data receipt: {path}")
        root = Path(receipt["root"]).resolve()
        dataset = roots.get(root)
        if dataset is None:
            raise ValueError(f"receipt does not describe a training root: {path}")
        if (
            receipt.get("summary_sha256") != dataset["summary_sha256"]
            or receipt.get("manifest_sha256") != dataset["manifest_sha256"]
        ):
            raise ValueError(f"training-data receipt hash mismatch: {path}")
        receipts.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "root": receipt["root"],
            "summary_sha256": receipt["summary_sha256"],
            "manifest_sha256": receipt["manifest_sha256"],
            "images": receipt["images"],
            "languages": receipt["languages"],
            "heldout_hash_overlap": receipt["heldout_hash_overlap"],
        })
    return receipts
