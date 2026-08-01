"""Content identity for exact Swin2SR recovery sampler resumes."""

from __future__ import annotations

from pathlib import Path

from training_data_contract import dataset_fingerprints


def recovery_corpus_identity(roots: list[Path]) -> dict:
    datasets = dataset_fingerprints(roots)
    return {
        "format": "backdropkit-swin2sr-recovery-corpus-v1",
        "datasets": [
            {
                "summary_sha256": dataset["summary_sha256"],
                "manifest_sha256": dataset["manifest_sha256"],
                "images": dataset["images"],
            }
            for dataset in datasets
        ],
    }
