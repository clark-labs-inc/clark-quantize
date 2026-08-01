"""Provenance contracts for development-only full-surface selection."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from training_data_contract import dataset_fingerprints, validation_receipts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_eval_contract(args, q4_reference: dict) -> None:
    bundle = json.loads(args.bundle.read_text())
    expected_hash = q4_reference.get("bundle_manifest_sha256")
    if bundle.get("manifest_sha256") != expected_hash:
        raise ValueError("evaluation bundle does not match the q4 reference")
    if len(q4_reference.get("samples", [])) != int(bundle["samples"]):
        raise ValueError("q4 reference and evaluation bundle sample counts differ")
    if args.eval_device != q4_reference.get("device"):
        raise ValueError(
            "evaluation device does not match the q4 reference: "
            f"{args.eval_device!r} != {q4_reference.get('device')!r}"
        )


def study_manifest(args, base, source_files: tuple[str, ...]) -> dict:
    source_root = Path(__file__).resolve().parent
    datasets = dataset_fingerprints(args.data)
    return {
        "format": "backdropkit-swin2sr-full-surface-comparison-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "recovered_checkpoint": str(args.recovered_checkpoint),
        "recovered_checkpoint_sha256": file_sha256(args.recovered_checkpoint),
        "base_quantization": {
            "method": base.method,
            "group_size": base.group_size,
            "rotation": base.rotation,
        },
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "training_data": datasets,
        "training_data_receipts": validation_receipts(
            args.data_receipt,
            datasets,
        ),
        "source_files": {
            name: file_sha256(source_root / name)
            for name in source_files
        },
        "lanes": list(args.only or ("fp32", "recovered_core")),
        "config": {
            key: (
                [str(item) for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "ranking_policy": [
            "worst_source_psnr_delta",
            "overall_psnr_delta",
            "edge_psnr_delta",
            "teacher_psnr_delta",
        ],
    }


def selection_provenance(
    args,
    q4_reference: dict,
    manifest_path: Path,
) -> dict:
    bundle = json.loads(args.bundle.read_text())
    return {
        "selection_source": "development",
        "study_manifest": str(manifest_path),
        "study_manifest_sha256": file_sha256(manifest_path),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "q4_reference_state_sha256": q4_reference["state_sha256"],
    }
