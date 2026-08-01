#!/usr/bin/env python3
"""Seal the M5 batch-capacity study before any capacity output exists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
PROTOCOL = ROOT / "protocol.json"
SOURCES = (
    "probe_batch.py",
    "run_batch_study.py",
    "test_probe_batch.py",
    "test_run_batch_study.py",
)
PARENT = REPOSITORY / (
    "experiments/browser_swin2sr/surface_group_recovery_20260731/refined/"
    "run/checkpoint-final.pt"
)
DATA = (
    Path("/Users/stan/.cache/clark-quantize/swin2sr-hf-100k-v1"),
    Path("/Users/stan/.cache/clark-quantize/swin2sr-flores200-text-102k-glyphsafe-v2"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(f"protocol already exists: {PROTOCOL}")
    forbidden = [ROOT / "decision.json", *(ROOT / "outputs").glob("batch-*.json")]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(f"capacity output exists before seal: {existing}")
    inputs = {str(PARENT): file_sha256(PARENT)}
    for data_root in DATA:
        for filename in ("summary.json", "manifest.jsonl"):
            path = data_root / filename
            inputs[str(path)] = file_sha256(path)
    overlay = REPOSITORY / (
        "experiments/browser_swin2sr/prism_bonsai_transfer_20260731/"
        "retained_precision_study_v2_20260731/sealed_overlay/experiments/"
        "browser_swin2sr"
    )
    base = REPOSITORY / (
        "experiments/browser_swin2sr/scale_codebook_qat_v2_20260731/"
        "sealed_sources/experiments/browser_swin2sr"
    )
    base_root = REPOSITORY / (
        "experiments/browser_swin2sr/scale_codebook_qat_v2_20260731/"
        "sealed_sources"
    )
    protocol = {
        "format": "backdropkit-swin2sr-max-stable-batch-study-v1",
        "status": "sealed_before_capacity_outputs",
        "scientific_scope": (
            "Capacity screen for the exact retained-precision Swin2SR training "
            "path on this 64 GiB Apple M5. It does not select quality or LR."
        ),
        "repository_root": str(REPOSITORY),
        "python": "/Applications/Xcode.app/Contents/Developer/usr/bin/python3",
        "pythonpath": ":".join((str(ROOT), str(overlay), str(base), str(base_root))),
        "parent_checkpoint": str(PARENT),
        "training_data": [str(path) for path in DATA],
        "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCES},
        "input_sha256": inputs,
        "probe_config": {
            "island": "image_io_fp16",
            "precision": "fp16",
            "seed": 20260734,
            "warmup_steps": 2,
            "measure_steps": 5,
            "learning_rate": 4e-6,
            "memory_fraction": 0.80,
            "headroom_fraction": 0.25,
            "crop_size": 128,
        },
        "search": {
            "start": 8,
            "geometric_multiplier": 2,
            "maximum": 1024,
            "quantum": 8,
            "refinement": "binary search after first rejection or failure",
        },
        "selection_rule": (
            "Largest multiple-of-eight batch with five measured finite updates "
            "and peak MPS driver allocation no greater than 75% of the process "
            "recommended maximum. A fresh process is required per batch."
        ),
        "throughput_caveat": (
            "Ascending capacity probes may benefit from filesystem cache. "
            "Throughput and quality are re-tested at equal images in a later "
            "randomized matched study before changing the training default."
        ),
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "protocol": str(PROTOCOL),
        "sha256": file_sha256(PROTOCOL),
        "source_files": len(SOURCES),
        "input_files": len(inputs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
