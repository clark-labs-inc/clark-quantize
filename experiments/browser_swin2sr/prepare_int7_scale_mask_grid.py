#!/usr/bin/env python3
"""Build the exhaustive four-layer max-versus-Lloyd int7 scale grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


STAGE_WEIGHTS = tuple(
    f"swin2sr.encoder.stages.{stage}.conv.weight"
    for stage in range(4)
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_endpoints(max_state: dict, lloyd_state: dict) -> None:
    if max_state.keys() != lloyd_state.keys():
        raise ValueError("endpoint state keys differ")
    for name in max_state:
        if name in STAGE_WEIGHTS:
            if torch.equal(max_state[name], lloyd_state[name]):
                raise ValueError(f"scale endpoints are identical: {name}")
        elif not torch.equal(max_state[name], lloyd_state[name]):
            raise ValueError(f"non-factor tensor differs: {name}")


def compose_mask(max_state: dict, lloyd_state: dict, mask: int) -> dict:
    if not 0 <= mask < 16:
        raise ValueError("four-layer mask must be between 0 and 15")
    state = dict(max_state)
    for stage, name in enumerate(STAGE_WEIGHTS):
        if mask & (1 << stage):
            state[name] = lloyd_state[name]
    return state


def prepare_diagnostic_bundle(source_bundle: Path, out: Path) -> dict:
    bundle = json.loads(source_bundle.read_text())
    source_manifest = Path(bundle["manifest"])
    records = [
        json.loads(line)
        for line in source_manifest.read_text().splitlines()
        if line.strip()
    ]
    rico = sorted(
        (record for record in records if record["source"].startswith("rico")),
        key=lambda record: record["source_sha256"],
    )[:200]
    by_language = {}
    for record in sorted(records, key=lambda item: item["source_sha256"]):
        language = record.get("language")
        if language is not None and language not in by_language:
            by_language[language] = record
    if len(by_language) != 204:
        raise ValueError(f"expected 204 languages, found {len(by_language)}")
    selected = rico + [by_language[name] for name in sorted(by_language)]
    source_root = source_bundle.parent
    absolute_records = []
    for record in selected:
        copy = dict(record)
        copy["low_path"] = str((source_root / record["low_path"]).resolve())
        copy["high_path"] = str((source_root / record["high_path"]).resolve())
        absolute_records.append(copy)
    manifest = out / "diagnostic-manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in absolute_records)
    )
    diagnostic = {
        "format": "backdropkit-swin2sr-eval-v1",
        "crop_size": bundle["crop_size"],
        "low_size": bundle["low_size"],
        "seed": 20260801,
        "samples": len(selected),
        "source_manifests": [{
            "path": str(source_manifest),
            "sha256": file_sha256(source_manifest),
        }],
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
    }
    bundle_path = out / "diagnostic-bundle.json"
    bundle_path.write_text(json.dumps(diagnostic, indent=2) + "\n")
    return {**diagnostic, "bundle": str(bundle_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-state", type=Path, required=True)
    parser.add_argument("--lloyd-state", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_state = torch.load(args.max_state, map_location="cpu", weights_only=True)
    lloyd_state = torch.load(args.lloyd_state, map_location="cpu", weights_only=True)
    validate_endpoints(max_state, lloyd_state)
    args.out.mkdir(parents=True, exist_ok=True)
    masks = []
    for mask in range(16):
        mask_dir = args.out / f"mask-{mask:02d}"
        mask_dir.mkdir(exist_ok=True)
        state_path = mask_dir / "dense-hard.pt"
        torch.save(compose_mask(max_state, lloyd_state, mask), state_path)
        masks.append({
            "mask": mask,
            "lloyd_layers": [
                name
                for stage, name in enumerate(STAGE_WEIGHTS)
                if mask & (1 << stage)
            ],
            "state": str(state_path.resolve()),
            "state_sha256": file_sha256(state_path),
        })
    diagnostic = prepare_diagnostic_bundle(args.source_bundle, args.out)
    receipt = {
        "format": "backdropkit-swin2sr-int7-layer-scale-grid-v1",
        "max_state": str(args.max_state.resolve()),
        "max_state_sha256": file_sha256(args.max_state),
        "lloyd_state": str(args.lloyd_state.resolve()),
        "lloyd_state_sha256": file_sha256(args.lloyd_state),
        "stage_weights": STAGE_WEIGHTS,
        "masks": masks,
        "diagnostic_bundle": diagnostic,
    }
    manifest = args.out / "grid.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "grid": str(manifest.resolve()),
        "masks": len(masks),
        "diagnostic_samples": diagnostic["samples"],
    }, indent=2))


if __name__ == "__main__":
    main()
