#!/usr/bin/env python3
"""Materialize exact FP16-island probes from one sealed ternary checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from materialize_surface_states import load_materialized_state, state_receipt
from retained_precision_islands import ISLAND_SPECS, materialize_fp16_island


FORMAT = "backdropkit-swin2sr-retained-fp16-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def prepare(checkpoint: Path, canonical_manifest: Path, out: Path) -> dict:
    canonical, baseline = load_materialized_state(
        canonical_manifest,
        "dense_state",
    )
    checkpoint_hash = file_sha256(checkpoint)
    if canonical["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("canonical state belongs to a different checkpoint")
    student, surfaces, payload = load_checkpoint_student(checkpoint, "cpu")
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, spec in sorted(ISLAND_SPECS.items()):
        state, projection = materialize_fp16_island(baseline, surfaces, spec)
        path = out / f"{name}.pt"
        torch.save(state, path)
        records[name] = {
            **projection,
            "state": state_receipt(path, state),
        }
    manifest = {
        "format": FORMAT,
        "interpretation": (
            "Every non-island tensor is byte-identical in tensor value to the "
            "sealed refined dense-hard state. Declared island weights are the "
            "checkpoint continuous masters rounded through IEEE FP16."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(payload["step"]),
        "canonical_manifest": str(canonical_manifest.resolve()),
        "canonical_manifest_sha256": file_sha256(canonical_manifest),
        "canonical_dense_state": canonical["dense_state"],
        "islands": records,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest": str(path.resolve())}


def main() -> None:
    args = parse_args()
    result = prepare(args.checkpoint, args.canonical_manifest, args.out)
    print(json.dumps({
        "manifest": result["manifest"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "islands": {
            name: {
                "weight_values": row["weight_values"],
                "projected_data_byte_delta": row["projected_data_byte_delta"],
                "state": row["state"],
            }
            for name, row in result["islands"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
