#!/usr/bin/env python3
"""Materialize a retained-precision checkpoint for sealed dense evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from materialize_surface_states import state_receipt
from surface_artifact_state import hard_surface_state_dict, runtime_surface_state_dict
from retained_precision_state import (
    CHECKPOINT_FORMAT as FORMAT,
    load_retained_precision_student as load_training_state,
)
from retained_precision_islands import ISLAND_SPECS


MATERIALIZATION_FORMAT = "backdropkit-swin2sr-retained-precision-materialization-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", default="cpu")
    return parser.parse_args()


def materialize(checkpoint: Path, out: Path, source_device: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("format") != FORMAT:
        raise ValueError("unsupported retained-precision checkpoint")
    config = payload["retained_fp16_training"]
    parent = Path(payload["parent_checkpoint"])
    student, surfaces, parent_payload, initialization, loaded = load_training_state(
        parent,
        config["island"],
        config["precision"],
        source_device,
        checkpoint,
    )
    if loaded["island_step"] != payload["island_step"]:
        raise RuntimeError("retained-precision checkpoint changed while loading")
    dense_state = hard_surface_state_dict(student, surfaces)
    out.mkdir(parents=True, exist_ok=True)
    dense_path = out / "dense-hard.pt"
    torch.save(dense_state, dense_path)
    precision = config["precision"]
    runtime_state = None
    runtime_path = None
    surface_contract = None
    if precision in {"ternary", "fp16"}:
        runtime_state = runtime_surface_state_dict(student, surfaces)
        runtime_path = out / "runtime-hard.pt"
        torch.save(runtime_state, runtime_path)
        retained_names = {
            name
            for name, _values in ISLAND_SPECS[config["island"]].modules
        } if precision == "fp16" else set()
        surface_contract = {
            f"{surface.name}.weight": {
                "category": surface.category,
                "method": surface.quantizer.method,
                "group_size": surface.quantizer.group_size,
                "rotation": surface.quantizer.rotation,
                "rotation_block": surface.quantizer.rotation_block,
            }
            for surface in surfaces
            if surface.name not in retained_names
        }
    receipt = {
        "format": MATERIALIZATION_FORMAT,
        "interpretation": (
            "All non-island quantized surfaces are hard ternary. The declared "
            "island is emitted at its trained precision; FP16 island values "
            "are exact IEEE FP16 roundtrips represented in a dense state."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_sha256": file_sha256(parent),
        "parent_step": int(parent_payload["step"]),
        "island_step": int(payload["island_step"]),
        "source_device": source_device,
        "retained_precision_training": config,
        "initialization": {
            name: value
            for name, value in initialization.items()
            if name != "parameters"
        },
        "dense_state": state_receipt(dense_path, dense_state),
    }
    if runtime_state is not None and runtime_path is not None:
        receipt.update({
            "all_target_surfaces_materialized_hard": True,
            "quantization": payload["quantization"],
            "surface_training": payload["surface_training"],
            "surfaces": surface_contract,
            "runtime_state": state_receipt(runtime_path, runtime_state),
        })
        if precision == "fp16":
            spec = ISLAND_SPECS[config["island"]]
            receipt["retained_precision_island"] = {
                "name": spec.name,
                "precision": "fp16",
                "weight_values": spec.weight_values,
                "weights": [f"{name}.weight" for name, _values in spec.modules],
                "storage_dtype": "float16",
            }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "manifest": str(manifest.resolve())}


def main() -> None:
    args = parse_args()
    receipt = materialize(args.checkpoint, args.out, args.source_device)
    print(json.dumps({
        "manifest": receipt["manifest"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "island": receipt["retained_precision_training"]["island"],
        "precision": receipt["retained_precision_training"]["precision"],
        "dense_state": receipt["dense_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
