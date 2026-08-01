#!/usr/bin/env python3
"""Fork a full-surface checkpoint with an explicit ternary group-size map."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from fork_sensitive_io_checkpoint import training_state_hashes
from full_surface_provenance import file_sha256
from run_state import atomic_torch_save


FORMAT = "backdropkit-swin2sr-surface-group-map-fork-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--surface-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expect-surface-step", type=int, required=True)
    return parser.parse_args()


def fork_payload(
    payload: dict,
    *,
    parent_checkpoint: Path,
    parent_sha256: str,
    surface_map: dict[str, int],
    expect_surface_step: int,
) -> tuple[dict, dict]:
    if payload["quantization"].get("scope") != "all_matrix_and_conv":
        raise ValueError("checkpoint is not a full-surface checkpoint")
    if int(payload.get("surface_step", -1)) != expect_surface_step:
        raise ValueError("checkpoint is not at the expected surface step")
    quantized = set(payload.get("quantized_modules", []))
    if len(quantized) != 204:
        raise ValueError("checkpoint does not cover all 204 surfaces")
    normalized = {
        str(name): int(group_size)
        for name, group_size in surface_map.items()
    }
    if any(group_size <= 0 for group_size in normalized.values()):
        raise ValueError("surface group sizes must be positive")
    unknown = sorted(set(normalized) - quantized)
    if unknown:
        raise ValueError(f"surface map contains unknown surfaces: {unknown}")
    quantization = payload["quantization"]
    training = payload["surface_training"]
    quantization_overrides = dict(
        quantization.get("surface_group_sizes", {}) or {}
    )
    training_overrides = dict(training.get("surface_group_sizes", {}) or {})
    if quantization_overrides != training_overrides:
        raise ValueError("checkpoint surface group-size metadata disagrees")
    overlap = sorted(set(normalized) & set(quantization_overrides))
    if overlap:
        raise ValueError(f"surface map replaces existing overrides: {overlap}")
    before = training_state_hashes(payload)
    merged = dict(sorted({**quantization_overrides, **normalized}.items()))
    quantization["surface_group_sizes"] = merged
    training["surface_group_sizes"] = merged
    payload["surface_group_map_fork"] = {
        "format": FORMAT,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_surface_step": expect_surface_step,
        "previous_surface_group_sizes": dict(sorted(
            quantization_overrides.items()
        )),
        "added_surface_group_sizes": dict(sorted(normalized.items())),
        "surface_group_sizes": merged,
    }
    after = training_state_hashes(payload)
    if after != before:
        raise RuntimeError("surface-map fork changed model, optimizer, or run state")
    return payload, {
        "surface_step": expect_surface_step,
        "previous_surface_group_sizes": dict(sorted(
            quantization_overrides.items()
        )),
        "added_surface_group_sizes": dict(sorted(normalized.items())),
        "surface_group_sizes": merged,
        "training_state_sha256": after,
    }


def write_fork(
    checkpoint: Path,
    out: Path,
    receipt_path: Path,
    *,
    surface_map: dict[str, int],
    expect_surface_step: int,
) -> dict:
    if out.exists() or receipt_path.exists():
        raise FileExistsError("output checkpoint or receipt already exists")
    parent_sha256 = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    forked, details = fork_payload(
        payload,
        parent_checkpoint=checkpoint,
        parent_sha256=parent_sha256,
        surface_map=surface_map,
        expect_surface_step=expect_surface_step,
    )
    atomic_torch_save(forked, out)
    reloaded = torch.load(out, map_location="cpu", weights_only=False)
    if training_state_hashes(reloaded) != details["training_state_sha256"]:
        raise RuntimeError("serialized fork changed training state")
    receipt = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "fork_checkpoint": str(out),
        "fork_checkpoint_sha256": file_sha256(out),
        **details,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    args = parse_args()
    surface_map = json.loads(args.surface_map.read_text())
    if not isinstance(surface_map, dict):
        raise ValueError("surface-map file must contain a JSON object")
    receipt = write_fork(
        args.checkpoint,
        args.out,
        args.receipt,
        surface_map=surface_map,
        expect_surface_step=args.expect_surface_step,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
