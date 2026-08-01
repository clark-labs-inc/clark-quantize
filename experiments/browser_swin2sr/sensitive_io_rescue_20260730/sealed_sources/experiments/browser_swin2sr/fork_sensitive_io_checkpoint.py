#!/usr/bin/env python3
"""Fork a sealed checkpoint with one explicit surface group-size change."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from full_surface_provenance import file_sha256
from run_state import atomic_torch_save


FORMAT = "backdropkit-swin2sr-surface-group-size-fork-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--expect-surface-step", type=int)
    parser.add_argument("--expect-current-group-size", type=int)
    return parser.parse_args()


def state_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                visit(child)
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode())
            digest.update(b"\0")
            digest.update(json.dumps(item, sort_keys=True).encode())
        elif isinstance(item, (torch.dtype, torch.device)):
            digest.update(type(item).__name__.encode())
            digest.update(b"\0")
            digest.update(str(item).encode())
        else:
            raise TypeError(f"unsupported checkpoint value: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def training_state_hashes(payload: dict) -> dict[str, str]:
    return {
        name: state_tree_sha256(payload[name])
        for name in ("model", "optimizer", "run_state")
    }


def fork_payload(
    payload: dict,
    *,
    parent_checkpoint: Path,
    parent_sha256: str,
    surface: str,
    group_size: int,
    expect_surface_step: int | None = None,
    expect_current_group_size: int | None = None,
) -> tuple[dict, dict]:
    if group_size <= 0:
        raise ValueError("group size must be positive")
    if payload["quantization"].get("scope") != "all_matrix_and_conv":
        raise ValueError("checkpoint is not a full-surface training checkpoint")
    if surface not in payload.get("quantized_modules", []):
        raise ValueError(f"surface is not quantized by this checkpoint: {surface}")
    surface_step = int(payload["surface_step"])
    if expect_surface_step is not None and surface_step != expect_surface_step:
        raise ValueError(
            f"unexpected surface step: {surface_step} != {expect_surface_step}"
        )

    quantization = payload["quantization"]
    training = payload["surface_training"]
    quantization_overrides = dict(
        quantization.get("surface_group_sizes", {}) or {}
    )
    training_overrides = dict(
        training.get("surface_group_sizes", {}) or {}
    )
    if quantization_overrides != training_overrides:
        raise ValueError("checkpoint surface group-size metadata disagrees")
    if surface in quantization_overrides:
        raise ValueError(f"surface already has a group-size override: {surface}")
    if expect_current_group_size is not None:
        if quantization.get("conv_grouping") != "kernel":
            raise ValueError("current convolution group size is not inferable")
        if expect_current_group_size != 9:
            raise ValueError(
                "the fork tool only infers the 3x3 kernel-grouped source as 9"
            )

    before = training_state_hashes(payload)
    overrides = dict(sorted({
        **quantization_overrides,
        surface: group_size,
    }.items()))
    quantization["surface_group_sizes"] = overrides
    training["surface_group_sizes"] = overrides
    payload["fork_provenance"] = {
        "format": FORMAT,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_surface_step": surface_step,
        "change": {
            "surface": surface,
            "previous_group_size": expect_current_group_size,
            "group_size": group_size,
        },
    }
    after = training_state_hashes(payload)
    if after != before:
        raise RuntimeError("checkpoint fork changed model, optimizer, or run state")
    return payload, {
        "parent_surface_step": surface_step,
        "surface_group_sizes": overrides,
        "training_state_sha256": after,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists() or args.receipt.exists():
        raise FileExistsError("output checkpoint or receipt already exists")
    parent_sha256 = file_sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu")
    forked, details = fork_payload(
        payload,
        parent_checkpoint=args.checkpoint,
        parent_sha256=parent_sha256,
        surface=args.surface,
        group_size=args.group_size,
        expect_surface_step=args.expect_surface_step,
        expect_current_group_size=args.expect_current_group_size,
    )
    atomic_torch_save(forked, args.out)
    reloaded = torch.load(args.out, map_location="cpu")
    if training_state_hashes(reloaded) != details["training_state_sha256"]:
        raise RuntimeError("serialized fork changed training state")
    receipt = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "fork_checkpoint": str(args.out),
        "fork_checkpoint_sha256": file_sha256(args.out),
        **details,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
