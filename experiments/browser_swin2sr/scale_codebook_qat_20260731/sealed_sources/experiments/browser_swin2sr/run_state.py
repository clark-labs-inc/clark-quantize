"""Durable, reproducible state helpers for Swin2SR QAT runs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def epoch_order(images: list[Path], seed: int, epoch: int) -> list[Path]:
    """Return an epoch permutation independent of augmentation randomness."""

    ordered = images.copy()
    random.Random(seed + epoch).shuffle(ordered)
    return ordered


def capture_run_state(
    augmentation_rng: random.Random,
    epoch: int,
    image_cursor: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    numpy_rng = np.random.get_state()
    state: dict[str, Any] = {
        "augmentation_rng": augmentation_rng.getstate(),
        "numpy_rng": {
            "bit_generator": numpy_rng[0],
            "state": torch.from_numpy(numpy_rng[1].copy()),
            "position": numpy_rng[2],
            "has_gauss": numpy_rng[3],
            "cached_gaussian": numpy_rng[4],
        },
        "torch_rng": torch.get_rng_state(),
        "epoch": epoch,
        "image_cursor": image_cursor,
        "elapsed_seconds": elapsed_seconds,
    }
    if torch.backends.mps.is_available():
        state["mps_rng"] = torch.mps.get_rng_state()
    return state


def restore_run_state(
    checkpoint: dict[str, Any],
    augmentation_rng: random.Random,
    examples_seen: int,
    image_count: int,
) -> tuple[int, int, float, bool]:
    """Restore stochastic state, with a deterministic fallback for old checkpoints."""

    state = checkpoint.get("run_state")
    if state is None:
        epoch, image_cursor = divmod(examples_seen, image_count)
        return epoch, image_cursor, 0.0, False

    augmentation_rng.setstate(state["augmentation_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((
        numpy_rng["bit_generator"],
        numpy_rng["state"].cpu().numpy().astype(np.uint32, copy=False),
        numpy_rng["position"],
        numpy_rng["has_gauss"],
        numpy_rng["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_rng"].cpu())
    if "mps_rng" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"].cpu())
    return (
        int(state["epoch"]),
        int(state["image_cursor"]),
        float(state.get("elapsed_seconds", 0.0)),
        True,
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint without exposing a partially serialized destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_checkpoint(
    path: Path,
    student,
    optimizer,
    quantizers,
    step: int,
    metrics,
    run_state,
    optimizer_config,
) -> None:
    first_quantizer = quantizers[0][1]
    atomic_torch_save({
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_config": optimizer_config,
        "step": step,
        "metrics": metrics,
        "quantized_modules": [name for name, _, _ in quantizers],
        "quantization": {
            "mode": first_quantizer.mode,
            "method": first_quantizer.method,
            "group_size": first_quantizer.group_size,
            "rotation": first_quantizer.rotation,
            "rotation_block": first_quantizer.rotation_block,
        },
        "run_state": run_state,
    }, path)
