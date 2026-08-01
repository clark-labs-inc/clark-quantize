"""Command-line contract for full-surface Swin2SR ternary training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from full_surface_optimizer import FULL_SURFACE_OPTIMIZERS
from surface_quantization import CONV_GROUPINGS
from surface_schedule import SCHEDULE_PROFILES


def surface_group_size(value: str) -> tuple[str, int]:
    name, separator, raw_size = value.partition("=")
    if not separator or not name or not raw_size:
        raise argparse.ArgumentTypeError(
            "surface group size must be NAME=POSITIVE_INTEGER"
        )
    try:
        group_size = int(raw_size)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "surface group size must be NAME=POSITIVE_INTEGER"
        ) from error
    if group_size <= 0:
        raise argparse.ArgumentTypeError("surface group size must be positive")
    return name, group_size


def normalize_surface_group_sizes(
    values: list[tuple[str, int]] | None,
) -> dict[str, int] | None:
    if values is None:
        return None
    normalized = {}
    for name, group_size in values:
        if name in normalized:
            raise ValueError(f"duplicate surface group-size override: {name}")
        normalized[name] = group_size
    return dict(sorted(normalized.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument(
        "--initialization",
        choices=("recovered_core", "fp32"),
        default="recovered_core",
    )
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--schedule-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int)
    parser.add_argument(
        "--surface-group-size",
        action="append",
        type=surface_group_size,
    )
    parser.add_argument(
        "--quantizer-method",
        choices=("optimal_mse", "absmean"),
    )
    parser.add_argument(
        "--conv-quantizer-method",
        choices=("optimal_mse", "absmean"),
    )
    parser.add_argument("--conv-grouping", choices=CONV_GROUPINGS)
    parser.add_argument("--schedule-profile", choices=SCHEDULE_PROFILES)
    parser.add_argument("--core-rotation", choices=("none", "hadamard"))
    parser.add_argument("--optimizer", choices=FULL_SURFACE_OPTIMIZERS)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--convolution-learning-rate-multiplier",
        type=float,
    )
    parser.add_argument(
        "--correction-learning-rate-multiplier",
        type=float,
    )
    parser.add_argument(
        "--loss-recipe",
        choices=("balanced", "teacher_heavy"),
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.05)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.25)
    parser.add_argument("--hard-shadow-every", type=int, default=2)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    args.surface_group_sizes = normalize_surface_group_sizes(
        args.surface_group_size
    )
    del args.surface_group_size
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.schedule_steps < 1 or args.crop_size % 16:
        raise ValueError("steps/schedule must be positive and crop divisible by 16")
    if (
        args.hard_shadow_weight_start < 0
        or args.hard_shadow_weight_end < args.hard_shadow_weight_start
        or args.hard_shadow_every < 0
    ):
        raise ValueError("invalid hard-shadow curriculum")
    for name in (
        "convolution_learning_rate_multiplier",
        "correction_learning_rate_multiplier",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
