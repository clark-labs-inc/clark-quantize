#!/usr/bin/env python3
"""Generate a deterministic full-model fixture for the independent runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_stratified import load_hard_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_size < 16 or args.input_size % 8:
        raise ValueError("input-size must be at least 16 and divisible by 8")
    model = load_hard_student(
        args.state,
        args.device,
        60,
        "optimal_mse",
        "none",
        4,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    values = torch.rand(
        1,
        3,
        args.input_size,
        args.input_size,
        generator=generator,
    ).to(args.device)
    with torch.inference_mode():
        expected = model(values).reconstruction.float().cpu()
    fixture = {
        "format": "backdropkit-swin2sr-full-runtime-fixture-v1",
        "state": str(args.state),
        "seed": args.seed,
        "input_shape": list(values.shape),
        "output_shape": list(expected.shape),
        "input": values.float().cpu().reshape(-1).tolist(),
        "expected": expected.reshape(-1).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
