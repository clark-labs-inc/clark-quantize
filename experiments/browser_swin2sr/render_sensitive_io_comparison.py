#!/usr/bin/env python3
"""Render a development-only visual comparison of sensitive I/O group sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_pipeline import fixed_validation
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from probe_sensitive_io_group_sizes import METHODS, TARGETS, parse_group_sizes
from qat_objective import psnr
from surface_curriculum import enter_hard_shadow, restore_live_amounts
from train_swin2sr_qat import VALIDATION_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="swin2sr.first_convolution",
    )
    parser.add_argument("--method", choices=METHODS, default="optimal_mse")
    parser.add_argument(
        "--group-sizes",
        type=parse_group_sizes,
        default=parse_group_sizes("9,2"),
    )
    return parser.parse_args()


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor[0]
        .detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def labelled_montage(columns: list[tuple[str, Image.Image]]) -> Image.Image:
    width, height = columns[0][1].size
    if any(image.size != (width, height) for _, image in columns):
        raise ValueError("montage columns must have equal dimensions")
    label_height = 24
    result = Image.new("RGB", (width * len(columns), height + label_height), "white")
    draw = ImageDraw.Draw(result)
    for index, (label, image) in enumerate(columns):
        x = index * width
        result.paste(image, (x, label_height))
        draw.text((x + 5, 5), label, fill="black")
    return result


@torch.inference_mode()
def render(args: argparse.Namespace) -> dict:
    student, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        args.device,
    )
    target = next(
        surface
        for surface in surfaces
        if surface.name == args.target
    )
    if float(target.quantizer.amount.item()) <= 0:
        raise ValueError(f"target surface has not started: {args.target}")
    samples = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    live_amounts = enter_hard_shadow(surfaces)
    original_method = target.quantizer.method
    original_group_size = target.quantizer.group_size
    outputs: dict[int, list[torch.Tensor]] = {}
    try:
        target.quantizer.method = args.method
        for group_size in args.group_sizes:
            target.quantizer.group_size = group_size
            outputs[group_size] = [
                student(low).reconstruction.clamp(0, 1)
                for low, _high in samples
            ]
    finally:
        target.quantizer.method = original_method
        target.quantizer.group_size = original_group_size
        restore_live_amounts(surfaces, live_amounts)
    args.out.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for index, ((low, high), filename) in enumerate(
        zip(samples, VALIDATION_FILES)
    ):
        high_image = image_from_tensor(high)
        low_image = image_from_tensor(low).resize(
            high_image.size,
            Image.Resampling.BICUBIC,
        )
        output_path = args.out / f"{index:02d}-{Path(filename).stem}.png"
        labelled_montage([
            ("bicubic input", low_image),
            *[
                (
                    f"{args.method} g{group_size}",
                    image_from_tensor(outputs[group_size][index]),
                )
                for group_size in args.group_sizes
            ],
            ("ground truth", high_image),
        ]).save(output_path)
        comparisons.append({
            "source": str(args.validation_data / filename),
            "source_sha256": file_sha256(args.validation_data / filename),
            "render": str(output_path),
            "render_sha256": file_sha256(output_path),
            "psnr_by_group_size": {
                str(group_size): psnr(outputs[group_size][index], high)
                for group_size in args.group_sizes
            },
        })
    return {
        "format": "backdropkit-swin2sr-sensitive-io-visual-v2",
        "diagnostic_only": True,
        "selection_source": "development",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "render_source_sha256": file_sha256(Path(__file__)),
        "surface_step": int(payload["surface_step"]),
        "device": args.device,
        "target": target.name,
        "method": args.method,
        "group_sizes": list(args.group_sizes),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop size must be divisible by 16")
    receipt = render(args)
    receipt_path = args.out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "receipt": str(receipt_path),
        "renders": len(receipt["comparisons"]),
        "mean_psnr_by_group_size": {
            str(group_size): float(np.mean([
                item["psnr_by_group_size"][str(group_size)]
                for item in receipt["comparisons"]
            ]))
            for group_size in args.group_sizes
        },
    }, indent=2))


if __name__ == "__main__":
    main()
