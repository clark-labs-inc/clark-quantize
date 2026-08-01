#!/usr/bin/env python3
"""Evaluate and render a trained low-bit Swin2SR state on product UI crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
SAMPLES = (
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-composer-current.png", 0.54, 0.52),
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-chat-current.png", 0.62, 0.56),
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-research-current.png", 0.43, 0.49),
    (
        "/Users/stan/Documents/git/clark-desktop/promos/"
        "clark-code-ad-parallel-research-vertical-poster.png",
        0.51,
        0.45,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mse = float(np.mean((actual.astype(np.float32) - expected.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def edges(image: np.ndarray) -> np.ndarray:
    dx = np.diff(image, axis=1, append=image[:, -1:, :])
    dy = np.diff(image, axis=0, append=image[-1:, :, :])
    return np.concatenate([dx, dy], axis=2)


def reconstruct(model, processor, low: Image.Image, device: str, output_size: int) -> np.ndarray:
    pixels = processor(images=low, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        output = model(pixels).reconstruction[:, :, :output_size, :output_size]
    return output.clamp(0, 1).float().cpu()[0].permute(1, 2, 0).numpy()


def panel(image: np.ndarray, label: str, size: int) -> Image.Image:
    result = Image.new("RGB", (size, size + 28), "#151515")
    result.paste(Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)), (0, 28))
    ImageDraw.Draw(result).text((5, 8), label, fill="white")
    return result


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=False
    )
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(args.device).eval()
    student = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(args.device).eval()
    student.load_state_dict(torch.load(args.state, map_location="cpu"), strict=True)
    student.to(args.device)
    metrics = []
    rows = []

    for source_path, center_x, center_y in SAMPLES:
        source = Image.open(source_path).convert("RGB")
        left = max(0, min(source.width - args.crop_size, round(source.width * center_x - args.crop_size / 2)))
        top = max(0, min(source.height - args.crop_size, round(source.height * center_y - args.crop_size / 2)))
        high = source.crop((left, top, left + args.crop_size, top + args.crop_size))
        low = high.resize((args.crop_size // 2, args.crop_size // 2), Image.Resampling.BICUBIC)
        truth = np.asarray(high, dtype=np.float32) / 255
        bicubic = np.asarray(
            low.resize((args.crop_size, args.crop_size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255
        teacher_output = reconstruct(teacher, processor, low, args.device, args.crop_size)
        student_output = reconstruct(student, processor, low, args.device, args.crop_size)
        metrics.append({
            "sample": Path(source_path).stem,
            "psnr_vs_ground_truth": psnr(student_output, truth),
            "edge_psnr_vs_ground_truth": psnr((edges(student_output) + 1) / 2, (edges(truth) + 1) / 2),
            "psnr_vs_teacher": psnr(student_output, teacher_output),
        })
        rows.append([truth, bicubic, teacher_output, student_output])

    width = args.crop_size * 4
    height = (args.crop_size + 28) * len(rows)
    montage = Image.new("RGB", (width, height), "#151515")
    labels = ("ground truth", "bicubic", "FP32 teacher", "trained low-bit")
    for row_index, row in enumerate(rows):
        for column, (image, label) in enumerate(zip(row, labels)):
            montage.paste(panel(image, label, args.crop_size), (column * args.crop_size, row_index * (args.crop_size + 28)))
    montage.save(args.out / "comparison.png")
    summary = {
        "samples": metrics,
        "mean": {
            key: float(np.mean([sample[key] for sample in metrics]))
            for key in ("psnr_vs_ground_truth", "edge_psnr_vs_ground_truth", "psnr_vs_teacher")
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
