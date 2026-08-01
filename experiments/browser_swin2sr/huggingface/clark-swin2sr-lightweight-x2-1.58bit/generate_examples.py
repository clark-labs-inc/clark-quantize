#!/usr/bin/env python3
"""Generate Set5 x2 examples and an auditable benchmark receipt."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw, ImageFont
from transformers import Swin2SRForImageSuperResolution


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
EXAMPLE_NAMES = ("baby", "butterfly", "woman")


def image_from_tar(archive: Path, suffix: str) -> Image.Image:
    with tarfile.open(archive, "r:gz") as bundle:
        member = next(item for item in bundle.getmembers() if item.name.endswith(suffix))
        handle = bundle.extractfile(member)
        if handle is None:
            raise FileNotFoundError(suffix)
        return Image.open(io.BytesIO(handle.read())).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def upscale(model: Swin2SRForImageSuperResolution, image: Image.Image) -> Image.Image:
    values = to_tensor(image)
    height, width = values.shape[-2:]
    padded_height = math.ceil(height / 8) * 8
    padded_width = math.ceil(width / 8) * 8
    values = functional.pad(
        values,
        (0, padded_width - width, 0, padded_height - height),
        mode="replicate",
    )
    with torch.inference_mode():
        result = model(values).reconstruction[:, :, : height * 2, : width * 2]
    pixels = (
        result.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(pixels, "RGB")


def psnr(actual: Image.Image, expected: Image.Image) -> float:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    error = np.mean((actual_values - expected_values) ** 2)
    return float("inf") if error == 0 else 10 * math.log10((255 * 255) / error)


def detail_crop(image: Image.Image, size: int = 192) -> tuple[int, int, int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    height, width = gray.shape
    crop_size = min(size, width, height)
    stride = max(8, crop_size // 8)
    best = (0.0, 0, 0)
    for top in range(0, max(1, height - crop_size + 1), stride):
        for left in range(0, max(1, width - crop_size + 1), stride):
            score = float(gradient[top : top + crop_size, left : left + crop_size].mean())
            if score > best[0]:
                best = (score, left, top)
    _, left, top = best
    return left, top, left + crop_size, top + crop_size


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def comparison_card(
    name: str,
    bicubic: Image.Image,
    ternary: Image.Image,
    output: Path,
) -> None:
    crop = detail_crop(ternary)
    before = bicubic.crop(crop).resize((512, 512), Image.Resampling.NEAREST)
    after = ternary.crop(crop).resize((512, 512), Image.Resampling.NEAREST)
    card = Image.new("RGB", (1120, 720), "#f6f4ee")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((28, 24, 1092, 696), radius=30, fill="#ffffff")
    draw.text((64, 52), f"{name.title()} · Set5 ×2", fill="#17171a", font=font(30, True))
    draw.text(
        (64, 96),
        "The same low-resolution input, enlarged two ways",
        fill="#717079",
        font=font(20),
    )
    card.paste(before, (48, 132))
    card.paste(after, (560, 132))
    draw.rounded_rectangle((68, 148, 224, 190), radius=18, fill="#17171a")
    draw.text((88, 157), "Before · Bicubic", fill="#ffffff", font=font(18, True))
    draw.rounded_rectangle((580, 148, 790, 190), radius=18, fill="#d8ff4f")
    draw.text((600, 157), "After · Clark Ternary", fill="#17171a", font=font(18, True))
    draw.text(
        (48, 662),
        "100% crop · no sharpening added after model output",
        fill="#717079",
        font=font(16),
    )
    card.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hr", type=Path, required=True)
    parser.add_argument("--lr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    results = []
    for name in ("baby", "bird", "butterfly", "head", "woman"):
        low_resolution = image_from_tar(args.lr, f"/{name}.png")
        reference = image_from_tar(args.hr, f"/{name}.png")
        bicubic = low_resolution.resize(reference.size, Image.Resampling.BICUBIC)
        ternary = upscale(model, low_resolution)
        if ternary.size != reference.size:
            raise ValueError(f"{name}: output {ternary.size} does not match reference {reference.size}")
        bicubic_score = psnr(bicubic, reference)
        ternary_score = psnr(ternary, reference)
        results.append(
            {
                "image": name,
                "bicubic_rgb_psnr_db": bicubic_score,
                "ternary_rgb_psnr_db": ternary_score,
                "gain_db": ternary_score - bicubic_score,
                "input_size": list(low_resolution.size),
                "output_size": list(reference.size),
            }
        )
        if name in EXAMPLE_NAMES:
            comparison_card(
                name,
                bicubic,
                ternary,
                args.out / f"{name}-before-after.png",
            )

    receipt = {
        "benchmark": "Set5 bicubic x2",
        "metric": "RGB PSNR over the full image",
        "model": "Clark Swin2SR lightweight x2 Ternary Bonsai",
        "samples": results,
        "mean_bicubic_rgb_psnr_db": float(
            np.mean([row["bicubic_rgb_psnr_db"] for row in results])
        ),
        "mean_ternary_rgb_psnr_db": float(
            np.mean([row["ternary_rgb_psnr_db"] for row in results])
        ),
        "mean_gain_db": float(np.mean([row["gain_db"] for row in results])),
    }
    (args.out / "set5-x2-results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
