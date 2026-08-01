"""Image discovery and deterministic UI-oriented degradation helpers."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def discover_images(roots: list[Path]) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    paths = []
    for root in roots:
        paths.extend(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in extensions
            and "node_modules" not in path.parts
            and path.stat().st_size > 0
        )
    if not paths:
        raise ValueError(f"no training images found under {roots}")
    return sorted(paths)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def ui_crop(image: Image.Image, crop_size: int, rng: random.Random) -> Image.Image:
    minimum = min(image.size)
    source_size = min(
        minimum,
        rng.randint(crop_size, max(crop_size, min(minimum, crop_size * 3))),
    )
    candidates = []
    for _ in range(6):
        left = rng.randint(0, max(0, image.width - source_size))
        top = rng.randint(0, max(0, image.height - source_size))
        candidate = image.crop((left, top, left + source_size, top + source_size))
        preview = np.asarray(
            candidate.resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        edge_score = float(
            np.abs(np.diff(preview, axis=0)).mean()
            + np.abs(np.diff(preview, axis=1)).mean()
        )
        candidates.append((edge_score, candidate))
    return max(candidates, key=lambda item: item[0])[1].resize(
        (crop_size, crop_size),
        Image.Resampling.LANCZOS,
    )


def degrade(high: Image.Image, rng: random.Random) -> Image.Image:
    low_size = high.width // 2
    low = high.resize((low_size, low_size), Image.Resampling.BICUBIC)
    if rng.random() < 0.35:
        buffer = io.BytesIO()
        low.save(buffer, format="JPEG", quality=rng.randint(55, 92), subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            low = compressed.convert("RGB").copy()
    return low


def to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def fixed_validation(
    data_root: Path,
    filenames: tuple[str, ...],
    crop_size: int,
    device: str,
):
    samples = []
    for filename in filenames:
        image = load_rgb(data_root / filename)
        left = max(0, (image.width - crop_size) // 2)
        top = max(0, (image.height - crop_size) // 2)
        high = image.crop((left, top, left + crop_size, top + crop_size))
        if high.size != (crop_size, crop_size):
            high = image.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
        low = high.resize((crop_size // 2, crop_size // 2), Image.Resampling.BICUBIC)
        samples.append((to_tensor(low, device), to_tensor(high, device)))
    return samples
