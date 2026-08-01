#!/usr/bin/env python3
"""Build the public Hugging Face package for the realtime UI video student."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file

from realtime_student import RealtimeLumaStudent, RealtimeStudentConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "run_1000step_12ch_20260731/checkpoint-final.pt"
DEFAULT_BROWSER = ROOT / "run_1000step_12ch_20260731/browser"
DEFAULT_OUTPUT = ROOT / "huggingface/clark-realtime-ui-video-upscaler-x2"
DEFAULT_RGB_RUNTIME_RECEIPT = ROOT / "runtime-receipt-rgb-20260801.json"
DEFAULT_VIDEOS = (
    Path("/Users/stan/Documents/git/clark-desktop/promos/clark-code-ad-direct-vs-claude-20s-16x9.mp4"),
    Path("/Users/stan/Documents/git/clark-desktop/promos/clark-code-demo.mp4"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def psnr(left: torch.Tensor, right: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(left.float(), right.float()).clamp_min(1e-12)
    return float((-10 * torch.log10(mse)).item())


def edge_psnr(left: torch.Tensor, right: torch.Tensor) -> float:
    def edges(values: torch.Tensor) -> torch.Tensor:
        horizontal = values[..., :, 1:] - values[..., :, :-1]
        vertical = values[..., 1:, :] - values[..., :-1, :]
        return torch.cat((
            torch.nn.functional.pad(horizontal, (0, 1, 0, 0)),
            torch.nn.functional.pad(vertical, (0, 0, 0, 1)),
        ), dim=1)

    return psnr(edges(left), edges(right))


def tensor_from_image(image: Image.Image, device: str) -> torch.Tensor:
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device)


def image_from_tensor(values: torch.Tensor) -> Image.Image:
    pixels = (
        values.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{timestamp:.6f}",
            "-i", str(video), "-frames:v", "1", str(output),
        ],
        check=True,
    )


def dense_crop(image: Image.Image, width: int = 360, height: int = 220) -> tuple[int, int, int, int]:
    """Pick a UI-dense detail using only reference edges, never model score."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    height = min(height, image.height)
    width = min(width, image.width)
    integral = np.pad(gradient, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1.0, 0, 0)
    for y in range(0, image.height - height + 1, 12):
        for x in range(0, image.width - width + 1, 12):
            score = (
                integral[y + height, x + width]
                - integral[y, x + width]
                - integral[y + height, x]
                + integral[y, x]
            )
            if score > best[0]:
                best = (float(score), x, y)
    return best[1], best[2], best[1] + width, best[2] + height


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def comparison_card(
    source: Image.Image,
    bicubic: Image.Image,
    clark: Image.Image,
    crop: tuple[int, int, int, int],
    caption: str,
) -> Image.Image:
    panels = [source.crop(crop), bicubic.crop(crop), clark.crop(crop)]
    target_height = 360
    panels = [
        panel.resize(
            (round(panel.width * target_height / panel.height), target_height),
            Image.Resampling.NEAREST,
        )
        for panel in panels
    ]
    gap = 18
    header = 118
    footer = 44
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * 4, header + target_height + footer),
        "#0e1017",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 16), caption, fill="#ffffff", font=font(27, True))
    labels = (
        ("SOURCE INPUT", "640 × 360", "#a8adb9"),
        ("BASELINE", "Bicubic x2", "#a8adb9"),
        ("AFTER", "Clark realtime x2", "#d8ff4f"),
    )
    x = gap
    for panel, (kicker, label, color) in zip(panels, labels):
        draw.text((x, 58), kicker, fill=color, font=font(13, True))
        draw.text((x, 78), label, fill="#ffffff", font=font(17, True))
        canvas.paste(panel, (x, header))
        x += panel.width + gap
    draw.text(
        (gap, header + target_height + 13),
        "Same low-resolution input · deterministic edge-dense crop · no post-sharpening",
        fill="#a8adb9",
        font=font(15),
    )
    return canvas


@torch.inference_mode()
def build_examples(
    model: RealtimeLumaStudent,
    videos: list[Path],
    output: Path,
    device: str,
) -> dict:
    examples = output / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    frames = []
    cards = []
    positions = (0.22, 0.51, 0.79)
    with tempfile.TemporaryDirectory(prefix="backdropkit-hf-") as temporary:
        temp = Path(temporary)
        for index, position in enumerate(positions):
            video = videos[index % len(videos)]
            duration = video_duration(video)
            timestamp = min(max(0.0, duration - 0.05), duration * position)
            frame_path = temp / f"frame-{index}.png"
            extract_frame(video, timestamp, frame_path)
            reference = Image.open(frame_path).convert("RGB")
            even_size = (reference.width // 2 * 2, reference.height // 2 * 2)
            reference = reference.crop((0, 0, even_size[0], even_size[1]))
            low = reference.resize(
                (reference.width // 2, reference.height // 2),
                Image.Resampling.LANCZOS,
            )
            before = low.resize(reference.size, Image.Resampling.BICUBIC)
            source_display = low.resize(reference.size, Image.Resampling.NEAREST)
            low_tensor = tensor_from_image(low, device)
            after_tensor = model(low_tensor)
            after = image_from_tensor(after_tensor)
            reference_tensor = tensor_from_image(reference, device)
            before_tensor = tensor_from_image(before, device)
            crop = dense_crop(reference)
            card = comparison_card(
                source_display,
                before,
                after,
                crop,
                f"Product-demo detail {index + 1}",
            )
            card_name = f"product-demo-{index + 1}-before-after.png"
            card.save(examples / card_name, optimize=True)
            cards.append(card)
            frames.append({
                "example": card_name,
                "source": video.name,
                "timestamp_seconds": timestamp,
                "reference_size": list(reference.size),
                "detail_crop": list(crop),
                "bicubic_psnr_db": psnr(before_tensor, reference_tensor),
                "model_psnr_db": psnr(after_tensor, reference_tensor),
                "bicubic_edge_psnr_db": edge_psnr(before_tensor, reference_tensor),
                "model_edge_psnr_db": edge_psnr(after_tensor, reference_tensor),
            })
    hero_width = max(card.width for card in cards)
    hero = Image.new(
        "RGB",
        (hero_width, sum(card.height for card in cards) + 12 * (len(cards) - 1)),
        "#0e1017",
    )
    y = 0
    for card in cards:
        hero.paste(card, ((hero_width - card.width) // 2, y))
        y += card.height + 12
    hero.save(examples / "product-demo-before-after.png", optimize=True)
    return {
        "format": "clark-realtime-ui-video-x2-example-evaluation-v1",
        "selection": "fixed video fractions; detail crop chosen only by reference edge density",
        "frames": frames,
        "mean": {
            "bicubic_psnr_db": float(np.mean([item["bicubic_psnr_db"] for item in frames])),
            "model_psnr_db": float(np.mean([item["model_psnr_db"] for item in frames])),
            "model_gain_db": float(np.mean([
                item["model_psnr_db"] - item["bicubic_psnr_db"] for item in frames
            ])),
            "bicubic_edge_psnr_db": float(np.mean([
                item["bicubic_edge_psnr_db"] for item in frames
            ])),
            "model_edge_psnr_db": float(np.mean([
                item["model_edge_psnr_db"] for item in frames
            ])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--browser", type=Path, default=DEFAULT_BROWSER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video", action="append", type=Path)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    videos = args.video or list(DEFAULT_VIDEOS)
    for path in [args.checkpoint, args.browser / "model.bin", *videos]:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "backdropkit-realtime-student-checkpoint-v1":
        raise ValueError("unsupported checkpoint")
    student = checkpoint["manifest"]["student"]
    runtime_receipt = args.runtime_receipt or (
        DEFAULT_RGB_RUNTIME_RECEIPT
        if student.get("color_mode") == "rgb"
        else ROOT / "runtime-receipt-20260731.json"
    )
    if not runtime_receipt.is_file():
        raise FileNotFoundError(runtime_receipt)
    model = RealtimeLumaStudent(RealtimeStudentConfig(
        channels=int(student["channels"]),
        blocks=int(student["blocks"]),
        color_mode=student.get("color_mode", "luma"),
    ))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "browser").mkdir(exist_ok=True)
    (args.out / "technical").mkdir(exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        args.out / "model.safetensors",
    )
    shutil.copy2(args.browser / "model.bin", args.out / "browser/model.bin")
    browser_manifest = json.loads((args.browser / "model.json").read_text())
    browser_manifest.pop("source_checkpoint", None)
    browser_manifest["source_checkpoint_sha256"] = sha256(args.checkpoint)
    (args.out / "browser/model.json").write_text(
        json.dumps(browser_manifest, indent=2) + "\n"
    )
    shutil.copy2(runtime_receipt, args.out / "technical/runtime-receipt.json")
    shutil.copy2(ROOT / "realtime_student.py", args.out / "realtime_student.py")
    evaluation = build_examples(model, videos, args.out, args.device)
    (args.out / "technical/example-evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n"
    )
    config = {
        "architectures": ["RealtimeLumaStudent"],
        "model_type": "clark_realtime_luma_x2",
        "channels": int(student["channels"]),
        "blocks": int(student["blocks"]),
        "color_mode": student.get("color_mode", "luma"),
        "scale": 2,
        "parameters": model.parameter_count(),
        "browser_format": browser_manifest["format"],
        "torch_dtype": "float32",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    ledger = {
        "format": "clark-realtime-ui-video-upscaler-x2-package-v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "artifacts": {
            str(path.relative_to(args.out)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                args.out / "model.safetensors",
                args.out / "browser/model.bin",
                args.out / "browser/model.json",
                args.out / "technical/runtime-receipt.json",
                args.out / "technical/example-evaluation.json",
            )
        },
    }
    (args.out / "technical/package-ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n"
    )
    print(json.dumps({"output": str(args.out), **ledger}, indent=2))


if __name__ == "__main__":
    main()
