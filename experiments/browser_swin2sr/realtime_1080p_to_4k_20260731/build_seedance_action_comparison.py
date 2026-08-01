"""Build an honest moving Seedance 2.0 x2 upscaling comparison.

The public 720p Seedance output is the held-out reference. This script adds a
deterministic product-demo HUD, downsamples that reference to a real 360p input,
and gives the exact same decoded input to bicubic and the Clark RGB student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from realtime_student import RealtimeLumaStudent, RealtimeStudentConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "assets/seedance-2-action-reference-720p.mp4"
DEFAULT_CHECKPOINT = ROOT / "run_rgb_5000step_12ch_20260801/checkpoint-final.pt"
DEFAULT_PACKAGE = ROOT / "huggingface/clark-realtime-ui-video-upscaler-x2"
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout or ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return info


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def product_overlay(frame_rgb: np.ndarray, index: int, total: int) -> np.ndarray:
    image = Image.fromarray(frame_rgb).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    progress = index / max(1, total - 1)

    draw.rounded_rectangle((32, 28, 438, 146), 22, fill=(12, 17, 28, 224))
    draw.ellipse((55, 53, 75, 73), fill=(172, 255, 47, 255))
    draw.text((88, 46), "MOTION TRACK", font=font(FONT_BOLD, 25), fill="white")
    draw.text(
        (56, 88),
        "Seedance action / live product demo",
        font=font(FONT_REGULAR, 16),
        fill=(207, 216, 232, 255),
    )
    draw.rounded_rectangle((56, 120, 408, 130), 5, fill=(63, 71, 89, 255))
    draw.rounded_rectangle(
        (56, 120, 56 + int(352 * progress), 130),
        5,
        fill=(172, 255, 47, 255),
    )

    card = (width - 414, height - 176, width - 30, height - 28)
    draw.rounded_rectangle(card, 25, fill=(250, 251, 255, 238))
    draw.text(
        (card[0] + 26, card[1] + 23),
        "Demo quality",
        font=font(FONT_BOLD, 21),
        fill=(24, 28, 38, 255),
    )
    draw.text(
        (card[0] + 26, card[1] + 57),
        "Fine edges • motion • small UI",
        font=font(FONT_REGULAR, 15),
        fill=(85, 91, 108, 255),
    )
    draw.rounded_rectangle(
        (card[0] + 242, card[1] + 96, card[2] - 20, card[3] - 17),
        14,
        fill=(93, 65, 246, 255),
    )
    draw.text(
        (card[0] + 264, card[1] + 105),
        "Enhance x2",
        font=font(FONT_BOLD, 15),
        fill="white",
    )
    draw.text(
        (card[0] + 26, card[1] + 107),
        f"Frame {index + 1:03d} / {total:03d}",
        font=font(FONT_REGULAR, 14),
        fill=(85, 91, 108, 255),
    )
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def raw_encoder(
    output: Path,
    width: int,
    height: int,
    fps: float,
    audio_source: Path,
    *,
    crf: int = 8,
) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0",
        "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a?", "-shortest",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "160k", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def close_encoder(process: subprocess.Popen) -> None:
    if process.stdin is None:
        raise RuntimeError("encoder stdin was not created")
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg encoder exited with {return_code}")


def build_hud_reference(source: Path, output: Path) -> VideoInfo:
    info = video_info(source)
    capture = cv2.VideoCapture(str(source))
    encoder = raw_encoder(output, info.width, info.height, info.fps, source)
    assert encoder.stdin is not None
    count = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        encoder.stdin.write(product_overlay(rgb, count, info.frames).tobytes())
        count += 1
    capture.release()
    close_encoder(encoder)
    if count != info.frames:
        raise RuntimeError(f"decoded {count} frames, expected {info.frames}")
    return video_info(output)


def resize_video(
    source: Path,
    output: Path,
    width: int,
    height: int,
    *,
    flags: str,
    crf: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vf", f"scale={width}:{height}:flags={flags}",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
        str(output),
    ])


def derive_source(reference: Path, output: Path) -> None:
    resize_video(reference, output, 640, 360, flags="lanczos", crf=8)


def build_bicubic(source: Path, output: Path, width: int, height: int) -> None:
    resize_video(source, output, width, height, flags="bicubic", crf=8)


def load_model(checkpoint_path: Path, device: str) -> RealtimeLumaStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student = checkpoint["manifest"]["student"]
    model = RealtimeLumaStudent(RealtimeStudentConfig(
        channels=int(student["channels"]),
        blocks=int(student["blocks"]),
        color_mode=student.get("color_mode", "luma"),
    ))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device)


def build_clark(source: Path, output: Path, model: RealtimeLumaStudent, device: str) -> None:
    info = video_info(source)
    capture = cv2.VideoCapture(str(source))
    encoder = raw_encoder(output, info.width * 2, info.height * 2, info.fps, source)
    assert encoder.stdin is not None
    with torch.inference_mode():
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
            enhanced = model(tensor).squeeze(0).permute(1, 2, 0)
            enhanced = (enhanced * 255.0).round().to(torch.uint8).cpu().numpy()
            encoder.stdin.write(enhanced.tobytes())
    capture.release()
    close_encoder(encoder)


def psnr(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0 * 255.0) / mse)


def evaluate(reference: Path, bicubic: Path, clark: Path) -> dict:
    captures = [cv2.VideoCapture(str(path)) for path in (reference, bicubic, clark)]
    totals = {
        "bicubic_mse": 0.0,
        "clark_mse": 0.0,
        "bicubic_edge_mse": 0.0,
        "clark_edge_mse": 0.0,
        "bicubic_temporal_mse": 0.0,
        "clark_temporal_mse": 0.0,
    }
    count = 0
    temporal_count = 0
    previous: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    while True:
        decoded = [capture.read() for capture in captures]
        if not all(ok for ok, _ in decoded):
            break
        frames = tuple(frame.astype(np.float32) for _, frame in decoded)
        ref, bic, cla = frames
        totals["bicubic_mse"] += float(np.mean((bic - ref) ** 2))
        totals["clark_mse"] += float(np.mean((cla - ref) ** 2))
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        bic_gray = cv2.cvtColor(bic, cv2.COLOR_BGR2GRAY)
        cla_gray = cv2.cvtColor(cla, cv2.COLOR_BGR2GRAY)
        edges = tuple(
            cv2.magnitude(
                cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1),
            )
            for gray in (ref_gray, bic_gray, cla_gray)
        )
        totals["bicubic_edge_mse"] += float(np.mean((edges[1] - edges[0]) ** 2))
        totals["clark_edge_mse"] += float(np.mean((edges[2] - edges[0]) ** 2))
        if previous is not None:
            ref_delta = ref - previous[0]
            totals["bicubic_temporal_mse"] += float(
                np.mean(((bic - previous[1]) - ref_delta) ** 2)
            )
            totals["clark_temporal_mse"] += float(
                np.mean(((cla - previous[2]) - ref_delta) ** 2)
            )
            temporal_count += 1
        previous = frames
        count += 1
    for capture in captures:
        capture.release()
    if not count or not temporal_count:
        raise RuntimeError("not enough frames for evaluation")
    measures = {
        "bicubic_psnr_db": psnr(totals["bicubic_mse"] / count),
        "clark_psnr_db": psnr(totals["clark_mse"] / count),
        "bicubic_edge_psnr_db": psnr(totals["bicubic_edge_mse"] / count),
        "clark_edge_psnr_db": psnr(totals["clark_edge_mse"] / count),
        "bicubic_temporal_psnr_db": psnr(
            totals["bicubic_temporal_mse"] / temporal_count
        ),
        "clark_temporal_psnr_db": psnr(
            totals["clark_temporal_mse"] / temporal_count
        ),
    }
    measures["psnr_gain_db"] = measures["clark_psnr_db"] - measures["bicubic_psnr_db"]
    measures["edge_gain_db"] = (
        measures["clark_edge_psnr_db"] - measures["bicubic_edge_psnr_db"]
    )
    measures["temporal_gain_db"] = (
        measures["clark_temporal_psnr_db"] - measures["bicubic_temporal_psnr_db"]
    )
    return {"frames": count, **measures}


def label_bar(text: str, width: int, height: int = 64) -> np.ndarray:
    image = Image.new("RGB", (width, height), (14, 17, 26))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), text, font=font(FONT_BOLD, 23), fill="white")
    return np.asarray(image)


def build_triptych(
    source: Path,
    bicubic: Path,
    clark: Path,
    audio_source: Path,
    output: Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    source_label: str = "REAL INPUT • 640×360 • nearest display",
    display_scale: int = 1,
) -> None:
    captures = [cv2.VideoCapture(str(path)) for path in (source, bicubic, clark)]
    fps = float(captures[1].get(cv2.CAP_PROP_FPS))
    output_width = int(captures[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(captures[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop is None:
        base_width, base_height = output_width, output_height
    else:
        base_width = crop[2] - crop[0]
        base_height = crop[3] - crop[1]
    panel_width = base_width * display_scale
    panel_height = base_height * display_scale
    labels = (
        source_label,
        f"BICUBIC x2 • {output_width}×{output_height}",
        f"CLARK x2 • {output_width}×{output_height} • 16.6 KiB",
    )
    bars = [label_bar(text, panel_width) for text in labels]
    encoder = raw_encoder(
        output,
        panel_width * 3,
        panel_height + bars[0].shape[0],
        fps,
        audio_source,
        crf=12,
    )
    assert encoder.stdin is not None
    while True:
        decoded = [capture.read() for capture in captures]
        if not all(ok for ok, _ in decoded):
            break
        frames = []
        for index, (_, bgr) in enumerate(decoded):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index == 0:
                rgb = cv2.resize(
                    rgb,
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if crop is not None:
                rgb = rgb[crop[1]:crop[3], crop[0]:crop[2]]
            if display_scale != 1:
                rgb = cv2.resize(
                    rgb,
                    (panel_width, panel_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            frames.append(np.vstack((bars[index], rgb)))
        encoder.stdin.write(np.hstack(frames).tobytes())
    for capture in captures:
        capture.release()
    close_encoder(encoder)


def build_before_after(
    source: Path,
    clark: Path,
    audio_source: Path,
    output: Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    display_scale: int = 1,
    source_label: str = "BEFORE • REAL 356×200 COMPRESSED INPUT",
    clark_label: str = "AFTER • CLARK x2 • 712×400 • 16.6 KiB",
) -> None:
    captures = [cv2.VideoCapture(str(path)) for path in (source, clark)]
    fps = float(captures[1].get(cv2.CAP_PROP_FPS))
    output_width = int(captures[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(captures[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop is None:
        base_width, base_height = output_width, output_height
    else:
        base_width = crop[2] - crop[0]
        base_height = crop[3] - crop[1]
    panel_width = base_width * display_scale
    panel_height = base_height * display_scale
    bars = [
        label_bar(source_label, panel_width),
        label_bar(clark_label, panel_width),
    ]
    encoder = raw_encoder(
        output,
        panel_width * 2,
        panel_height + bars[0].shape[0],
        fps,
        audio_source,
        crf=12,
    )
    assert encoder.stdin is not None
    while True:
        decoded = [capture.read() for capture in captures]
        if not all(ok for ok, _ in decoded):
            break
        frames = []
        for index, (_, bgr) in enumerate(decoded):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index == 0:
                rgb = cv2.resize(
                    rgb,
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if crop is not None:
                rgb = rgb[crop[1]:crop[3], crop[0]:crop[2]]
            if display_scale != 1:
                rgb = cv2.resize(
                    rgb,
                    (panel_width, panel_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            frames.append(np.vstack((bars[index], rgb)))
        encoder.stdin.write(np.hstack(frames).tobytes())
    for capture in captures:
        capture.release()
    close_encoder(encoder)


def build_wipe_comparison(
    source: Path,
    clark: Path,
    audio_source: Path,
    output: Path,
) -> None:
    """Build an aligned moving wipe so the eye compares the same pixels."""
    captures = [cv2.VideoCapture(str(path)) for path in (source, clark)]
    fps = float(captures[1].get(cv2.CAP_PROP_FPS))
    frames_total = int(captures[1].get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = int(captures[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(captures[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_width = output_width * 2
    display_height = output_height * 2
    header = np.hstack((
        label_bar("BEFORE • REAL 200×112 • CRF 36", display_width // 2),
        label_bar("AFTER • CLARK x2 • 400×224", display_width // 2),
    ))
    encoder = raw_encoder(
        output,
        display_width,
        display_height + header.shape[0],
        fps,
        audio_source,
        crf=12,
    )
    assert encoder.stdin is not None
    frame_index = 0
    while True:
        decoded = [capture.read() for capture in captures]
        if not all(ok for ok, _ in decoded):
            break
        source_rgb = cv2.cvtColor(decoded[0][1], cv2.COLOR_BGR2RGB)
        clark_rgb = cv2.cvtColor(decoded[1][1], cv2.COLOR_BGR2RGB)
        source_display = cv2.resize(
            source_rgb,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        clark_display = cv2.resize(
            clark_rgb,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        progress = frame_index / max(1, frames_total - 1)
        divider = int(display_width * (0.5 - 0.32 * math.cos(progress * math.tau)))
        comparison = source_display.copy()
        comparison[:, divider:] = clark_display[:, divider:]
        cv2.line(comparison, (divider - 3, 0), (divider - 3, display_height), (0, 0, 0), 7)
        cv2.line(comparison, (divider, 0), (divider, display_height), (172, 255, 47), 5)
        cv2.circle(
            comparison,
            (divider, display_height // 2),
            18,
            (14, 17, 26),
            thickness=-1,
        )
        cv2.circle(
            comparison,
            (divider, display_height // 2),
            14,
            (172, 255, 47),
            thickness=3,
        )
        encoder.stdin.write(np.vstack((header, comparison)).tobytes())
        frame_index += 1
    for capture in captures:
        capture.release()
    close_encoder(encoder)


def make_web_proxy(source: Path, output: Path) -> None:
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", "scale=1920:-2:flags=lanczos",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(output),
    ])


def still_comparison(reference: Path, source: Path, bicubic: Path, clark: Path, output: Path) -> None:
    paths = (source, bicubic, clark, reference)
    frames = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_MSEC, 2500)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cannot extract still from {path}")
        if frame.shape[1] == 640:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels = (
        "REAL 360p INPUT",
        "BICUBIC x2",
        "CLARK x2",
        "HELD-OUT 720p REFERENCE",
    )
    crop = (320, 80, 960, 720)
    panels = []
    for frame, text in zip(frames, labels):
        detail = frame[crop[1]:crop[3], crop[0]:crop[2]]
        panels.append(np.vstack((label_bar(text, 640), detail)))
    canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    for path in (args.input, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")

    assets = args.package / "examples/seedance-action"
    technical = args.package / "technical"
    original = assets / "seedance-2-action-original-720p.mp4"
    reference = assets / "seedance-2-action-product-reference-720p.mp4"
    low = assets / "seedance-2-action-real-input-360p.mp4"
    bicubic = assets / "seedance-2-action-bicubic-x2-720p.mp4"
    clark = assets / "seedance-2-action-clark-x2-720p.mp4"
    triptych = assets / "seedance-2-action-native-triptych.mp4"
    proxy = assets / "seedance-2-action-triptych-web.mp4"
    detail = assets / "seedance-2-action-detail-triptych.mp4"
    still = assets / "seedance-2-action-still-comparison.png"
    stress_target = assets / "seedance-2-action-held-out-target-400p.mp4"
    stress_low = assets / "seedance-2-action-real-input-200p-crf32.mp4"
    stress_bicubic = assets / "seedance-2-action-bicubic-x2-400p.mp4"
    stress_clark = assets / "seedance-2-action-clark-x2-400p.mp4"
    stress_triptych = assets / "seedance-2-action-200p-triptych.mp4"
    stress_proxy = assets / "seedance-2-action-200p-triptych-web.mp4"
    stress_detail = assets / "seedance-2-action-200p-magnified-detail.mp4"
    stress_before_after = assets / "seedance-2-action-200p-before-after.mp4"
    stress_before_after_detail = assets / "seedance-2-action-200p-before-after-magnified.mp4"
    stress_poster = assets / "seedance-2-action-200p-before-after-poster.png"
    tiny_target = assets / "seedance-2-action-held-out-target-400px.mp4"
    tiny_low = assets / "seedance-2-action-real-input-200px-crf36.mp4"
    tiny_bicubic = assets / "seedance-2-action-bicubic-x2-400px.mp4"
    tiny_clark = assets / "seedance-2-action-clark-x2-400px.mp4"
    tiny_triptych = assets / "seedance-2-action-200px-triptych.mp4"
    tiny_before_after = assets / "seedance-2-action-200px-before-after.mp4"
    tiny_poster = assets / "seedance-2-action-200px-before-after-poster.png"
    tiny_wipe = assets / "seedance-2-action-200px-wipe.mp4"
    tiny_wipe_poster = assets / "seedance-2-action-200px-wipe-poster.png"
    assets.mkdir(parents=True, exist_ok=True)
    technical.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.input, original)
    reference_info = build_hud_reference(args.input, reference)
    derive_source(reference, low)
    build_bicubic(low, bicubic, 1280, 720)
    model = load_model(args.checkpoint, args.device)
    build_clark(low, clark, model, args.device)
    metrics = evaluate(reference, bicubic, clark)
    build_triptych(low, bicubic, clark, reference, triptych)
    make_web_proxy(triptych, proxy)
    build_triptych(
        low,
        bicubic,
        clark,
        reference,
        detail,
        crop=(320, 80, 960, 720),
    )
    still_comparison(reference, low, bicubic, clark, still)

    # Deliberately severe demo: a compressed 356x200 source, honest x2 output,
    # and a separately derived 712x400 held-out target for measurement.
    resize_video(reference, stress_target, 712, 400, flags="lanczos", crf=8)
    resize_video(stress_target, stress_low, 356, 200, flags="lanczos", crf=32)
    build_bicubic(stress_low, stress_bicubic, 712, 400)
    build_clark(stress_low, stress_clark, model, args.device)
    stress_metrics = evaluate(stress_target, stress_bicubic, stress_clark)
    build_triptych(
        stress_low,
        stress_bicubic,
        stress_clark,
        stress_target,
        stress_triptych,
        source_label="REAL COMPRESSED INPUT • 356×200 • CRF 32",
    )
    make_web_proxy(stress_triptych, stress_proxy)
    build_triptych(
        stress_low,
        stress_bicubic,
        stress_clark,
        stress_target,
        stress_detail,
        crop=(178, 40, 534, 396),
        source_label="200p INPUT • 2× MAGNIFIED CROP",
        display_scale=2,
    )
    build_before_after(
        stress_low,
        stress_clark,
        stress_target,
        stress_before_after,
    )
    build_before_after(
        stress_low,
        stress_clark,
        stress_target,
        stress_before_after_detail,
        crop=(178, 40, 534, 396),
        display_scale=2,
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "2.5", "-i", str(stress_before_after_detail),
        "-frames:v", "1", str(stress_poster),
    ])

    # Model-card hero: an actual 200-pixel-wide source. Both sides occupy the
    # same display area, so the source is nearest-neighbor magnified 4x while
    # the honest x2 model output is nearest-neighbor magnified 2x. The
    # triptych preserves bicubic as the scientific control.
    resize_video(reference, tiny_target, 400, 224, flags="lanczos", crf=8)
    resize_video(tiny_target, tiny_low, 200, 112, flags="lanczos", crf=36)
    build_bicubic(tiny_low, tiny_bicubic, 400, 224)
    build_clark(tiny_low, tiny_clark, model, args.device)
    tiny_metrics = evaluate(tiny_target, tiny_bicubic, tiny_clark)
    build_triptych(
        tiny_low,
        tiny_bicubic,
        tiny_clark,
        tiny_target,
        tiny_triptych,
        source_label="REAL INPUT • 200×112 • H.264 CRF 36 • 4× DISPLAY",
        display_scale=2,
    )
    build_before_after(
        tiny_low,
        tiny_clark,
        tiny_target,
        tiny_before_after,
        display_scale=2,
        source_label="BEFORE • REAL 200×112 • CRF 36 • 4× DISPLAY",
        clark_label="AFTER • CLARK x2 • 400×224 • 2× DISPLAY",
    )
    build_wipe_comparison(tiny_low, tiny_clark, tiny_target, tiny_wipe)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "2.5", "-i", str(tiny_before_after),
        "-frames:v", "1", str(tiny_poster),
    ])
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "1.25", "-i", str(tiny_wipe),
        "-frames:v", "1", str(tiny_wipe_poster),
    ])

    artifacts = [
        original, reference, low, bicubic, clark, triptych, proxy, detail, still,
        stress_target, stress_low, stress_bicubic, stress_clark, stress_triptych,
        stress_proxy, stress_detail, stress_before_after, stress_before_after_detail,
        stress_poster, tiny_target, tiny_low, tiny_bicubic, tiny_clark,
        tiny_triptych, tiny_before_after, tiny_poster, tiny_wipe,
        tiny_wipe_poster,
    ]
    receipt = {
        "format": "clark-seedance-action-video-comparison-v1",
        "provenance": {
            "generator": "ByteDance Seedance 2.0",
            "public_prediction": "https://replicate.com/p/wckk6zjh01rmt0cxdt09fmsdzw",
            "prompt": "A golden retriever puppy chasing butterflies through a sunlit meadow, soft bokeh background, cinematic camera slowly tracking the puppy",
            "seed": 42,
            "source_sha256": sha256(args.input),
            "deterministic_overlay": "BackdropKit-style product HUD added before downsampling",
        },
        "comparison_contract": {
            "reference": f"{reference_info.width}x{reference_info.height} at {reference_info.fps:g} fps",
            "real_input": "640x360 Lanczos downsample encoded before either scaler",
            "bicubic": "ffmpeg bicubic x2 from the decoded real input",
            "clark": "RGB student x2 from the same decoded real input",
            "reference_visibility": "held out for metrics; shown only in the labeled still",
            "detail_crop": [320, 80, 960, 720],
        },
        "metrics": metrics,
        "compressed_200p_stress_test": {
            "contract": {
                "held_out_target": "712x400 Lanczos derivative of the 720p reference",
                "real_input": "356x200 H.264 CRF 32 derivative of the held-out target",
                "bicubic": "bicubic x2 from the decoded compressed input",
                "clark": "RGB student x2 from the same decoded compressed input",
                "detail_crop": [178, 40, 534, 396],
            },
            "metrics": stress_metrics,
        },
        "compressed_200px_wide_stress_test": {
            "contract": {
                "held_out_target": "400x224 Lanczos derivative of the 720p reference",
                "real_input": "200x112 H.264 CRF 36 derivative of the held-out target",
                "bicubic": "bicubic x2 from the decoded compressed input",
                "clark": "RGB student x2 from the same decoded compressed input",
                "display": "source nearest-neighbor 4x; x2 outputs nearest-neighbor 2x",
            },
            "metrics": tiny_metrics,
        },
        "artifacts": {
            str(path.relative_to(args.package)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        },
    }
    receipt_path = technical / "seedance-action-video-evaluation.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))


if __name__ == "__main__":
    main()
