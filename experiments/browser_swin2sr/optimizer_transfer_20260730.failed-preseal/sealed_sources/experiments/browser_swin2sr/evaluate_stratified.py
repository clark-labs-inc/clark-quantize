#!/usr/bin/env python3
"""Evaluate hard-ternary Swin2SR across held-out UI and language strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

from artifact_state import hard_state_dict
from train_swin2sr_qat import (
    MODEL_ID,
    MODEL_REVISION,
    add_ternary_parametrizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument(
        "--quantizer",
        choices=("optimal_mse", "absmean", "paretoq_seq"),
        default="optimal_mse",
        help="Used for hard states and legacy checkpoints without metadata.",
    )
    parser.add_argument("--rotation", choices=("none", "hadamard"), default="none")
    parser.add_argument("--rotation-block", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(manifests: list[Path]) -> list[dict]:
    records = []
    for manifest in manifests:
        root = manifest.parent
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["manifest"] = str(manifest)
            record["path"] = str(root / record["local_path"])
            records.append(record)
    return records


def load_bundle_records(bundle_path: Path) -> tuple[dict, list[dict]]:
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("format") != "backdropkit-swin2sr-eval-v1":
        raise ValueError(f"unsupported bundle format: {bundle.get('format')}")
    manifest_path = Path(bundle["manifest"])
    if file_sha256(manifest_path) != bundle["manifest_sha256"]:
        raise ValueError("evaluation bundle manifest hash mismatch")
    records = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["sha256"] = record["source_sha256"]
        record["path"] = str(bundle_path.parent / record["high_path"])
        record["sealed_low_path"] = str(bundle_path.parent / record["low_path"])
        record["sealed_high_path"] = str(bundle_path.parent / record["high_path"])
        records.append(record)
    return bundle, records


def select_records(records: list[dict], maximum: int | None, seed: int) -> list[dict]:
    if maximum is None or len(records) <= maximum:
        return records
    ordered = records.copy()
    random.Random(seed).shuffle(ordered)
    return ordered[:maximum]


def load_hard_student(
    path: Path,
    device: str,
    group_size: int,
    quantizer_method: str,
    rotation: str,
    rotation_block: int,
) -> Swin2SRForImageSuperResolution:
    payload = torch.load(path, map_location="cpu")
    student = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    if "model" in payload and "optimizer" in payload:
        quantization = payload.get("quantization", {})
        mode = quantization.get("mode", "ternary")
        method = quantization.get("method", quantizer_method)
        checkpoint_group_size = int(quantization.get("group_size", group_size))
        checkpoint_rotation = quantization.get("rotation", rotation)
        checkpoint_rotation_block = int(
            quantization.get("rotation_block", rotation_block)
        )
        quantizers = add_ternary_parametrizations(
            student,
            mode,
            checkpoint_group_size,
            method,
            checkpoint_rotation,
            checkpoint_rotation_block,
        )
        student.load_state_dict(payload["model"], strict=True)
        state = hard_state_dict(student, quantizers)
        student = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        student.load_state_dict(state, strict=True)
    else:
        student.load_state_dict(payload, strict=True)
    return student.to(device).eval()


def deterministic_crop(record: dict, crop_size: int) -> tuple[Image.Image, Image.Image]:
    image = Image.open(record["path"]).convert("RGB")
    digest = record.get("sha256") or hashlib.sha256(record["path"].encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    minimum = min(image.size)
    source_size = min(
        minimum,
        rng.randint(crop_size, max(crop_size, min(minimum, crop_size * 3))),
    )
    candidates = []
    for _ in range(12):
        left = rng.randint(0, max(0, image.width - source_size))
        top = rng.randint(0, max(0, image.height - source_size))
        candidate = image.crop((left, top, left + source_size, top + source_size))
        preview = np.asarray(
            candidate.resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        score = float(
            np.abs(np.diff(preview, axis=0)).mean()
            + np.abs(np.diff(preview, axis=1)).mean()
        )
        candidates.append((score, candidate))
    high = max(candidates, key=lambda item: item[0])[1]
    high = high.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    low = high.resize((crop_size // 2, crop_size // 2), Image.Resampling.BICUBIC)
    return low, high


def to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def reconstruct(model, low: torch.Tensor, device: str) -> tuple[np.ndarray, float]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(low).reconstruction.clamp(0, 1)
    synchronize(device)
    elapsed = time.perf_counter() - started
    array = output.float().cpu()[0].permute(1, 2, 0).numpy()
    return array, elapsed


def psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mse = float(np.mean((actual.astype(np.float32) - expected.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def edges(image: np.ndarray) -> np.ndarray:
    dx = np.diff(image, axis=1, append=image[:, -1:, :])
    dy = np.diff(image, axis=0, append=image[-1:, :, :])
    return np.concatenate([dx, dy], axis=2)


def product_pixels(image: np.ndarray) -> np.ndarray:
    """Match the uint8 RawImage returned by BackdropKit's Transformers.js pipeline."""
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0


def sample_metrics(record, teacher, student, crop_size: int, device: str) -> dict:
    if record.get("sealed_low_path"):
        with Image.open(record["sealed_low_path"]) as source:
            low_image = source.convert("RGB").copy()
        with Image.open(record["sealed_high_path"]) as source:
            high_image = source.convert("RGB").copy()
        if low_image.size != (crop_size // 2, crop_size // 2):
            raise ValueError(f"sealed low image has unexpected size: {record['sha256']}")
        if high_image.size != (crop_size, crop_size):
            raise ValueError(f"sealed high image has unexpected size: {record['sha256']}")
    else:
        low_image, high_image = deterministic_crop(record, crop_size)
    low = to_tensor(low_image, device)
    truth = np.asarray(high_image, dtype=np.float32) / 255.0
    bicubic = np.asarray(
        low_image.resize((crop_size, crop_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    teacher_output, teacher_seconds = reconstruct(teacher, low, device)
    student_output, student_seconds = reconstruct(student, low, device)
    teacher_output = product_pixels(teacher_output)
    student_output = product_pixels(student_output)
    return {
        "sha256": record.get("sha256"),
        "source": record.get("source", "unknown"),
        "split": record.get("split", "unknown"),
        "language": record.get("language"),
        "script": (
            record["language"].rsplit("_", 1)[-1]
            if record.get("language") and "_" in record["language"]
            else None
        ),
        "psnr_vs_ground_truth": psnr(student_output, truth),
        "edge_psnr_vs_ground_truth": psnr(
            (edges(student_output) + 1) / 2,
            (edges(truth) + 1) / 2,
        ),
        "psnr_vs_teacher": psnr(student_output, teacher_output),
        "bicubic_psnr_vs_ground_truth": psnr(bicubic, truth),
        "teacher_psnr_vs_ground_truth": psnr(teacher_output, truth),
        "teacher_seconds": teacher_seconds,
        "student_seconds": student_seconds,
    }


METRIC_KEYS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
    "bicubic_psnr_vs_ground_truth",
    "teacher_psnr_vs_ground_truth",
    "teacher_seconds",
    "student_seconds",
)


def summarize(samples: list[dict]) -> dict:
    result = {"samples": len(samples)}
    for key in METRIC_KEYS:
        values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    gains = np.asarray([
        sample["psnr_vs_ground_truth"] - sample["bicubic_psnr_vs_ground_truth"]
        for sample in samples
    ])
    result["psnr_gain_over_bicubic"] = {
        "mean": float(gains.mean()),
        "p10": float(np.quantile(gains, 0.10)),
    }
    return result


def grouped_summaries(samples: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for sample in samples:
        if sample.get(key):
            groups[str(sample[key])].append(sample)
    return {
        name: summarize(group)
        for name, group in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    if bool(args.bundle) == bool(args.manifest):
        raise ValueError("provide exactly one of --bundle or --manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = None
    if args.bundle:
        bundle, records = load_bundle_records(args.bundle)
        if args.max_samples:
            records = records[:args.max_samples]
    else:
        records = select_records(
            load_records(args.manifest),
            args.max_samples,
            args.seed,
        )
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(args.device).eval()
    student = load_hard_student(
        args.state,
        args.device,
        args.group_size,
        args.quantizer,
        args.rotation,
        args.rotation_block,
    )
    started = time.perf_counter()
    samples = []
    for index, record in enumerate(records, start=1):
        samples.append(sample_metrics(
            record,
            teacher,
            student,
            args.crop_size,
            args.device,
        ))
        if index % 100 == 0:
            print(json.dumps({"evaluated": index, "total": len(records)}), flush=True)
    report = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "state": str(args.state),
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "device": args.device,
        "crop_size": args.crop_size,
        "manifests": [str(path) for path in args.manifest or []],
        "bundle": str(args.bundle) if args.bundle else None,
        "bundle_manifest_sha256": (
            bundle["manifest_sha256"] if bundle else None
        ),
        "overall": summarize(samples),
        "by_source": grouped_summaries(samples, "source"),
        "by_split": grouped_summaries(samples, "split"),
        "by_script": grouped_summaries(samples, "script"),
        "by_language": grouped_summaries(samples, "language"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "samples": samples,
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
