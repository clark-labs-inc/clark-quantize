#!/usr/bin/env python3
"""Score a sealed external Swin2SR runtime against held-out truth and FP teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

from evaluate_stratified import (
    MODEL_ID,
    MODEL_REVISION,
    edges,
    grouped_summaries,
    product_pixels,
    psnr,
    reconstruct,
    summarize,
    to_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as source:
        image = source.convert("RGB").copy()
    return image, np.asarray(image, dtype=np.float32) / 255.0


def score_sample(
    runtime_output: np.ndarray,
    truth: np.ndarray,
    teacher_output: np.ndarray,
    bicubic: np.ndarray,
) -> dict[str, float]:
    return {
        "psnr_vs_ground_truth": psnr(runtime_output, truth),
        "edge_psnr_vs_ground_truth": psnr(
            (edges(runtime_output) + 1) / 2,
            (edges(truth) + 1) / 2,
        ),
        "psnr_vs_teacher": psnr(runtime_output, teacher_output),
        "bicubic_psnr_vs_ground_truth": psnr(bicubic, truth),
        "teacher_psnr_vs_ground_truth": psnr(teacher_output, truth),
    }


def load_bundle(bundle_path: Path) -> tuple[dict, dict[str, dict], Path]:
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("format") != "backdropkit-swin2sr-eval-v1":
        raise ValueError(f"unsupported bundle format: {bundle.get('format')}")
    root = bundle_path.parent
    manifest = Path(bundle["manifest"])
    if file_sha256(manifest) != bundle["manifest_sha256"]:
        raise ValueError("evaluation bundle manifest hash mismatch")
    records = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["source_sha256"]
        if key in records:
            raise ValueError(f"duplicate source sha256 in bundle: {key}")
        records[key] = record
    return bundle, records, root


def main() -> None:
    args = parse_args()
    bundle, records, root = load_bundle(args.bundle)
    inference = json.loads(args.inference.read_text())
    if inference.get("bundle_manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("runtime results were not generated from this evaluation bundle")
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(args.device).eval()
    output_root = args.inference.parent
    samples = []
    started = time.perf_counter()
    for index, runtime_sample in enumerate(inference["samples"], start=1):
        key = runtime_sample["source_sha256"]
        record = records.get(key)
        if record is None:
            raise ValueError(f"runtime result is absent from bundle: {key}")
        low_path = root / record["low_path"]
        high_path = root / record["high_path"]
        output_path = output_root / runtime_sample["output_path"]
        for path, expected in (
            (low_path, record["low_sha256"]),
            (high_path, record["high_sha256"]),
            (output_path, runtime_sample["output_sha256"]),
        ):
            if file_sha256(path) != expected:
                raise ValueError(f"image hash mismatch: {path}")
        low_image, _ = load_image(low_path)
        _, truth = load_image(high_path)
        _, runtime_output = load_image(output_path)
        if runtime_output.shape != truth.shape:
            raise ValueError(
                f"output shape {runtime_output.shape} does not match truth {truth.shape}: {key}"
            )
        bicubic = np.asarray(
            low_image.resize(
                (truth.shape[1], truth.shape[0]),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0
        teacher_output, teacher_seconds = reconstruct(
            teacher,
            to_tensor(low_image, args.device),
            args.device,
        )
        teacher_output = product_pixels(teacher_output)
        metrics = score_sample(runtime_output, truth, teacher_output, bicubic)
        samples.append({
            "sha256": key,
            "source": record.get("source", "unknown"),
            "split": record.get("split", "unknown"),
            "language": record.get("language"),
            "script": (
                record["language"].rsplit("_", 1)[-1]
                if record.get("language") and "_" in record["language"]
                else None
            ),
            **metrics,
            "teacher_seconds": teacher_seconds,
            "student_seconds": runtime_sample["inference_seconds"],
        })
        if index % 100 == 0:
            print(json.dumps({"scored": index, "total": len(inference["samples"])}), flush=True)
    report = {
        "model": inference["model"],
        "model_revision": inference["revision"],
        "state": (
            f"{inference['model']}@{inference['revision']}"
            f":{inference['dtype']}:{inference['backend']}"
        ),
        "state_sha256": next(
            file["sha256"]
            for file in inference["artifact_files"]
            if file["relative_path"] == "onnx/model_q4.onnx"
        ),
        "teacher_model": MODEL_ID,
        "teacher_model_revision": MODEL_REVISION,
        "device": args.device,
        "runtime_backend": inference["backend"],
        "crop_size": bundle["crop_size"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "artifact_bytes": inference["artifact_bytes"],
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
