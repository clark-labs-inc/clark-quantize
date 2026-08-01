#!/usr/bin/env python3
"""Rank ternary surfaces by one-at-a-time continuous-master restoration."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_hard_started_checkpoints import load_checkpoint_student
from evaluate_stratified import (
    edges,
    file_sha256,
    load_bundle_records,
    product_pixels,
    psnr,
    reconstruct,
    to_tensor,
)
from surface_artifact_state import (
    continuous_surface_state_dict,
    hard_surface_state_dict,
)
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-surface-restore-sensitivity-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--surface", action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def stratified_records(records: list[dict], maximum: int) -> list[dict]:
    if maximum < 1:
        raise ValueError("max-samples must be positive")
    if maximum >= len(records):
        return records
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source", "unknown"))].append(record)
    source_names = sorted(by_source)
    selected = []
    remaining = maximum
    for index, source in enumerate(source_names):
        allocation = remaining // (len(source_names) - index)
        members = by_source[source]
        by_stratum: dict[str, deque] = defaultdict(deque)
        for record in sorted(members, key=lambda item: item["sha256"]):
            stratum = (
                record.get("script")
                or record.get("language")
                or record.get("split")
                or "default"
            )
            by_stratum[str(stratum)].append(record)
        queues = [by_stratum[name] for name in sorted(by_stratum)]
        chosen = []
        while queues and len(chosen) < allocation:
            next_queues = []
            for queue in queues:
                if len(chosen) >= allocation:
                    break
                chosen.append(queue.popleft())
                if queue:
                    next_queues.append(queue)
            queues = next_queues
        selected.extend(chosen)
        remaining -= len(chosen)
    return selected


def load_sample(record: dict, device: str) -> dict:
    with Image.open(record["sealed_low_path"]) as source:
        low_image = source.convert("RGB").copy()
    with Image.open(record["sealed_high_path"]) as source:
        high_image = source.convert("RGB").copy()
    truth = np.asarray(high_image, dtype=np.float32) / 255.0
    return {
        "sha256": record["sha256"],
        "source": record.get("source"),
        "script": record.get("script"),
        "language": record.get("language"),
        "low": to_tensor(low_image, device),
        "truth": truth,
        "truth_edges": (edges(truth) + 1) / 2,
    }


def output_metrics(
    output: np.ndarray,
    sample: dict,
    teacher_output: np.ndarray,
) -> dict[str, float]:
    return {
        "psnr_vs_ground_truth": psnr(output, sample["truth"]),
        "edge_psnr_vs_ground_truth": psnr(
            (edges(output) + 1) / 2,
            sample["truth_edges"],
        ),
        "psnr_vs_teacher": psnr(output, teacher_output),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }


def state_metrics(
    model,
    samples: list[dict],
    teacher_outputs: list[np.ndarray],
    device: str,
) -> dict:
    rows = []
    outputs = []
    for sample, teacher_output in zip(samples, teacher_outputs):
        output, _elapsed = reconstruct(model, sample["low"], device)
        output = product_pixels(output)
        outputs.append(output)
        rows.append(output_metrics(output, sample, teacher_output))
    return {"summary": summarize(rows), "rows": rows, "outputs": outputs}


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        metric: candidate[metric] - baseline[metric]
        for metric in METRICS
    }


def copy_surface_state(
    model,
    state: dict[str, torch.Tensor],
    surface_name: str,
) -> None:
    module = model.get_submodule(surface_name)
    with torch.no_grad():
        module.weight.copy_(state[f"{surface_name}.weight"].to(module.weight))
        bias_name = f"{surface_name}.bias"
        if module.bias is not None and bias_name in state:
            module.bias.copy_(state[bias_name].to(module.bias))


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"sensitivity output exists: {args.out}")
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    bundle, records = load_bundle_records(args.bundle)
    selected_records = stratified_records(records, args.max_samples)
    samples = [load_sample(record, args.device) for record in selected_records]

    source, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        "cpu",
    )
    if len(surfaces) != 204:
        raise RuntimeError("checkpoint does not cover all 204 surfaces")
    hard_state = hard_surface_state_dict(source, surfaces)
    continuous_state = continuous_surface_state_dict(source, surfaces)
    available = {surface.name: surface for surface in surfaces}
    requested = args.surface or sorted(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown surfaces requested: {unknown}")

    teacher = pretrained(args.device).eval()
    hard = pretrained(args.device).eval()
    hard.load_state_dict(hard_state, strict=True)
    continuous = pretrained(args.device).eval()
    continuous.load_state_dict(continuous_state, strict=True)
    with torch.inference_mode():
        teacher_outputs = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in samples
        ]
        baseline = state_metrics(hard, samples, teacher_outputs, args.device)
        continuous_metrics = state_metrics(
            continuous,
            samples,
            teacher_outputs,
            args.device,
        )
        per_surface = []
        started = time.perf_counter()
        for index, name in enumerate(requested, start=1):
            copy_surface_state(hard, continuous_state, name)
            candidate = state_metrics(
                hard,
                samples,
                teacher_outputs,
                args.device,
            )
            copy_surface_state(hard, hard_state, name)
            surface = available[name]
            per_surface.append({
                "name": name,
                "category": surface.category,
                "values": surface.module.weight.numel(),
                "method": surface.quantizer.method,
                "group_size": surface.quantizer.group_size,
                "metrics": candidate["summary"],
                "delta_vs_hard": metric_delta(
                    candidate["summary"],
                    baseline["summary"],
                ),
            })
            if index % 10 == 0:
                print(json.dumps({
                    "evaluated_surfaces": index,
                    "total_surfaces": len(requested),
                }), flush=True)
    per_surface.sort(
        key=lambda item: (
            -item["delta_vs_hard"]["psnr_vs_ground_truth"],
            item["name"],
        )
    )
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "development_screen",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "device": args.device,
        "torch_threads": args.torch_threads,
        "samples": [
            {
                key: sample.get(key)
                for key in ("sha256", "source", "script", "language")
            }
            for sample in samples
        ],
        "sample_count": len(samples),
        "surface_count": len(requested),
        "hard": baseline["summary"],
        "continuous_master": continuous_metrics["summary"],
        "continuous_minus_hard": metric_delta(
            continuous_metrics["summary"],
            baseline["summary"],
        ),
        "surfaces": per_surface,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "sample_count": report["sample_count"],
        "surface_count": report["surface_count"],
        "continuous_minus_hard": report["continuous_minus_hard"],
        "top_surfaces": [
            {
                "name": surface["name"],
                "delta_vs_hard": surface["delta_vs_hard"],
            }
            for surface in per_surface[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
