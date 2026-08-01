#!/usr/bin/env python3
"""Measure one physical training batch in a fresh MPS process.

The probe uses the exact retained-precision student, full-precision teacher,
128 px crop pipeline, and production training objective.  A fresh process per
batch keeps failed capacity probes from poisoning later MPS measurements.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from data_pipeline import degrade, load_rgb, to_tensor, ui_crop
from recovery_data import BalancedSourceSampler
from surface_model_state import pretrained
from train_retained_fp16_island import load_training_state
from train_scale_codebook_recovery import gradient_receipt, task_loss


FORMAT = "backdropkit-swin2sr-batch-capacity-probe-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--island", default="image_io_fp16")
    parser.add_argument("--precision", choices=("fp16", "ternary"), default="fp16")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260734)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--headroom-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if min(args.warmup_steps, args.measure_steps) < 1:
        raise ValueError("warmup and measurement steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    if not 0 < args.headroom_fraction < 1:
        raise ValueError("headroom fraction must be in (0, 1)")


def mps_memory() -> dict[str, int]:
    torch.mps.synchronize()
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def update_peak(peak: dict[str, int], sample: dict[str, int]) -> None:
    for key in ("current_allocated_bytes", "driver_allocated_bytes"):
        peak[key] = max(peak.get(key, 0), sample[key])
    peak["recommended_max_bytes"] = sample["recommended_max_bytes"]


def prepare_batch(args: argparse.Namespace, crop_size: int):
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    paths = sampler.paths(0)
    rng = random.Random(args.seed)
    low_batch = []
    high_batch = []
    started = time.perf_counter()
    for path in paths[: args.batch_size]:
        high = ui_crop(load_rgb(path), crop_size, rng)
        low_batch.append(to_tensor(degrade(high, rng), args.device))
        high_batch.append(to_tensor(high, args.device))
    low = torch.cat(low_batch)
    high = torch.cat(high_batch)
    del low_batch, high_batch
    torch.mps.synchronize()
    return low, high, time.perf_counter() - started, sampler.metadata


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("capacity probing requires the MPS device")
    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    student, _, parent_payload, initialization, _ = load_training_state(
        args.parent,
        args.island,
        args.precision,
        args.device,
    )
    student.train()
    teacher = pretrained(args.device).eval()
    teacher.requires_grad_(False)
    parameters = initialization["parameters"]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    training = parent_payload["surface_training"]
    peak: dict[str, int] = {}
    stage_memory = {"model_loaded": mps_memory()}
    update_peak(peak, stage_memory["model_loaded"])

    low, high, preparation_seconds, sampler_metadata = prepare_batch(
        args,
        int(training["crop_size"]),
    )
    stage_memory["batch_prepared"] = mps_memory()
    update_peak(peak, stage_memory["batch_prepared"])

    step_seconds = []
    losses = []
    gradient_receipts = []
    total_steps = args.warmup_steps + args.measure_steps
    for step in range(total_steps):
        torch.mps.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            teacher_output = teacher(low, output_hidden_states=True)
        update_peak(peak, mps_memory())
        student_output = student(low, output_hidden_states=True)
        update_peak(peak, mps_memory())
        loss, _ = task_loss(student_output, teacher_output, high, training)
        update_peak(peak, mps_memory())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        receipt = gradient_receipt(parameters)
        update_peak(peak, mps_memory())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        update_peak(peak, mps_memory())
        if step >= args.warmup_steps:
            step_seconds.append(elapsed)
            losses.append(float(loss.item()))
            gradient_receipts.append(receipt)
        del teacher_output, student_output, loss

    recommended = peak["recommended_max_bytes"]
    driver_fraction = peak["driver_allocated_bytes"] / recommended
    median_step = statistics.median(step_seconds)
    p95_step = percentile(step_seconds, 0.95)
    finite = all(
        receipt["finite"] and np.isfinite(value)
        for receipt, value in zip(gradient_receipts, losses)
    )
    accepted = finite and driver_fraction <= 1.0 - args.headroom_fraction
    result = {
        "format": FORMAT,
        "status": "accepted" if accepted else "rejected",
        "acceptance": {
            "all_losses_and_gradients_finite": finite,
            "required_headroom_fraction": args.headroom_fraction,
            "observed_driver_fraction_of_recommended": driver_fraction,
        },
        "config": {
            "parent": str(args.parent.resolve()),
            "island": args.island,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "crop_size": int(training["crop_size"]),
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "learning_rate": args.learning_rate,
            "memory_fraction": args.memory_fraction,
            "device": args.device,
        },
        "throughput": {
            "preparation_seconds": preparation_seconds,
            "preparation_images_per_second": args.batch_size / preparation_seconds,
            "median_step_seconds": median_step,
            "p95_step_seconds": p95_step,
            "compute_images_per_second": args.batch_size / median_step,
            "projected_serial_end_to_end_images_per_second": (
                args.batch_size / (median_step + preparation_seconds)
            ),
        },
        "memory": {"peak": peak, "stages": stage_memory},
        "losses": losses,
        "gradient_receipts": gradient_receipts,
        "sampler": sampler_metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
