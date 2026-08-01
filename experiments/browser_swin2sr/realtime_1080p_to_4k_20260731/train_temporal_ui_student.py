#!/usr/bin/env python3
"""Train a slightly wider realtime student on deterministic UI motion pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Swin2SRForImageSuperResolution

STUDY_ROOT = Path(__file__).resolve().parent
SWIN_ROOT = STUDY_ROOT.parent
sys.path.insert(0, str(SWIN_ROOT))

from data_pipeline import discover_images, load_rgb, to_tensor, ui_crop
from realtime_student import (
    RealtimeLumaStudent,
    RealtimeStudentConfig,
    initialize_wider_rgb_from_rgb,
)
from train_realtime_student import (
    MODEL_ID,
    MODEL_REVISION,
    charbonnier,
    edge_loss,
    grouped_paths,
    psnr,
    sampling_groups,
    split_stratified,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.20)
    parser.add_argument("--teacher-weight", type=float, default=0.20)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def translated_pair(image, crop_size: int, rng: random.Random, device: str) -> torch.Tensor:
    margin = 4
    expanded = ui_crop(image, crop_size + margin * 2, rng)
    values = to_tensor(expanded, device)
    dx = rng.choice([value for value in range(-margin, margin + 1) if value != 0])
    dy = rng.randint(-margin, margin)
    first = values[..., margin:margin + crop_size, margin:margin + crop_size]
    second = values[
        ...,
        margin + dy:margin + dy + crop_size,
        margin + dx:margin + dx + crop_size,
    ]
    return torch.cat((first, second))


def degrade_pair(high: torch.Tensor, rng: random.Random) -> torch.Tensor:
    low = F.interpolate(
        high,
        scale_factor=0.5,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    levels = rng.choice((63, 95, 127, 191, 255))
    return (low * levels).round().div(levels)


def sample_pairs(groups, batch_size, crop_size, device, rng):
    high_pairs = []
    weights = [item[2] for item in groups]
    while len(high_pairs) < batch_size:
        _name, paths, _weight = rng.choices(groups, weights=weights, k=1)[0]
        try:
            image = load_rgb(paths[rng.randrange(len(paths))])
            high_pairs.append(translated_pair(image, crop_size, rng, device))
        except Exception:
            continue
    high = torch.stack(high_pairs)
    low = torch.stack([degrade_pair(pair, rng) for pair in high])
    return low, high


def temporal_loss(output: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output_delta = output[:, 1] - output[:, 0]
    high_delta = high[:, 1] - high[:, 0]
    delta = charbonnier(output_delta, high_delta)
    static = high_delta.abs().mean(dim=1, keepdim=True) < 0.01
    error = (output_delta - high_delta).abs()
    static_error = error.masked_select(static.expand_as(error))
    static_loss = static_error.mean() if static_error.numel() else error.mean() * 0
    return delta, static_loss


def edge_psnr(output: torch.Tensor, high: torch.Tensor) -> float:
    output_x = output[..., :, 1:] - output[..., :, :-1]
    high_x = high[..., :, 1:] - high[..., :, :-1]
    output_y = output[..., 1:, :] - output[..., :-1, :]
    high_y = high[..., 1:, :] - high[..., :-1, :]
    mse = (F.mse_loss(output_x.float(), high_x.float()) + F.mse_loss(
        output_y.float(), high_y.float()
    )) / 2
    return float((-10 * torch.log10(mse.clamp_min(1e-12))).item())


@torch.no_grad()
def validate(student, teacher, samples, crop_size, device, seed):
    student.eval()
    scores = {
        "psnr_vs_ground_truth": [],
        "edge_psnr_vs_ground_truth": [],
        "psnr_vs_teacher": [],
        "temporal_delta_psnr": [],
        "static_delta_mae": [],
    }
    for index, (_group, path) in enumerate(samples):
        rng = random.Random(seed + index)
        high = translated_pair(load_rgb(path), crop_size, rng, device).unsqueeze(0)
        low = degrade_pair(high[0], rng).unsqueeze(0)
        flat_low = low.flatten(0, 1)
        flat_high = high.flatten(0, 1)
        output = student(flat_low).reshape_as(high)
        teacher_output = teacher(flat_low).reconstruction.clamp(0, 1).reshape_as(high)
        scores["psnr_vs_ground_truth"].append(psnr(output, high))
        scores["edge_psnr_vs_ground_truth"].append(edge_psnr(output, high))
        scores["psnr_vs_teacher"].append(psnr(output, teacher_output))
        output_delta = output[:, 1] - output[:, 0]
        high_delta = high[:, 1] - high[:, 0]
        scores["temporal_delta_psnr"].append(psnr(output_delta, high_delta))
        static = high_delta.abs().mean(dim=1, keepdim=True) < 0.01
        error = (output_delta - high_delta).abs()
        selected = error.masked_select(static.expand_as(error))
        scores["static_delta_mae"].append(float(
            (selected.mean() if selected.numel() else error.mean()).item()
        ))
    student.train()
    return {name: float(np.mean(values)) for name, values in scores.items()}


def gate_failure(step: int, delta: dict[str, float]) -> str | None:
    spatial = (
        "psnr_vs_ground_truth",
        "edge_psnr_vs_ground_truth",
        "psnr_vs_teacher",
    )
    if step == 250:
        if any(delta[name] < -0.02 for name in spatial):
            return "step-250 spatial metric degraded by more than 0.02 dB"
        if delta["temporal_delta_psnr"] <= 0.02:
            return "step-250 temporal-delta gain did not exceed 0.02 dB"
        if delta["static_delta_mae"] >= 0:
            return "step-250 static flicker did not decrease"
    if step == 500:
        if any(delta[name] < 0 for name in spatial):
            return "step-500 requires nonnegative spatial metrics"
        if delta["temporal_delta_psnr"] <= 0 or delta["static_delta_mae"] >= 0:
            return "step-500 requires both temporal metrics to improve"
    return None


def main() -> None:
    args = parse_args()
    ground_truth_weight = 1 - args.temporal_weight - args.teacher_weight - args.edge_weight
    if ground_truth_weight <= 0:
        raise ValueError("loss weights must leave positive ground-truth weight")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    paths = discover_images(args.data)
    groups = grouped_paths(paths, args.data)
    training_by_group, validation_paths = split_stratified(groups, 16, 1, args.seed)
    training_groups = sampling_groups(training_by_group)

    parent = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    parent_config = parent["manifest"]["student"]
    if parent_config["color_mode"] != "rgb":
        raise ValueError("temporal student requires an RGB parent")
    student = RealtimeLumaStudent(RealtimeStudentConfig(
        channels=args.channels,
        blocks=int(parent_config["blocks"]),
        color_mode="rgb",
    )).to(args.device)
    initialize_wider_rgb_from_rgb(student, parent["model"])
    experiment = {
        "format": "backdropkit-realtime-temporal-ui-student-v1",
        "parent": str(args.initialize_from.resolve()),
        "parent_sha256": hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
        "student": {
            "channels": args.channels,
            "blocks": student.config.blocks,
            "parameters": student.parameter_count(),
            "color_mode": "rgb",
        },
        "training": {
            "steps": args.steps,
            "batch_size_pairs": args.batch_size,
            "crop_size": args.crop_size,
            "learning_rate": args.learning_rate,
            "loss_weights": {
                "ground_truth": ground_truth_weight,
                "teacher": args.teacher_weight,
                "edge": args.edge_weight,
                "temporal": args.temporal_weight,
            },
            "motion": "same-source translated adjacent UI crops",
            "seed": args.seed,
        },
        "training_images": sum(len(value) for value in training_by_group.values()),
        "validation_pairs": len(validation_paths),
    }
    if args.inspect_only:
        print(json.dumps(experiment, indent=2))
        return

    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    ).to(args.device).eval()
    teacher.requires_grad_(False)
    baseline = validate(
        student, teacher, validation_paths, args.crop_size, args.device,
        args.seed + 1_000_000,
    )
    experiment["baseline"] = baseline
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(experiment, indent=2) + "\n")
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    started = time.perf_counter()
    with (args.out / "train.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            low, high = sample_pairs(
                training_groups, args.batch_size, args.crop_size, args.device, rng
            )
            flat_low = low.flatten(0, 1)
            flat_high = high.flatten(0, 1)
            with torch.no_grad():
                teacher_output = teacher(flat_low).reconstruction.clamp(0, 1)
            output = student(flat_low)
            ground_truth = charbonnier(output, flat_high)
            teacher_term = charbonnier(output, teacher_output)
            edges = edge_loss(output, flat_high)
            temporal, static = temporal_loss(
                output.reshape_as(high),
                high,
            )
            loss = (
                ground_truth_weight * ground_truth
                + args.teacher_weight * teacher_term
                + args.edge_weight * edges
                + args.temporal_weight * (temporal + static)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.validate_every == 0 or step == args.steps:
                metrics = validate(
                    student, teacher, validation_paths, args.crop_size, args.device,
                    args.seed + 1_000_000,
                )
                delta = {name: metrics[name] - baseline[name] for name in baseline}
                record = {
                    "event": "validation",
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    "loss": float(loss.item()),
                    "ground_truth_loss": float(ground_truth.item()),
                    "teacher_loss": float(teacher_term.item()),
                    "edge_loss": float(edges.item()),
                    "temporal_loss": float(temporal.item()),
                    "static_loss": float(static.item()),
                    "metrics": metrics,
                    "delta_from_baseline": delta,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                failure = gate_failure(step, delta)
                torch.save({
                    "format": "backdropkit-realtime-student-checkpoint-v1",
                    "model": student.state_dict(),
                    "manifest": experiment,
                    "step": step,
                    "metrics": metrics,
                }, args.out / f"checkpoint-{step}.pt")
                if failure:
                    event = {"event": "sealed_gate_failed", "step": step, "reason": failure}
                    log.write(json.dumps(event) + "\n")
                    log.flush()
                    print(json.dumps(event, sort_keys=True), flush=True)
                    break


if __name__ == "__main__":
    main()
