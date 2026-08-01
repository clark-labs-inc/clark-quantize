#!/usr/bin/env python3
"""Recover G3 quality by tuning only four existing RSTB fusion biases."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data_pipeline import degrade, fixed_validation, load_rgb, to_tensor, ui_crop
from full_surface_provenance import file_sha256
from recovery_data import BalancedSourceSampler
from rstb_bias_recovery import (
    bias_delta_receipt,
    clamp_bias_deltas,
    load_bias_student,
)
from train_fixed_int7_scales import sealed_gate_failure
from train_scale_codebook_recovery import gradient_receipt, task_loss
from train_swin2sr_qat import VALIDATION_FILES, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--anchor-weight", type=float, required=True)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.validate_every, args.checkpoint_every) < 1:
        raise ValueError("step and batch arguments must be positive")
    if args.learning_rate <= 0 or args.anchor_weight < 0:
        raise ValueError("learning rate must be positive and anchor nonnegative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    student, parameters, origins = load_bias_student(args.state, args.device)
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    training = {
        "crop_size": 128,
        "loss_recipe": "balanced",
        "loss_weights": {
            "teacher": 0.55,
            "ground_truth": 0.2,
            "edge": 0.15,
            "feature": 0.1,
        },
        "pixel_objective": "psnr_aligned",
        "ground_truth_mse_floor": 0.00032,
        "teacher_mse_floor": 0.00002,
    }
    config = {
        "format": "backdropkit-swin2sr-rstb-bias-recovery-v1",
        "state": str(args.state.resolve()),
        "state_sha256": file_sha256(args.state),
        "trainable_scope": "four_existing_rstb_fusion_biases",
        "trainable_values": 240,
        "weight_codes_and_scales_frozen": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "seed": args.seed,
        "sampler": sampler.metadata,
    }
    inspection = {"config": config, "biases": bias_delta_receipt(student, origins)}
    if args.inspect_only:
        print(json.dumps(inspection, indent=2))
        return

    teacher = __import__("surface_model_state").pretrained(args.device).eval()
    teacher.requires_grad_(False)
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        training["crop_size"],
        args.device,
    )
    baseline = validate(student, teacher, validation)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    args.out.mkdir(parents=True, exist_ok=True)
    epoch = 0
    cursor = 0
    epoch_paths = sampler.paths(epoch)
    started = time.perf_counter()
    with (args.out / "train.jsonl").open("w") as log:
        start = {"event": "start", **inspection, "baseline": baseline}
        log.write(json.dumps(start) + "\n")
        log.flush()
        for step in range(1, args.steps + 1):
            low_values = []
            high_values = []
            for _ in range(args.batch_size):
                if cursor >= len(epoch_paths):
                    epoch += 1
                    epoch_paths = sampler.paths(epoch)
                    cursor = 0
                high_image = ui_crop(
                    load_rgb(epoch_paths[cursor]),
                    training["crop_size"],
                    rng,
                )
                cursor += 1
                low_values.append(to_tensor(degrade(high_image, rng), args.device))
                high_values.append(to_tensor(high_image, args.device))
            low = torch.cat(low_values)
            high = torch.cat(high_values)
            with torch.no_grad():
                teacher_output = teacher(low, output_hidden_states=True)
            student_output = student(low, output_hidden_states=True)
            task, losses = task_loss(student_output, teacher_output, high, training)
            anchor = torch.stack([
                (parameter - origins[name]).square().mean()
                for (name, parameter) in (
                    (name, dict(student.named_parameters())[name])
                    for name in origins
                )
            ]).mean()
            loss = task + args.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            should_validate = step == 1 or step % args.validate_every == 0
            gradients = gradient_receipt(parameters) if should_validate else None
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            clamp_bias_deltas(student, origins)
            gate_failure = None
            metrics = None
            if should_validate:
                metrics = validate(student, teacher, validation)
                delta = {name: metrics[name] - baseline[name] for name in baseline}
                record = {
                    "event": "validation",
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    "loss": float(loss),
                    "task_loss": float(task),
                    "anchor": float(anchor),
                    "gradients_before_clip": gradients,
                    "biases": bias_delta_receipt(student, origins),
                    "delta_from_baseline": delta,
                    **{name: float(value) for name, value in losses.items()},
                    **metrics,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                gate_failure = sealed_gate_failure(step, delta)
            if step % args.checkpoint_every == 0 or step == args.steps:
                torch.save({
                    "format": config["format"],
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "metrics": metrics,
                    "config": config,
                }, args.out / f"checkpoint-bias-{step}.pt")
            if gate_failure:
                event = {"event": "sealed_gate_failed", "step": step, "reason": gate_failure}
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(json.dumps(event, sort_keys=True), flush=True)
                break


if __name__ == "__main__":
    main()
