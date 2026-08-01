#!/usr/bin/env python3
"""Recover Swin2SR quality by training only 240 deployed int7 scales."""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from data_pipeline import degrade, fixed_validation, load_rgb, to_tensor, ui_crop
from fixed_int7_scale_recovery import (
    CHECKPOINT_FORMAT,
    clamp_scale_deltas,
    load_fixed_int7_scale_student,
    scale_receipt,
    trainable_scale_parameters,
)
from full_surface_provenance import file_sha256
from recovery_data import BalancedSourceSampler
from run_state import atomic_torch_save, capture_run_state
from surface_model_state import pretrained
from train_scale_codebook_recovery import gradient_receipt, task_loss
from train_swin2sr_qat import VALIDATION_FILES, validate


def sealed_gate_failure(scale_step: int, delta: dict[str, float]) -> str | None:
    ground_truth = delta["psnr_vs_ground_truth"]
    teacher = delta["psnr_vs_teacher"]
    if scale_step == 250:
        if ground_truth < -0.002 or teacher < -0.002:
            return "step-250 metric degraded by more than 0.002 dB"
        if max(ground_truth, teacher) <= 0:
            return "step-250 produced no positive metric"
    if scale_step == 500 and (ground_truth < 0 or teacher < 0):
        return "step-500 requires both metrics to be nonnegative"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--lloyd-mask", type=int, required=True)
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


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.lloyd_mask < 16:
        raise ValueError("Lloyd mask must be between 0 and 15")
    if min(args.steps, args.batch_size, args.validate_every, args.checkpoint_every) < 1:
        raise ValueError("steps, batch size, and receipt intervals must be positive")
    if args.learning_rate <= 0 or args.anchor_weight < 0:
        raise ValueError("learning rate must be positive and anchor nonnegative")


def config(args: argparse.Namespace, sampler: BalancedSourceSampler, training: dict) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "lloyd_mask": args.lloyd_mask,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "crop_size": int(training["crop_size"]),
        "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "trainable_scope": "four_rstb_int7_per_output_fp16_scales_only",
        "trainable_values": 240,
        "int7_codes_frozen": True,
        "loss_recipe": training["loss_recipe"],
        "pixel_objective": training.get("pixel_objective", "charbonnier"),
        "sampler": sampler.metadata,
    }


def checkpoint_payload(
    args,
    student,
    optimizer,
    parent_payload,
    experiment,
    scale_step,
    metrics,
    run_state,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "parent_checkpoint": str(args.parent.resolve()),
        "parent_checkpoint_sha256": file_sha256(args.parent),
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scale_step": scale_step,
        "step": int(parent_payload["step"]) + scale_step,
        "metrics": metrics,
        "run_state": run_state,
        "quantization": parent_payload["quantization"],
        "surface_training": parent_payload["surface_training"],
        "fixed_int7_scale_training": experiment,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    student, _surfaces, fixed, parent_payload, resume_payload = (
        load_fixed_int7_scale_student(
            args.parent,
            args.lloyd_mask,
            args.device,
            args.resume,
        )
    )
    training = parent_payload["surface_training"]
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    experiment = config(args, sampler, training)
    if resume_payload is not None:
        if resume_payload["parent_checkpoint_sha256"] != file_sha256(args.parent):
            raise ValueError("fixed-scale resume parent identity changed")
        if resume_payload["fixed_int7_scale_training"] != experiment:
            raise ValueError("fixed-scale resume configuration changed")
    parameters = trainable_scale_parameters(fixed)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    start_step = int(resume_payload["scale_step"]) if resume_payload else 0
    augmentation_rng = random.Random(args.seed)
    if resume_payload is not None:
        epoch, image_cursor, exact_resume = sampler.restore(
            resume_payload,
            augmentation_rng,
            start_step * args.batch_size,
        )
    else:
        epoch, image_cursor, exact_resume = 0, 0, True
    if not exact_resume:
        raise RuntimeError("fixed-scale training requires exact data resume")
    inspection = {
        "parent_checkpoint": str(args.parent.resolve()),
        "parent_checkpoint_sha256": file_sha256(args.parent),
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "resume_checkpoint_sha256": file_sha256(args.resume) if args.resume else None,
        "start_step": start_step,
        "exact_data_resume": exact_resume,
        "scales": scale_receipt(fixed),
        "config": experiment,
    }
    if args.inspect_only:
        print(json.dumps(inspection, indent=2))
        return
    if start_step >= args.steps:
        raise ValueError("resume checkpoint already reached requested steps")

    teacher = pretrained(args.device).eval()
    teacher.requires_grad_(False)
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        int(training["crop_size"]),
        args.device,
    )
    metrics = validate(student, teacher, validation)
    baseline = dict(metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    epoch_paths = sampler.paths(epoch)
    elapsed_before_resume = float(
        (resume_payload or {}).get("run_state", {}).get("elapsed_seconds", 0.0)
    )
    started = time.perf_counter() - elapsed_before_resume
    stop_signal = [None]
    previous_handlers = {
        signum: signal.signal(
            signum,
            lambda value, _frame: stop_signal.__setitem__(0, value),
        )
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        log_mode = "a" if resume_payload else "w"
        with (args.out / "train.jsonl").open(log_mode) as log:
            log.write(json.dumps({
                "event": "resume" if resume_payload else "start",
                **inspection,
                "baseline": baseline,
            }) + "\n")
            log.flush()
            for scale_step in range(start_step + 1, args.steps + 1):
                gate_failure = None
                low_batch = []
                high_batch = []
                for _ in range(args.batch_size):
                    if image_cursor >= len(epoch_paths):
                        epoch += 1
                        epoch_paths = sampler.paths(epoch)
                        image_cursor = 0
                    source = load_rgb(epoch_paths[image_cursor])
                    image_cursor += 1
                    high = ui_crop(source, int(training["crop_size"]), augmentation_rng)
                    low_batch.append(to_tensor(degrade(high, augmentation_rng), args.device))
                    high_batch.append(to_tensor(high, args.device))
                low = torch.cat(low_batch)
                high = torch.cat(high_batch)
                with torch.no_grad():
                    teacher_output = teacher(low, output_hidden_states=True)
                student_output = student(low, output_hidden_states=True)
                task, losses = task_loss(student_output, teacher_output, high, training)
                anchor = torch.stack([
                    parameter.square().mean()
                    for parameter in parameters
                ]).mean()
                loss = task + args.anchor_weight * anchor
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                should_validate = (
                    scale_step == 1
                    or scale_step % args.validate_every == 0
                    or scale_step == args.steps
                )
                gradients = gradient_receipt(parameters) if should_validate else None
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                clamp_scale_deltas(fixed)
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "scale_step": scale_step,
                        "loss": float(loss.item()),
                        "task_loss": float(task.item()),
                        "scale_anchor": float(anchor.item()),
                        "seconds": time.perf_counter() - started,
                        "gradients_before_clip": gradients,
                        "scales": scale_receipt(fixed),
                        "delta_from_baseline": {
                            key: float(metrics[key] - baseline[key])
                            for key in baseline
                        },
                        **{name: float(value.item()) for name, value in losses.items()},
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    gate_failure = sealed_gate_failure(
                        scale_step,
                        record["delta_from_baseline"],
                    )
                if (
                    scale_step % args.checkpoint_every == 0
                    or scale_step == args.steps
                ):
                    atomic_torch_save(
                        checkpoint_payload(
                            args,
                            student,
                            optimizer,
                            parent_payload,
                            experiment,
                            scale_step,
                            metrics,
                            capture_run_state(
                                augmentation_rng,
                                epoch,
                                image_cursor,
                                time.perf_counter() - started,
                            ),
                        ),
                        args.out / f"checkpoint-scale-{scale_step}.pt",
                    )
                if gate_failure is not None:
                    gate_record = {
                        "event": "sealed_gate_failed",
                        "scale_step": scale_step,
                        "reason": gate_failure,
                    }
                    log.write(json.dumps(gate_record) + "\n")
                    log.flush()
                    print(json.dumps(gate_record, sort_keys=True), flush=True)
                    break
                if stop_signal[0] is not None:
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
