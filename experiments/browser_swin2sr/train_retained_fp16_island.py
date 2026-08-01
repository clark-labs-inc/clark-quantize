#!/usr/bin/env python3
"""Train one exact retained-precision island from a ternary Swin2SR parent."""

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
from full_surface_provenance import file_sha256
from recovery_data import BalancedSourceSampler
from retained_precision_islands import (
    ISLAND_SPECS,
    boost_int7_boundary_gradients,
    retained_precision_configs,
    storage_projection,
)
from retained_precision_state import (
    CHECKPOINT_FORMAT as FORMAT,
    load_retained_precision_student as load_training_state,
)
from run_state import atomic_torch_save, capture_run_state
from surface_model_state import pretrained
from train_scale_codebook_recovery import gradient_receipt, task_loss
from train_swin2sr_qat import VALIDATION_FILES, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--island", choices=sorted(ISLAND_SPECS), required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "int7", "ternary"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Physical batch override; defaults to the parent training batch.",
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--boundary-boost", type=float, default=0.0)
    parser.add_argument("--boundary-temperature", type=float, default=0.1)
    parser.add_argument(
        "--int7-scale-method",
        choices=("max", "mse_iterative"),
        default="max",
    )
    parser.add_argument("--int7-scale-blend", type=float, default=1.0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def receipt_without_parameters(initialization: dict) -> dict:
    return {
        name: value
        for name, value in initialization.items()
        if name != "parameters"
    }


def merged_retained_precision_history(parent_payload: dict, config: dict) -> list[dict]:
    history = retained_precision_configs(parent_payload)
    history.append(config)
    merged = []
    positions = {}
    for retained in history:
        island = retained["island"]
        if island in positions:
            merged[positions[island]] = retained
        else:
            positions[island] = len(merged)
            merged.append(retained)
    return merged


def experiment_config(
    args: argparse.Namespace,
    training: dict,
    sampler: BalancedSourceSampler,
) -> dict:
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(training["batch_size"])
    )
    return {
        "format": FORMAT,
        "island": args.island,
        "precision": args.precision,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "boundary_boost": float(getattr(args, "boundary_boost", 0.0)),
        "boundary_temperature": float(
            getattr(args, "boundary_temperature", 0.1)
        ),
        "int7_scale_method": getattr(args, "int7_scale_method", "max"),
        "int7_scale_blend": float(getattr(args, "int7_scale_blend", 1.0)),
        "batch_size": batch_size,
        "crop_size": int(training["crop_size"]),
        "loss_recipe": training["loss_recipe"],
        "pixel_objective": training.get("pixel_objective", "charbonnier"),
        "initialization": {
            "fp16": "hard_ternary_then_fp16_roundtrip_ste",
            "int7": "parent_then_symmetric_int7_per_output_ste",
            "ternary": "hard_ternary_control",
        }[args.precision],
        "trainable_scope": "declared_island_weights_only",
        "initialize_only": bool(getattr(args, "initialize_only", False)),
        "sampler": sampler.metadata,
    }


def checkpoint_payload(
    args: argparse.Namespace,
    student,
    optimizer,
    parent_payload: dict,
    config: dict,
    initialization: dict,
    island_step: int,
    metrics: dict,
    run_state: dict,
) -> dict:
    return {
        "format": FORMAT,
        "parent_checkpoint": str(args.parent.resolve()),
        "parent_checkpoint_sha256": file_sha256(args.parent),
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "island_step": island_step,
        "step": int(parent_payload["step"]) + island_step,
        "metrics": metrics,
        "run_state": run_state,
        "quantization": parent_payload["quantization"],
        "surface_training": parent_payload["surface_training"],
        "recovery": {"sampler": config["sampler"]},
        "retained_fp16_training": config,
        "retained_precision_history": merged_retained_precision_history(
            parent_payload,
            config,
        ),
        "initialization": receipt_without_parameters(initialization),
    }


def validate_args(args: argparse.Namespace) -> None:
    if min(args.steps, args.validate_every, args.checkpoint_every) < 1:
        raise ValueError("steps and receipt intervals must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    boundary_boost = float(getattr(args, "boundary_boost", 0.0))
    boundary_temperature = float(getattr(args, "boundary_temperature", 0.1))
    precision = getattr(args, "precision", "fp16")
    if boundary_boost < 0 or boundary_temperature <= 0:
        raise ValueError("invalid int7 boundary preconditioner")
    if precision != "int7" and boundary_boost != 0:
        raise ValueError("boundary boost is only valid for int7 training")
    if not 0 <= float(getattr(args, "int7_scale_blend", 1.0)) <= 1:
        raise ValueError("int7 scale blend must be between zero and one")
    if getattr(args, "inspect_only", False) and getattr(
        args,
        "initialize_only",
        False,
    ):
        raise ValueError("inspect-only and initialize-only are mutually exclusive")
    if getattr(args, "initialize_only", False) and args.resume is not None:
        raise ValueError("initialize-only cannot resume a checkpoint")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    student, surfaces, parent_payload, initialization, resume_payload = (
        load_training_state(
            args.parent,
            args.island,
            args.precision,
            args.device,
            args.resume,
            args.int7_scale_method,
            args.int7_scale_blend,
        )
    )
    training = parent_payload["surface_training"]
    crop_size = int(training["crop_size"])
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    augmentation_rng = random.Random(args.seed)
    config = experiment_config(args, training, sampler)
    batch_size = int(config["batch_size"])
    start_step = int(resume_payload["island_step"]) if resume_payload else 0
    if resume_payload is not None:
        if resume_payload["retained_fp16_training"] != config:
            raise ValueError("resume retained-precision configuration changed")
        epoch, image_cursor, exact_resume = sampler.restore(
            resume_payload,
            augmentation_rng,
            start_step * batch_size,
        )
        if not exact_resume:
            raise RuntimeError("retained-precision training requires exact resume")
    else:
        epoch, image_cursor, exact_resume = 0, 0, True
    parameters = initialization["parameters"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=config["weight_decay"],
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    projection = storage_projection(surfaces, ISLAND_SPECS[args.island])
    inspection = {
        "format": FORMAT,
        "parent": str(args.parent.resolve()),
        "parent_sha256": file_sha256(args.parent),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "resume_sha256": file_sha256(args.resume) if args.resume else None,
        "start_step": start_step,
        "exact_data_resume": exact_resume,
        "initialization": receipt_without_parameters(initialization),
        "storage_projection": projection,
        "config": config,
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
        crop_size,
        args.device,
    )
    metrics = validate(student, teacher, validation)
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
                "baseline": metrics,
            }) + "\n")
            log.flush()
            if args.initialize_only:
                checkpoint = args.out / "checkpoint-island-0.pt"
                atomic_torch_save(
                    checkpoint_payload(
                        args,
                        student,
                        optimizer,
                        parent_payload,
                        config,
                        initialization,
                        0,
                        metrics,
                        capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                    ),
                    checkpoint,
                )
                print(json.dumps({
                    "event": "initialized",
                    "checkpoint": str(checkpoint),
                    **metrics,
                }, sort_keys=True), flush=True)
                return
            for island_step in range(start_step + 1, args.steps + 1):
                low_batch = []
                high_batch = []
                for _ in range(batch_size):
                    if image_cursor >= len(epoch_paths):
                        epoch += 1
                        epoch_paths = sampler.paths(epoch)
                        image_cursor = 0
                    source = load_rgb(epoch_paths[image_cursor])
                    image_cursor += 1
                    high = ui_crop(source, crop_size, augmentation_rng)
                    low_batch.append(
                        to_tensor(degrade(high, augmentation_rng), args.device)
                    )
                    high_batch.append(to_tensor(high, args.device))
                low = torch.cat(low_batch)
                high = torch.cat(high_batch)
                with torch.no_grad():
                    teacher_output = teacher(low, output_hidden_states=True)
                student_output = student(low, output_hidden_states=True)
                loss, losses = task_loss(
                    student_output,
                    teacher_output,
                    high,
                    training,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                should_validate = (
                    island_step == 1
                    or island_step % args.validate_every == 0
                    or island_step == args.steps
                )
                raw_gradients = (
                    gradient_receipt(parameters) if should_validate else None
                )
                boundary_receipt = None
                if config["boundary_boost"] > 0:
                    boundary_receipt = boost_int7_boundary_gradients(
                        parameters,
                        config["boundary_boost"],
                        config["boundary_temperature"],
                    )
                gradients = (
                    gradient_receipt(parameters) if should_validate else None
                )
                torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"])
                optimizer.step()
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "island_step": island_step,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        "gradients_before_clip": gradients,
                        "raw_gradients_before_boundary_boost": raw_gradients,
                        "boundary_preconditioner": boundary_receipt,
                        **{name: value.item() for name, value in losses.items()},
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                if (
                    island_step % args.checkpoint_every == 0
                    or island_step == args.steps
                ):
                    atomic_torch_save(
                        checkpoint_payload(
                            args,
                            student,
                            optimizer,
                            parent_payload,
                            config,
                            initialization,
                            island_step,
                            metrics,
                            capture_run_state(
                                augmentation_rng,
                                epoch,
                                image_cursor,
                                time.perf_counter() - started,
                            ),
                        ),
                        args.out / f"checkpoint-island-{island_step}.pt",
                    )
                if stop_signal[0] is not None:
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
