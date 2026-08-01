#!/usr/bin/env python3
"""Train a fresh or recovered Swin2SR onto the 97.24% ternary surface."""

from __future__ import annotations

import json
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline import degrade, fixed_validation, load_rgb, to_tensor, ui_crop
from full_surface_cli import parse_args, validate_args
from full_surface_optimizer import (
    build_full_surface_optimizer,
    optimizer_config,
)
from optimizer_controls import attach_bonsai_callbacks
from qat_objective import distillation_losses
from recovery_controls import (
    localized_feature_loss,
    recovery_loss_weights,
)
from recovery_data import BalancedSourceSampler
from run_state import capture_run_state
from surface_artifact_state import hard_surface_state_dict
from surface_curriculum import hard_shadow_backward
from surface_model_state import load_student, pretrained
from surface_quantization import (
    set_surface_progress,
    surface_parameter_groups,
)
from surface_run_state import save_checkpoint
from train_swin2sr_qat import (
    VALIDATION_FILES,
    validate,
)


def train(
    args,
    payload,
    student,
    surfaces,
    group_size,
    start_step,
    base_step,
    resume,
    hard_core,
):
    teacher = pretrained(args.device).eval()
    teacher.requires_grad_(False)
    groups = surface_parameter_groups(student, surfaces)
    if args.optimizer == "bonsai_qat8_muon":
        attach_bonsai_callbacks([
            (
                surface.module.parametrizations.weight.original,
                surface.quantizer,
            )
            for surface in surfaces
            if surface.module.weight.ndim == 2
        ])
    optimizer = build_full_surface_optimizer(
        groups,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        convolution_learning_rate_multiplier=(
            args.convolution_learning_rate_multiplier
        ),
        correction_learning_rate_multiplier=(
            args.correction_learning_rate_multiplier
        ),
    )
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    weights = recovery_loss_weights(args.loss_recipe)
    config = {
        "format": "backdropkit-swin2sr-full-surface-v1",
        "base_checkpoint": (
            payload["surface_training"].get("base_checkpoint")
            if resume
            else str(args.base_checkpoint) if args.base_checkpoint else None
        ),
        "resume_checkpoint": str(args.base_checkpoint) if resume else None,
        "initialization": (
            payload.get("surface_training", {}).get("initialization")
            if resume
            else args.initialization
        ),
        "base_global_step": base_step,
        "hard_core_from_start": hard_core,
        "scope": "all_matrix_and_conv",
        "group_size": group_size,
        "surface_group_sizes": dict(
            sorted((getattr(args, "surface_group_sizes", {}) or {}).items())
        ),
        "convolution_learning_rate_multiplier": (
            args.convolution_learning_rate_multiplier
        ),
        "correction_learning_rate_multiplier": (
            args.correction_learning_rate_multiplier
        ),
        "quantizer_method": next(
            surface.quantizer.method
            for surface in surfaces
            if surface.module.weight.ndim == 2
        ),
        "conv_quantizer_method": next(
            surface.quantizer.method
            for surface in surfaces
            if surface.module.weight.ndim == 4
        ),
        "conv_grouping": args.conv_grouping,
        "schedule_profile": args.schedule_profile,
        "core_rotation": payload["quantization"].get("rotation", "none"),
        "schedule_steps": args.schedule_steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "loss_recipe": args.loss_recipe,
        "loss_weights": weights,
        "hard_shadow_weight_start": args.hard_shadow_weight_start,
        "hard_shadow_weight_end": args.hard_shadow_weight_end,
        "hard_shadow_every": args.hard_shadow_every,
        "sampler": sampler.metadata,
        "optimizer": optimizer_config(
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            convolution_learning_rate_multiplier=(
                args.convolution_learning_rate_multiplier
            ),
            correction_learning_rate_multiplier=(
                args.correction_learning_rate_multiplier
            ),
            reset_from_base=not resume,
        ),
    }
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    baseline = validate(student, teacher, validation)
    augmentation_rng = random.Random(args.seed)
    if resume:
        epoch, image_cursor, exact_resume = sampler.restore(
            payload,
            augmentation_rng,
            start_step * args.batch_size,
        )
    else:
        epoch, image_cursor, exact_resume = sampler.restore(
            {},
            augmentation_rng,
            0,
        )
    epoch_paths = sampler.paths(epoch)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.jsonl"
    started = time.perf_counter()
    stop_signal = [None]

    def request_stop(signum, _frame):
        stop_signal[0] = signum

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    metrics = baseline
    best_psnr = float("-inf")
    try:
        with log_path.open("a" if resume else "w") as log:
            log.write(json.dumps({
                "event": "start",
                **config,
                "start_surface_step": start_step,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "training_images": len(sampler),
                "exact_data_resume": exact_resume,
                "baseline": baseline,
                "surface_categories": {
                    category: sum(
                        surface.module.weight.numel()
                        for surface in surfaces
                        if surface.category == category
                    )
                    for category in sorted({surface.category for surface in surfaces})
                },
                "parameter_groups": {
                    name: sum(parameter.numel() for parameter in parameters)
                    for name, parameters in groups.items()
                },
            }) + "\n")
            log.flush()
            for surface_step in range(start_step + 1, args.steps + 1):
                progress = min(1.0, surface_step / max(1, args.schedule_steps))
                set_surface_progress(
                    surfaces,
                    progress,
                    hard_core=hard_core,
                )
                low_batch = []
                high_batch = []
                for _ in range(args.batch_size):
                    if image_cursor >= len(epoch_paths):
                        epoch += 1
                        epoch_paths = sampler.paths(epoch)
                        image_cursor = 0
                    source = load_rgb(epoch_paths[image_cursor])
                    image_cursor += 1
                    high = ui_crop(source, args.crop_size, augmentation_rng)
                    low_batch.append(
                        to_tensor(degrade(high, augmentation_rng), args.device)
                    )
                    high_batch.append(to_tensor(high, args.device))
                low = torch.cat(low_batch)
                high = torch.cat(high_batch)
                with torch.no_grad():
                    teacher_output = teacher(low, output_hidden_states=True)
                student_output = student(low, output_hidden_states=True)
                reconstruction = student_output.reconstruction.clamp(0, 1)
                teacher_reconstruction = teacher_output.reconstruction.clamp(0, 1)
                losses = distillation_losses(
                    reconstruction,
                    teacher_reconstruction,
                    high,
                    (student_output.hidden_states, teacher_output.hidden_states),
                )
                losses["feature"] = localized_feature_loss(
                    student_output.hidden_states,
                    teacher_output.hidden_states,
                    None,
                    "l2" if args.loss_recipe == "teacher_heavy" else "layernorm",
                )
                loss = sum(weights[name] * losses[name] for name in weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                shadow_metrics = hard_shadow_backward(
                    student,
                    low,
                    high,
                    teacher_output,
                    teacher_reconstruction,
                    surfaces,
                    weights,
                    loss_recipe=args.loss_recipe,
                    surface_step=surface_step,
                    every=args.hard_shadow_every,
                    weight_start=args.hard_shadow_weight_start,
                    weight_end=args.hard_shadow_weight_end,
                )
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

                should_validate = (
                    surface_step == start_step + 1
                    or surface_step % args.validate_every == 0
                    or surface_step == args.steps
                )
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "surface_step": surface_step,
                        "global_step": base_step + surface_step,
                        "progress": progress,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        **{name: value.item() for name, value in losses.items()},
                        **shadow_metrics,
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    all_hard = all(
                        surface.quantizer.amount.item() >= 1
                        for surface in surfaces
                    )
                    if all_hard and metrics["psnr_vs_ground_truth"] > best_psnr:
                        best_psnr = metrics["psnr_vs_ground_truth"]
                        save_checkpoint(
                            args.out / "checkpoint-best.pt",
                            student,
                            optimizer,
                            surfaces,
                            surface_step=surface_step,
                            base_global_step=base_step,
                            metrics=metrics,
                            run_state=capture_run_state(
                                augmentation_rng,
                                epoch,
                                image_cursor,
                                time.perf_counter() - started,
                            ),
                            config=config,
                        )
                if surface_step % args.checkpoint_every == 0:
                    save_checkpoint(
                        args.out / f"checkpoint-surface-{surface_step}.pt",
                        student,
                        optimizer,
                        surfaces,
                        surface_step=surface_step,
                        base_global_step=base_step,
                        metrics=metrics,
                        run_state=capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                        config=config,
                    )
                if stop_signal[0] is not None:
                    path = args.out / f"checkpoint-interrupted-{surface_step}.pt"
                    save_checkpoint(
                        path,
                        student,
                        optimizer,
                        surfaces,
                        surface_step=surface_step,
                        base_global_step=base_step,
                        metrics=metrics,
                        run_state=capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                        config=config,
                    )
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    save_checkpoint(
        args.out / "checkpoint-final.pt",
        student,
        optimizer,
        surfaces,
        surface_step=args.steps,
        base_global_step=base_step,
        metrics=metrics,
        run_state=capture_run_state(
            augmentation_rng,
            epoch,
            image_cursor,
            time.perf_counter() - started,
        ),
        config=config,
    )
    torch.save(
        hard_surface_state_dict(student, surfaces),
        args.out / "swin2sr-full-surface-hard.pt",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.base_checkpoint:
        payload = torch.load(args.base_checkpoint, map_location=args.device)
    elif args.initialization == "fp32":
        if args.core_rotation is None:
            raise ValueError("fp32 initialization requires --core-rotation")
        payload = {
            "quantization": {
                "mode": "ternary",
                "method": args.quantizer_method or "optimal_mse",
                "group_size": args.group_size or 60,
                "rotation": args.core_rotation,
                "rotation_block": 4,
                "surface_group_sizes": args.surface_group_sizes or {},
            },
            "step": 0,
        }
    else:
        raise ValueError("recovered_core initialization requires --base-checkpoint")
    loaded = load_student(args, payload)
    student, surfaces, group_size, start_step, base_step, resume, hard_core = loaded
    if args.steps <= start_step:
        raise ValueError(
            f"--steps must exceed checkpoint surface step {start_step}"
        )
    inspection = {
        "base_checkpoint": (
            str(args.base_checkpoint) if args.base_checkpoint else None
        ),
        "initialization": (
            payload.get("surface_training", {}).get("initialization")
            if resume
            else args.initialization
        ),
        "resume": resume,
        "start_surface_step": start_step,
        "base_global_step": base_step,
        "group_size": group_size,
        "surface_group_sizes": dict(
            sorted((getattr(args, "surface_group_sizes", {}) or {}).items())
        ),
        "convolution_learning_rate_multiplier": (
            args.convolution_learning_rate_multiplier
        ),
        "correction_learning_rate_multiplier": (
            args.correction_learning_rate_multiplier
        ),
        "optimizer": args.optimizer,
        "quantizer_method": next(
            surface.quantizer.method
            for surface in surfaces
            if surface.module.weight.ndim == 2
        ),
        "conv_quantizer_method": next(
            surface.quantizer.method
            for surface in surfaces
            if surface.module.weight.ndim == 4
        ),
        "conv_grouping": args.conv_grouping,
        "schedule_profile": args.schedule_profile,
        "core_rotation": payload["quantization"].get("rotation", "none"),
        "surfaces": len(surfaces),
        "ternary_values": sum(surface.module.weight.numel() for surface in surfaces),
        "categories": sorted({surface.category for surface in surfaces}),
        "hard_core_from_start": hard_core,
    }
    if args.inspect_only:
        print(json.dumps(inspection, indent=2))
        return
    train(args, payload, *loaded)


if __name__ == "__main__":
    main()
