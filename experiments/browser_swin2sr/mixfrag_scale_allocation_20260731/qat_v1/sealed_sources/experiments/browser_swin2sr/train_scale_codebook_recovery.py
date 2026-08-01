#!/usr/bin/env python3
"""Continue a hard-ternary Swin2SR parent into learned scale codebooks."""

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
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from qat_objective import distillation_losses
from recovery_controls import localized_feature_loss, recovery_loss_weights
from recovery_data import BalancedSourceSampler
from run_state import atomic_torch_save, capture_run_state
from scale_codebook_qat import (
    add_per_surface_scale_codebooks,
    add_scale_codebooks,
    scale_codebook_parameter_group,
    scale_codebook_summary,
    set_scale_codebook_progress,
)
from scale_codebook_state import CHECKPOINT_FORMAT, load_scale_codebook_student
from surface_model_state import pretrained
from surface_quantization import surface_parameter_groups
from train_swin2sr_qat import VALIDATION_FILES, validate


def parse_mapping(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        name, separator, raw_bits = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate category bit mapping: {value}")
        bits = int(raw_bits)
        if bits < 1 or bits > 8:
            raise ValueError(f"scale bits must be between one and eight: {value}")
        result[name] = bits
    return dict(sorted(result.items()))


def load_target_map(path: Path) -> tuple[dict, dict[str, int]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "backdropkit-swin2sr-per-surface-scale-map-v1":
        raise ValueError("unsupported per-surface scale map")
    raw = payload.get("target_bits_by_surface")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("per-surface scale map is empty")
    result = {}
    for name, raw_bits in raw.items():
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("per-surface scale map has an invalid surface name")
        bits = int(raw_bits)
        if bits < 1 or bits > 8:
            raise ValueError(f"scale bits must be between one and eight: {name}")
        result[name] = bits
    return payload, dict(sorted(result.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-bits", action="append")
    target.add_argument("--target-map", type=Path)
    parser.add_argument("--master-learning-rate", type=float, required=True)
    parser.add_argument("--center-learning-rate", type=float, required=True)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--hard-shadow-every", type=int, default=2)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.05)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def task_loss(student_output, teacher_output, high, training: dict):
    reconstruction = student_output.reconstruction.clamp(0, 1)
    teacher_reconstruction = teacher_output.reconstruction.clamp(0, 1)
    losses = distillation_losses(
        reconstruction,
        teacher_reconstruction,
        high,
        (student_output.hidden_states, teacher_output.hidden_states),
        pixel_objective=training.get("pixel_objective", "charbonnier"),
        ground_truth_mse_floor=float(training.get("ground_truth_mse_floor", 3.2e-4)),
        teacher_mse_floor=float(training.get("teacher_mse_floor", 2.0e-5)),
    )
    losses["feature"] = localized_feature_loss(
        student_output.hidden_states,
        teacher_output.hidden_states,
        None,
        "l2" if training["loss_recipe"] == "teacher_heavy" else "layernorm",
    )
    weights = recovery_loss_weights(training["loss_recipe"])
    return sum(weights[name] * losses[name] for name in weights), losses


def gradient_receipt(parameters) -> dict:
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return {"parameters": 0, "l2": 0.0, "max_abs": 0.0, "finite": True}
    squared = sum(float(torch.dot(value, value).item()) for value in gradients)
    return {
        "parameters": sum(value.numel() for value in gradients),
        "l2": squared ** 0.5,
        "max_abs": max(float(value.abs().max().item()) for value in gradients),
        "finite": all(bool(torch.isfinite(value).all().item()) for value in gradients),
    }


def scale_hard_shadow_backward(
    student,
    low,
    high,
    teacher_output,
    codebooks,
    training,
    *,
    scale_step: int,
    every: int,
    weight_start: float,
    weight_end: float,
) -> dict:
    active = [
        float(surface.projection.amount.item())
        for surface in codebooks
        if 0 < float(surface.projection.amount.item()) < 1
    ]
    if not active or every <= 0 or scale_step % every:
        return {}
    transition = max(active)
    weight = weight_start + transition * (weight_end - weight_start)
    live = [float(surface.projection.amount.item()) for surface in codebooks]
    for surface, amount in zip(codebooks, live):
        surface.projection.amount.fill_(1.0 if amount > 0 else 0.0)
    try:
        shadow_output = student(low, output_hidden_states=True)
        shadow_loss, shadow_losses = task_loss(
            shadow_output, teacher_output, high, training
        )
        (weight * shadow_loss).backward()
        return {
            "scale_hard_shadow_loss": shadow_loss.item(),
            "scale_hard_shadow_weight": weight,
            "scale_hard_shadow_transition_fraction": transition,
            **{
                f"scale_hard_shadow_{name}": value.item()
                for name, value in shadow_losses.items()
            },
        }
    finally:
        for surface, amount in zip(codebooks, live):
            surface.projection.amount.fill_(amount)


def checkpoint_payload(
    args,
    student,
    optimizer,
    parent_payload,
    surfaces,
    codebooks,
    config,
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
        "step": int(parent_payload.get("step", parent_payload["surface_step"])) + scale_step,
        "metrics": metrics,
        "run_state": run_state,
        "quantization": parent_payload["quantization"],
        "surface_training": parent_payload["surface_training"],
        "scale_codebook": config,
        "surfaces": {
            f"{surface.name}.weight": {
                "category": surface.category,
                "group_size": surface.quantizer.group_size,
                "method": surface.quantizer.method,
                "rotation": surface.quantizer.rotation,
                "rotation_block": surface.quantizer.rotation_block,
                "target_bits": next(
                    item.target_bits for item in codebooks if item.name == surface.name
                ),
            }
            for surface in surfaces
        },
    }


def main() -> None:
    args = parse_args()
    if min(
        args.steps,
        args.validate_every,
        args.checkpoint_every,
    ) < 1:
        raise ValueError("steps and receipt intervals must be positive")
    if args.master_learning_rate <= 0 or args.center_learning_rate <= 0:
        raise ValueError("scale-codebook learning rates must be positive")
    if not 0 <= args.hard_shadow_weight_start <= args.hard_shadow_weight_end:
        raise ValueError("scale hard-shadow weights are inconsistent")
    target_map_payload = None
    if args.target_map:
        target_map_payload, target_bits = load_target_map(args.target_map)
        allocation_scope = "per_surface"
    else:
        target_bits = parse_mapping(args.target_bits)
        allocation_scope = "category"
    resume_payload = None
    start_scale_step = 0
    if args.resume:
        (
            student,
            surfaces,
            codebooks,
            resume_payload,
            parent_payload,
        ) = load_scale_codebook_student(args.resume, args.device)
        expected_parent = Path(resume_payload["parent_checkpoint"]).resolve()
        if args.parent.resolve() != expected_parent:
            raise ValueError("resume checkpoint belongs to a different parent path")
        if file_sha256(args.parent) != resume_payload["parent_checkpoint_sha256"]:
            raise ValueError("resume checkpoint parent identity changed")
        start_scale_step = int(resume_payload["scale_step"])
    else:
        student, surfaces, parent_payload = load_checkpoint_student(
            args.parent, args.device
        )
        for surface in surfaces:
            surface.quantizer.amount.fill_(1.0)
        if allocation_scope == "per_surface":
            codebooks = add_per_surface_scale_codebooks(
                surfaces,
                target_bits_by_surface=target_bits,
            )
        else:
            codebooks = add_scale_codebooks(
                surfaces,
                target_bits_by_category=target_bits,
            )
    if allocation_scope == "per_surface":
        surface_names = {surface.name for surface in surfaces}
        if set(target_bits) != surface_names:
            raise ValueError("target map must cover the exact parent surface set")
        if any(surface.target_bits != target_bits[surface.name] for surface in codebooks):
            raise ValueError("resume per-surface bit allocation differs from the CLI")
    else:
        categories = {surface.category for surface in surfaces}
        if set(target_bits) != categories:
            raise ValueError(f"target bits must cover exactly {sorted(categories)}")
        if any(surface.target_bits != target_bits[surface.category] for surface in codebooks):
            raise ValueError("resume category bit allocation differs from the CLI")
    training = parent_payload["surface_training"]
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    crop_size = int(training["crop_size"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    groups = surface_parameter_groups(student, surfaces)
    center_parameters = scale_codebook_parameter_group(codebooks)
    optimizer = torch.optim.AdamW([
        {"params": groups["matrix_masters"], "lr": args.master_learning_rate},
        {
            "params": groups["convolution_masters"],
            "lr": args.master_learning_rate
            * float(training["convolution_learning_rate_multiplier"]),
        },
        {
            "params": groups["corrections"],
            "lr": args.master_learning_rate
            * float(training["correction_learning_rate_multiplier"]),
            "weight_decay": 0.0,
        },
        {
            "params": center_parameters,
            "lr": args.center_learning_rate,
            "weight_decay": 0.0,
        },
    ], weight_decay=1e-4)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    sampler = BalancedSourceSampler.from_roots(args.data, seed)
    augmentation_rng = random.Random(seed)
    restore_payload = resume_payload or parent_payload
    epoch, image_cursor, exact_resume = sampler.restore(
        restore_payload,
        augmentation_rng,
        (
            int(parent_payload["surface_step"]) + start_scale_step
        ) * batch_size,
    )
    if not exact_resume:
        raise RuntimeError("scale-codebook recovery requires exact parent data resume")
    config = {
        "format": "backdropkit-swin2sr-scale-codebook-training-v1",
        "allocation_scope": allocation_scope,
        "steps": args.steps,
        "master_learning_rate": args.master_learning_rate,
        "center_learning_rate": args.center_learning_rate,
        "optimizer": "adamw",
        "hard_shadow_every": args.hard_shadow_every,
        "hard_shadow_weight_start": args.hard_shadow_weight_start,
        "hard_shadow_weight_end": args.hard_shadow_weight_end,
        "sampler": sampler.metadata,
    }
    if allocation_scope == "per_surface":
        config["target_bits_by_surface"] = target_bits
        config["target_map"] = {
            "path": str(args.target_map.resolve()),
            "sha256": file_sha256(args.target_map),
            "format": target_map_payload["format"],
        }
    else:
        config["target_bits_by_category"] = target_bits
    if resume_payload is not None and resume_payload["scale_codebook"] != config:
        raise ValueError("resume scale-codebook training configuration changed")
    if start_scale_step >= args.steps and not args.inspect_only:
        raise ValueError("resume checkpoint already reached the requested step count")
    if args.inspect_only:
        print(json.dumps({
            "parent_sha256": file_sha256(args.parent),
            "resume_sha256": file_sha256(args.resume) if args.resume else None,
            "start_scale_step": start_scale_step,
            "exact_data_resume": exact_resume,
            "scale_codebooks": scale_codebook_summary(codebooks),
            "config": config,
        }, indent=2))
        return
    teacher = pretrained(args.device).eval()
    teacher.requires_grad_(False)
    validation = fixed_validation(
        args.validation_data, VALIDATION_FILES, crop_size, args.device
    )
    metrics = validate(student, teacher, validation)
    args.out.mkdir(parents=True, exist_ok=True)
    epoch_paths = sampler.paths(epoch)
    stop_signal = [None]
    previous_handlers = {
        signum: signal.signal(signum, lambda value, _frame: stop_signal.__setitem__(0, value))
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    elapsed_before_resume = float(
        restore_payload.get("run_state", {}).get("elapsed_seconds", 0.0)
    )
    started = time.perf_counter() - elapsed_before_resume
    try:
        log_mode = "a" if resume_payload is not None else "w"
        with (args.out / "train.jsonl").open(log_mode) as log:
            log.write(json.dumps({
                "event": "resume" if resume_payload is not None else "start",
                "parent_checkpoint": str(args.parent),
                "parent_checkpoint_sha256": file_sha256(args.parent),
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "resume_checkpoint_sha256": (
                    file_sha256(args.resume) if args.resume else None
                ),
                "start_scale_step": start_scale_step,
                "exact_data_resume": exact_resume,
                "baseline": metrics,
                "config": config,
            }) + "\n")
            log.flush()
            for scale_step in range(start_scale_step + 1, args.steps + 1):
                set_scale_codebook_progress(codebooks, scale_step / args.steps)
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
                    low_batch.append(to_tensor(degrade(high, augmentation_rng), args.device))
                    high_batch.append(to_tensor(high, args.device))
                low = torch.cat(low_batch)
                high = torch.cat(high_batch)
                with torch.no_grad():
                    teacher_output = teacher(low, output_hidden_states=True)
                student_output = student(low, output_hidden_states=True)
                loss, losses = task_loss(student_output, teacher_output, high, training)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                shadow = scale_hard_shadow_backward(
                    student,
                    low,
                    high,
                    teacher_output,
                    codebooks,
                    training,
                    scale_step=scale_step,
                    every=args.hard_shadow_every,
                    weight_start=args.hard_shadow_weight_start,
                    weight_end=args.hard_shadow_weight_end,
                )
                should_validate = (
                    scale_step == 1
                    or scale_step % args.validate_every == 0
                    or scale_step == args.steps
                )
                gradients = None
                if should_validate:
                    gradients = {
                        "matrix_masters": gradient_receipt(groups["matrix_masters"]),
                        "convolution_masters": gradient_receipt(
                            groups["convolution_masters"]
                        ),
                        "corrections": gradient_receipt(groups["corrections"]),
                        "scale_centers": gradient_receipt(center_parameters),
                    }
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "scale_step": scale_step,
                        "progress": scale_step / args.steps,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        "scale_summary": scale_codebook_summary(codebooks),
                        "gradients_before_clip": gradients,
                        **{name: value.item() for name, value in losses.items()},
                        **shadow,
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                if scale_step % args.checkpoint_every == 0 or scale_step == args.steps:
                    atomic_torch_save(
                        checkpoint_payload(
                            args,
                            student,
                            optimizer,
                            parent_payload,
                            surfaces,
                            codebooks,
                            config,
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
                if stop_signal[0] is not None:
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
