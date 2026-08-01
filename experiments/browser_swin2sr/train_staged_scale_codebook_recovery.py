#!/usr/bin/env python3
"""Collapse a recovered mixed-bit scale bridge gradually into a q1 endpoint."""

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
from run_state import atomic_torch_save, capture_run_state
from scale_codebook_qat import (
    scale_codebook_parameter_group,
    scale_codebook_summary,
)
from scale_codebook_state import load_scale_codebook_student
from staged_scale_codebook import (
    STAGED_SCHEDULE_PROFILES,
    add_staged_collapse,
    bridge_summary,
    freeze_hard_bridge,
    set_staged_collapse_progress,
)
from staged_scale_codebook_state import (
    STAGED_CHECKPOINT_FORMAT,
    load_staged_scale_codebook_student,
)
from surface_model_state import pretrained
from surface_quantization import surface_parameter_groups
from train_scale_codebook_recovery import (
    gradient_receipt,
    parse_mapping,
    scale_hard_shadow_backward,
    task_loss,
)
from train_swin2sr_qat import VALIDATION_FILES, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--target-bits", action="append", required=True)
    parser.add_argument("--master-learning-rate", type=float, required=True)
    parser.add_argument("--center-learning-rate", type=float, required=True)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--hard-shadow-every", type=int, default=2)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.05)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.25)
    parser.add_argument("--schedule-profile", default="late_io_v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def checkpoint_payload(
    args,
    student,
    optimizer,
    bridge_payload,
    surfaces,
    bridge_codebooks,
    collapse,
    config,
    collapse_step,
    metrics,
    run_state,
) -> dict:
    collapse_by_name = {surface.name: surface for surface in collapse}
    return {
        "format": STAGED_CHECKPOINT_FORMAT,
        "bridge_checkpoint": str(args.bridge.resolve()),
        "bridge_checkpoint_sha256": file_sha256(args.bridge),
        "bridge_scale_step": int(bridge_payload["scale_step"]),
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "collapse_step": collapse_step,
        "step": int(bridge_payload["step"]) + collapse_step,
        "metrics": metrics,
        "run_state": run_state,
        "quantization": bridge_payload["quantization"],
        "surface_training": bridge_payload["surface_training"],
        "scale_codebook": bridge_payload["scale_codebook"],
        "staged_scale_codebook": config,
        "bridge_summary": bridge_summary(bridge_codebooks),
        "surfaces": {
            f"{surface.name}.weight": {
                "category": surface.category,
                "group_size": surface.quantizer.group_size,
                "method": surface.quantizer.method,
                "rotation": surface.quantizer.rotation,
                "rotation_block": surface.quantizer.rotation_block,
                "target_bits": collapse_by_name[surface.name].target_bits,
            }
            for surface in surfaces
        },
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.validate_every, args.checkpoint_every) < 1:
        raise ValueError("steps and receipt intervals must be positive")
    if args.master_learning_rate <= 0 or args.center_learning_rate <= 0:
        raise ValueError("staged-collapse learning rates must be positive")
    if not 0 <= args.hard_shadow_weight_start <= args.hard_shadow_weight_end:
        raise ValueError("scale hard-shadow weights are inconsistent")
    if args.schedule_profile not in STAGED_SCHEDULE_PROFILES:
        raise ValueError(f"unknown staged-collapse schedule: {args.schedule_profile}")
    target_bits = parse_mapping(args.target_bits)
    resume_payload = None
    start_collapse_step = 0
    if args.resume:
        (
            student,
            surfaces,
            bridge_codebooks,
            collapse,
            resume_payload,
            bridge_payload,
            _parent_payload,
        ) = load_staged_scale_codebook_student(args.resume, args.device)
        expected_bridge = Path(resume_payload["bridge_checkpoint"]).resolve()
        if args.bridge.resolve() != expected_bridge:
            raise ValueError("resume checkpoint belongs to a different bridge path")
        if file_sha256(args.bridge) != resume_payload["bridge_checkpoint_sha256"]:
            raise ValueError("resume checkpoint bridge identity changed")
        start_collapse_step = int(resume_payload["collapse_step"])
    else:
        (
            student,
            surfaces,
            bridge_codebooks,
            bridge_payload,
            _parent_payload,
        ) = load_scale_codebook_student(args.bridge, args.device)
        freeze_hard_bridge(bridge_codebooks)
        collapse = add_staged_collapse(
            surfaces,
            bridge_codebooks,
            target_bits_by_category=target_bits,
        )
    if int(bridge_payload["scale_step"]) != int(
        bridge_payload["scale_codebook"]["steps"]
    ):
        raise ValueError("staged collapse requires a completed scale bridge")
    categories = {surface.category for surface in surfaces}
    if set(target_bits) != categories:
        raise ValueError(f"target bits must cover exactly {sorted(categories)}")
    if any(
        surface.target_bits != target_bits[surface.category]
        for surface in collapse
    ):
        raise ValueError("resume staged-collapse bit allocation differs from the CLI")
    training = bridge_payload["surface_training"]
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    crop_size = int(training["crop_size"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    groups = surface_parameter_groups(student, surfaces)
    bridge_parameters = freeze_hard_bridge(bridge_codebooks)
    center_parameters = scale_codebook_parameter_group(collapse)
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
    restore_payload = resume_payload or bridge_payload
    epoch, image_cursor, exact_resume = sampler.restore(
        restore_payload,
        augmentation_rng,
        (int(bridge_payload["step"]) + start_collapse_step) * batch_size,
    )
    if not exact_resume:
        raise RuntimeError("staged collapse requires exact bridge data resume")
    config = {
        "format": "backdropkit-swin2sr-staged-scale-codebook-training-v1",
        "target_bits_by_category": target_bits,
        "bridge_target_bits_by_category": bridge_payload["scale_codebook"][
            "target_bits_by_category"
        ],
        "steps": args.steps,
        "master_learning_rate": args.master_learning_rate,
        "center_learning_rate": args.center_learning_rate,
        "optimizer": "adamw",
        "hard_shadow_every": args.hard_shadow_every,
        "hard_shadow_weight_start": args.hard_shadow_weight_start,
        "hard_shadow_weight_end": args.hard_shadow_weight_end,
        "schedule_profile": args.schedule_profile,
        "sampler": sampler.metadata,
    }
    if resume_payload is not None and resume_payload["staged_scale_codebook"] != config:
        raise ValueError("resume staged-collapse training configuration changed")
    if start_collapse_step >= args.steps and not args.inspect_only:
        raise ValueError("resume checkpoint already reached the requested step count")
    if args.inspect_only:
        print(json.dumps({
            "bridge_sha256": file_sha256(args.bridge),
            "resume_sha256": file_sha256(args.resume) if args.resume else None,
            "start_collapse_step": start_collapse_step,
            "exact_data_resume": exact_resume,
            "bridge": bridge_summary(bridge_codebooks),
            "collapse": scale_codebook_summary(collapse),
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
        signum: signal.signal(
            signum,
            lambda value, _frame: stop_signal.__setitem__(0, value),
        )
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
                "bridge_checkpoint": str(args.bridge),
                "bridge_checkpoint_sha256": file_sha256(args.bridge),
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "resume_checkpoint_sha256": (
                    file_sha256(args.resume) if args.resume else None
                ),
                "start_collapse_step": start_collapse_step,
                "exact_data_resume": exact_resume,
                "bridge": bridge_summary(bridge_codebooks),
                "baseline": metrics,
                "config": config,
            }) + "\n")
            log.flush()
            for collapse_step in range(start_collapse_step + 1, args.steps + 1):
                set_staged_collapse_progress(
                    collapse,
                    collapse_step / args.steps,
                    args.schedule_profile,
                )
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
                    collapse,
                    training,
                    scale_step=collapse_step,
                    every=args.hard_shadow_every,
                    weight_start=args.hard_shadow_weight_start,
                    weight_end=args.hard_shadow_weight_end,
                )
                should_validate = (
                    collapse_step == 1
                    or collapse_step % args.validate_every == 0
                    or collapse_step == args.steps
                )
                gradients = None
                if should_validate:
                    gradients = {
                        "matrix_masters": gradient_receipt(groups["matrix_masters"]),
                        "convolution_masters": gradient_receipt(
                            groups["convolution_masters"]
                        ),
                        "corrections": gradient_receipt(groups["corrections"]),
                        "bridge_centers": gradient_receipt(bridge_parameters),
                        "collapse_centers": gradient_receipt(center_parameters),
                    }
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "collapse_step": collapse_step,
                        "progress": collapse_step / args.steps,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        "bridge": bridge_summary(bridge_codebooks),
                        "collapse": scale_codebook_summary(collapse),
                        "gradients_before_clip": gradients,
                        **{name: value.item() for name, value in losses.items()},
                        **shadow,
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                if (
                    collapse_step % args.checkpoint_every == 0
                    or collapse_step == args.steps
                ):
                    atomic_torch_save(
                        checkpoint_payload(
                            args,
                            student,
                            optimizer,
                            bridge_payload,
                            surfaces,
                            bridge_codebooks,
                            collapse,
                            config,
                            collapse_step,
                            metrics,
                            capture_run_state(
                                augmentation_rng,
                                epoch,
                                image_cursor,
                                time.perf_counter() - started,
                            ),
                        ),
                        args.out / f"checkpoint-collapse-{collapse_step}.pt",
                    )
                if stop_signal[0] is not None:
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
