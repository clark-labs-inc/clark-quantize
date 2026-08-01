#!/usr/bin/env python3
"""Winner-only recovery training for BackdropKit's ternary Swin2SR.

This runner starts from a completed factorial checkpoint, keeps the model hard
ternary from the first recovery step, resets optimizer state intentionally,
and ablates only preregistered compensation policies.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifact_state import hard_state_dict
from data_pipeline import (
    degrade,
    fixed_validation,
    load_rgb,
    to_tensor,
    ui_crop,
)
from island_policy import POLICIES
from optimizer_controls import attach_bonsai_callbacks
from qat_objective import distillation_losses
from recovery_controls import (
    add_recovery_parametrizations,
    build_recovery_optimizer,
    localized_feature_loss,
    recovery_parameter_groups,
    recovery_loss_weights,
    scalar_storage_bytes,
)
from recovery_data import BalancedSourceSampler
from recovery_run_state import (
    checkpoint_recovery_config,
    recovery_offsets,
)
from run_state import (
    atomic_torch_save,
    capture_run_state,
)
from train_swin2sr_qat import (
    MODEL_ID,
    MODEL_REVISION,
    VALIDATION_FILES,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "bonsai_qat8", "bonsai_qat8_muon"),
        default="bonsai_qat8_muon",
    )
    parser.add_argument("--island-policy", choices=POLICIES, default="frozen")
    parser.add_argument("--scalar-decomposition", action="store_true")
    parser.add_argument(
        "--loss-recipe",
        choices=("balanced", "teacher_heavy"),
        default="balanced",
    )
    parser.add_argument("--localized-stage-steps", type=int, default=100)
    parser.add_argument("--muon-every", type=int, default=16)
    parser.add_argument("--muon-steps", type=int, default=2)
    parser.add_argument("--scalar-lr-multiplier", type=float, default=100.0)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--reset-data-stream", action="store_true")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def load_recovery_model(args, payload):
    quantization = payload["quantization"]
    if quantization["mode"] != "ternary":
        raise ValueError("recovery requires a ternary base checkpoint")
    group_size = args.group_size or int(quantization["group_size"])
    student = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(args.device).train()
    quantizers = add_recovery_parametrizations(
        student,
        group_size=group_size,
        method=quantization["method"],
        rotation=quantization["rotation"],
        rotation_block=int(quantization["rotation_block"]),
        scalar_decomposition=args.scalar_decomposition,
    )
    incompatible = student.load_state_dict(payload["model"], strict=False)
    scalar_keys = set()
    if args.scalar_decomposition:
        for name, _quantizer, _module in quantizers:
            prefix = f"{name}.parametrizations.weight.0"
            scalar_keys.update({
                f"{prefix}.log_scale_multiplier",
                f"{prefix}.offset_multiplier",
            })
    present_scalar_keys = scalar_keys & set(payload["model"])
    if present_scalar_keys and present_scalar_keys != scalar_keys:
        raise ValueError("base checkpoint has a partial scalar-decomposition state")
    expected_missing = scalar_keys - present_scalar_keys
    if set(incompatible.missing_keys) != expected_missing:
        raise ValueError(
            "unexpected missing recovery state: "
            f"{sorted(set(incompatible.missing_keys) - expected_missing)}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected base checkpoint state: {incompatible.unexpected_keys}"
        )
    for _name, quantizer, _module in quantizers:
        quantizer.set_progress(1.0)
    return student, quantizers, group_size


def checkpoint_payload(
    student,
    optimizer,
    quantizers,
    *,
    step,
    metrics,
    run_state,
    config,
):
    quantizer = quantizers[0][1]
    return {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_config": config["optimizer"],
        "step": step,
        "metrics": metrics,
        "quantized_modules": [name for name, _quantizer, _module in quantizers],
        "quantization": {
            "mode": quantizer.mode,
            "method": quantizer.method,
            "group_size": quantizer.group_size,
            "rotation": quantizer.rotation,
            "rotation_block": quantizer.rotation_block,
        },
        "run_state": run_state,
        "recovery": checkpoint_recovery_config(config, step),
    }


def save_checkpoint(path, *args, **kwargs):
    atomic_torch_save(checkpoint_payload(*args, **kwargs), path)


def active_stage(local_step: int, stage_steps: int) -> int | None:
    if stage_steps <= 0 or local_step > stage_steps * 4:
        return None
    return (local_step - 1) // stage_steps


def train(args, payload, student, quantizers, group_size, groups) -> None:
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(args.device).eval()
    teacher.requires_grad_(False)
    attach_bonsai_callbacks([
        (module.parametrizations.weight.original, quantizer)
        for _name, quantizer, module in quantizers
    ])
    optimizer = build_recovery_optimizer(
        args.optimizer,
        groups,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        muon_every=args.muon_every,
        muon_steps=args.muon_steps,
        scalar_lr_multiplier=args.scalar_lr_multiplier,
    )
    if args.resume_optimizer:
        optimizer.load_state_dict(payload["optimizer"])
    optimizer_config = {
        "name": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "muon_every": args.muon_every,
        "muon_steps": args.muon_steps,
        "scalar_lr_multiplier": args.scalar_lr_multiplier,
        "reset_from_base": not args.resume_optimizer,
    }
    initial_global_step, previous_recovery_steps = recovery_offsets(payload)
    config = {
        "format": "backdropkit-swin2sr-recovery-v1",
        "base_checkpoint": str(args.base_checkpoint),
        "base_step": int(payload["step"]),
        "initial_global_step": initial_global_step,
        "recovery_steps_before": previous_recovery_steps,
        "optimizer": optimizer_config,
        "island_policy": args.island_policy,
        "scalar_decomposition": args.scalar_decomposition,
        "localized_stage_steps": args.localized_stage_steps,
        "loss_recipe": args.loss_recipe,
        "data_stream_reset_requested": args.reset_data_stream,
        "loss_weights": recovery_loss_weights(args.loss_recipe),
        "feature_normalization": (
            "l2" if args.loss_recipe == "teacher_heavy" else "layernorm"
        ),
        "group_size": group_size,
        "scalar_storage_bytes": (
            scalar_storage_bytes(len(quantizers))
            if args.scalar_decomposition
            else 0
        ),
    }
    sampler = BalancedSourceSampler.from_roots(args.data, args.seed)
    config["sampler"] = sampler.metadata
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    baseline = validate(student, teacher, validation)
    augmentation_rng = random.Random(args.seed)
    base_examples = previous_recovery_steps * args.batch_size
    epoch, image_cursor, exact_resume = sampler.restore(
        payload,
        augmentation_rng,
        base_examples,
        reset_data_stream=args.reset_data_stream,
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
    best_psnr = baseline["psnr_vs_ground_truth"]
    metrics = baseline
    loss_weights = recovery_loss_weights(args.loss_recipe)
    try:
        with log_path.open("w") as log:
            log.write(json.dumps({
                "event": "start",
                **config,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "training_images": len(sampler),
                "training_roots": [str(path) for path in args.data],
                "exact_data_resume": exact_resume,
                "baseline": baseline,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in student.parameters()
                    if parameter.requires_grad
                ),
                "parameter_groups": {
                    name: sum(parameter.numel() for parameter in parameters)
                    for name, parameters in groups.items()
                },
            }) + "\n")
            log.flush()
            for local_step in range(1, args.steps + 1):
                low_batch = []
                high_batch = []
                for _ in range(args.batch_size):
                    if image_cursor >= len(epoch_paths):
                        epoch += 1
                        epoch_paths = sampler.paths(epoch)
                        image_cursor = 0
                    source = load_rgb(epoch_paths[image_cursor])
                    image_cursor += 1
                    high_image = ui_crop(source, args.crop_size, augmentation_rng)
                    low_batch.append(
                        to_tensor(degrade(high_image, augmentation_rng), args.device)
                    )
                    high_batch.append(to_tensor(high_image, args.device))
                low = torch.cat(low_batch)
                high = torch.cat(high_batch)
                with torch.no_grad():
                    teacher_output = teacher(low, output_hidden_states=True)
                student_output = student(low, output_hidden_states=True)
                reconstruction = student_output.reconstruction.clamp(0, 1)
                teacher_reconstruction = teacher_output.reconstruction.clamp(0, 1)
                losses = distillation_losses(
                    reconstruction, teacher_reconstruction, high,
                    (student_output.hidden_states, teacher_output.hidden_states),
                    pixel_objective="charbonnier", ground_truth_mse_floor=3.2e-4,
                    teacher_mse_floor=2e-5,
                )
                recovery_step = previous_recovery_steps + local_step
                stage = active_stage(
                    recovery_step,
                    args.localized_stage_steps,
                )
                losses["feature"] = localized_feature_loss(
                    student_output.hidden_states,
                    teacher_output.hidden_states,
                    stage,
                    config["feature_normalization"],
                )
                losses["redistribution"] = torch.stack([
                    module.input_shift.square().mean()
                    for _name, _quantizer, module in quantizers
                    if module.bias is not None
                ]).mean()
                loss = (
                    loss_weights["teacher"] * losses["teacher"]
                    + loss_weights["ground_truth"] * losses["ground_truth"]
                    + loss_weights["edge"] * losses["edge"]
                    + loss_weights["feature"] * losses["feature"]
                    + 1e-4 * losses["redistribution"]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

                global_step = int(payload["step"]) + local_step
                should_validate = (
                    local_step == 1
                    or local_step % args.validate_every == 0
                    or local_step == args.steps
                )
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "local_step": local_step,
                        "global_step": global_step,
                        "recovery_step": recovery_step,
                        "examples_seen": recovery_step * args.batch_size,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        "active_stage": stage,
                        **{name: value.item() for name, value in losses.items()},
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if metrics["psnr_vs_ground_truth"] > best_psnr:
                        best_psnr = metrics["psnr_vs_ground_truth"]
                        save_checkpoint(
                            args.out / "checkpoint-best.pt",
                            student,
                            optimizer,
                            quantizers,
                            step=global_step,
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
                            hard_state_dict(student, quantizers),
                            args.out / "swin2sr-best-hard.pt",
                        )
                if local_step % args.checkpoint_every == 0:
                    save_checkpoint(
                        args.out / f"checkpoint-{global_step}.pt",
                        student,
                        optimizer,
                        quantizers,
                        step=global_step,
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
                    name = f"checkpoint-interrupted-{global_step}.pt"
                    save_checkpoint(
                        args.out / name,
                        student,
                        optimizer,
                        quantizers,
                        step=global_step,
                        metrics=metrics,
                        run_state=capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                        config=config,
                    )
                    log.write(json.dumps({
                        "event": "interrupted",
                        "signal": signal.Signals(stop_signal[0]).name,
                        "checkpoint": name,
                    }) + "\n")
                    log.flush()
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    global_step = int(payload["step"]) + args.steps
    save_checkpoint(
        args.out / "checkpoint-final.pt",
        student,
        optimizer,
        quantizers,
        step=global_step,
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
        hard_state_dict(student, quantizers),
        args.out / "swin2sr-ternary-hard.pt",
    )
    (args.out / "summary.json").write_text(json.dumps({
        **config,
        "steps": args.steps,
        "final_global_step": global_step,
        "baseline": baseline,
        "final": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.crop_size % 16:
        raise ValueError("steps must be positive and crop-size divisible by 16")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = torch.load(args.base_checkpoint, map_location=args.device)
    student, quantizers, group_size = load_recovery_model(args, payload)
    groups = recovery_parameter_groups(
        student,
        quantizers,
        args.island_policy,
    )
    inspection = {
        "base_checkpoint": str(args.base_checkpoint),
        "base_step": int(payload["step"]),
        "group_size": group_size,
        "rotation": payload["quantization"]["rotation"],
        "scalar_decomposition": args.scalar_decomposition,
        "loss_recipe": args.loss_recipe,
        "island_policy": args.island_policy,
        "quantized_modules": len(quantizers),
        "parameter_groups": {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in groups.items()
        },
    }
    if args.inspect_only:
        print(json.dumps(inspection, indent=2))
        return
    train(args, payload, student, quantizers, group_size, groups)


if __name__ == "__main__":
    main()
