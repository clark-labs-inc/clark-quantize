#!/usr/bin/env python3
"""Data-driven ternary QAT for the browser Swin2SR model.

The student keeps high-precision master weights but executes core attention and
MLP projections through hard groupwise ternary weights with a straight-through
gradient. A full-precision teacher supplies output and intermediate-feature
targets over UI-heavy crops.
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
from torch.nn.utils import parametrize
from transformers import Swin2SRForImageSuperResolution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline import (
    degrade,
    discover_images,
    fixed_validation,
    load_rgb,
    to_tensor,
    ui_crop,
)
from artifact_state import hard_state_dict
from optimizer_controls import (
    OPTIMIZERS,
    apply_proximal_pull,
    attach_bonsai_callbacks,
    build_optimizer,
    configure_resumed_optimizer,
    scheduled_prox_rho,
)
from qat_objective import distillation_losses, psnr
from quantizer_controls import ProgressiveLowBit
from rotation_control import block_hadamard
from run_state import (
    capture_run_state,
    epoch_order,
    restore_run_state,
    save_training_checkpoint,
)

MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
DEFAULT_DATA = Path("/Users/stan/Documents/git/clark-desktop/promos")
VALIDATION_FILES = (
    "source-ui-composer-current.png",
    "source-ui-chat-current.png",
    "source-ui-research-current.png",
    "clark-code-ad-parallel-research-vertical-poster.png",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Training image root. Repeat to mix corpora; defaults to Clark promos.",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=DEFAULT_DATA,
        help="Held-out product screenshots. Never sampled for training.",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/browser_swin2sr/qat_out"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument(
        "--quantizer",
        choices=("optimal_mse", "absmean", "paretoq_seq"),
        default="optimal_mse",
    )
    parser.add_argument("--rotation", choices=("none", "hadamard"), default="none")
    parser.add_argument("--rotation-block", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--prox-rho", type=float, default=0.05)
    parser.add_argument("--muon-every", type=int, default=16)
    parser.add_argument("--muon-steps", type=int, default=2)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()

def add_ternary_parametrizations(
    model: torch.nn.Module,
    mode: str,
    group_size: int,
    method: str = "optimal_mse",
    rotation: str = "none",
    rotation_block: int = 4,
) -> list[tuple[str, ProgressiveLowBit, torch.nn.Linear]]:
    quantizers = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "continuous_position_bias_mlp" in name:
            continue
        if ".attention." in name:
            schedule = (0.00, 0.12)
        elif ".intermediate." in name or ".output.dense" in name:
            schedule = (0.08, 0.28)
        else:
            continue
        if rotation == "hadamard":
            module.weight.data.copy_(
                block_hadamard(module.weight.data, rotation_block)
            )
        quantizer = ProgressiveLowBit(
            mode,
            method,
            group_size,
            *schedule,
            module.weight,
            rotation,
            rotation_block,
        )
        parametrize.register_parametrization(module, "weight", quantizer)
        if module.bias is not None:
            module.register_parameter(
                "input_shift",
                torch.nn.Parameter(torch.zeros(module.in_features, device=module.weight.device)),
            )

        if rotation == "hadamard" or module.bias is not None:
            def prepare_inputs(target, inputs):
                values = inputs[0]
                quantizer = target.parametrizations.weight[0]
                if quantizer.rotation == "hadamard":
                    values = block_hadamard(values, quantizer.rotation_block)
                if hasattr(target, "input_shift"):
                    values = values + target.input_shift
                return (values, *inputs[1:])

            module.register_forward_pre_hook(prepare_inputs)
        quantizers.append((name, quantizer, module))
    return quantizers

def freeze_high_precision_islands(student, quantizers) -> int:
    student.requires_grad_(False)
    for _, _, module in quantizers:
        module.parametrizations.weight.original.requires_grad_(True)
        for parameter in module.parametrizations.weight[0].parameters():
            parameter.requires_grad_(True)
        if module.bias is not None:
            module.bias.requires_grad_(True)
            module.input_shift.requires_grad_(True)
    return sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)

@torch.no_grad()
def validate(student, teacher, samples) -> dict[str, float]:
    student.eval()
    teacher.eval()
    ground_truth_scores = []
    teacher_scores = []
    for low, high in samples:
        teacher_output = teacher(low).reconstruction.clamp(0, 1)
        student_output = student(low).reconstruction.clamp(0, 1)
        ground_truth_scores.append(psnr(student_output, high))
        teacher_scores.append(psnr(student_output, teacher_output))
    student.train()
    return {
        "psnr_vs_ground_truth": float(np.mean(ground_truth_scores)),
        "psnr_vs_teacher": float(np.mean(teacher_scores)),
    }

def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    augmentation_rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    training_roots = args.data or [DEFAULT_DATA]
    image_paths = discover_images(training_roots)
    images = image_paths
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(args.device).eval()
    student = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(args.device).train()
    teacher.requires_grad_(False)
    quantizers = add_ternary_parametrizations(
        student,
        args.mode,
        args.group_size,
        args.quantizer,
        args.rotation,
        args.rotation_block,
    )
    trainable_parameters = freeze_high_precision_islands(student, quantizers)
    master_quantizers = [
        (module.parametrizations.weight.original, quantizer)
        for _, quantizer, module in quantizers
    ]
    masters = [master for master, _ in master_quantizers]
    master_ids = {id(master) for master in masters}
    other_parameters = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad and id(parameter) not in master_ids
    ]
    attach_bonsai_callbacks(master_quantizers)
    optimizer = build_optimizer(
        args.optimizer,
        masters,
        other_parameters,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        group_size=args.group_size,
        muon_every=args.muon_every,
        muon_steps=args.muon_steps,
    )
    optimizer_config = {
        "name": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "prox_rho": args.prox_rho,
        "muon_every": args.muon_every,
        "muon_steps": args.muon_steps,
    }
    schedule_steps = args.schedule_steps or args.steps
    start_step = 1
    epoch = 0
    image_cursor = 0
    elapsed_before_resume = 0.0
    exact_resume = True
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        student.load_state_dict(checkpoint["model"], strict=True)
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            configure_resumed_optimizer(
                optimizer,
                checkpoint.get("optimizer_config"),
                optimizer_config,
            )
        start_step = int(checkpoint["step"]) + 1
        epoch, image_cursor, elapsed_before_resume, exact_resume = restore_run_state(
            checkpoint,
            augmentation_rng,
            examples_seen=int(checkpoint["step"]) * args.batch_size,
            image_count=len(images),
        )
    validation = fixed_validation(
        args.validation_data,
        VALIDATION_FILES,
        args.crop_size,
        args.device,
    )
    baseline = validate(student, teacher, validation)
    log_path = args.out / "train.jsonl"
    started = time.perf_counter() - elapsed_before_resume
    epoch_paths = epoch_order(images, args.seed, epoch)
    stop_signal = [None]

    def request_stop(signum, _frame) -> None:
        stop_signal[0] = signum

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        with log_path.open("a" if args.resume else "w") as log:
            log.write(json.dumps({
            "event": "resume" if args.resume else "start",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": args.device,
            "mode": args.mode,
            "quantizer": args.quantizer,
            "rotation": args.rotation,
            "rotation_block": args.rotation_block,
            "optimizer": args.optimizer,
            "prox_rho": args.prox_rho,
            "muon_every": args.muon_every,
            "muon_steps": args.muon_steps,
            "reset_optimizer": args.reset_optimizer,
            "steps": args.steps,
            "schedule_steps": schedule_steps,
            "batch_size": args.batch_size,
            "images": len(images),
            "training_roots": [str(path) for path in training_roots],
            "validation_root": str(args.validation_data),
            "quantized_modules": len(quantizers),
            "trainable_parameters": trainable_parameters,
            "start_step": start_step,
            "exact_resume": exact_resume,
            "baseline": baseline,
            }) + "\n")
            log.flush()
            metrics = baseline
            best_hard_psnr = float("-inf")
            for step in range(start_step, args.steps + 1):
                progress = min(1.0, step / schedule_steps)
                for _, quantizer, _ in quantizers:
                    quantizer.set_progress(progress)
                low_batch = []
                high_batch = []
                for _ in range(args.batch_size):
                    if image_cursor >= len(epoch_paths):
                        epoch += 1
                        epoch_paths = epoch_order(images, args.seed, epoch)
                        image_cursor = 0
                    source = load_rgb(epoch_paths[image_cursor])
                    image_cursor += 1
                    high_image = ui_crop(source, args.crop_size, augmentation_rng)
                    low_image = degrade(high_image, augmentation_rng)
                    low_batch.append(to_tensor(low_image, args.device))
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
                    teacher_mse_floor=2e-5)
                losses["redistribution"] = torch.stack([
                        module.input_shift.square().mean()
                        for _, _, module in quantizers
                        if module.bias is not None
                    ]).mean()
                loss = (
                    0.55 * losses["teacher"]
                    + 0.20 * losses["ground_truth"]
                    + 0.15 * losses["edge"]
                    + 0.10 * losses["feature"]
                    + 1e-4 * losses["redistribution"]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                rho = scheduled_prox_rho(
                    args.optimizer,
                    progress,
                    maximum=args.prox_rho,
                )
                if hasattr(optimizer, "set_prox_rho"):
                    optimizer.set_prox_rho(rho)
                optimizer.step()
                if args.optimizer == "adamw_prox":
                    apply_proximal_pull(master_quantizers, rho)

                should_validate = step == 1 or step % args.validate_every == 0 or step == args.steps
                if should_validate:
                    metrics = validate(student, teacher, validation)
                    record = {
                        "event": "validation",
                        "step": step,
                        "examples_seen": step * args.batch_size,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - started,
                        "attention_amount": quantizers[0][1].amount.item(),
                        "mlp_amount": quantizers[-1][1].amount.item(),
                        "prox_rho": rho,
                        **{name: value.item() for name, value in losses.items()},
                        **metrics,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if quantizers[-1][1].amount.item() >= 1.0 and metrics["psnr_vs_ground_truth"] > best_hard_psnr:
                        best_hard_psnr = metrics["psnr_vs_ground_truth"]
                        save_training_checkpoint(
                            args.out / "checkpoint-best-hard.pt",
                            student,
                            optimizer,
                            quantizers,
                            step,
                            metrics,
                            capture_run_state(
                                augmentation_rng,
                                epoch,
                                image_cursor,
                                time.perf_counter() - started,
                            ),
                            optimizer_config,
                        )
                        torch.save(
                            hard_state_dict(student, quantizers),
                            args.out / "swin2sr-best-hard.pt",
                        )
                if step % args.checkpoint_every == 0:
                    save_training_checkpoint(
                        args.out / f"checkpoint-{step}.pt",
                        student,
                        optimizer,
                        quantizers,
                        step,
                        metrics,
                        capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                        optimizer_config,
                    )
                if stop_signal[0] is not None:
                    checkpoint_name = f"checkpoint-interrupted-{step}.pt"
                    save_training_checkpoint(
                        args.out / checkpoint_name,
                        student,
                        optimizer,
                        quantizers,
                        step,
                        metrics,
                        capture_run_state(
                            augmentation_rng,
                            epoch,
                            image_cursor,
                            time.perf_counter() - started,
                        ),
                        optimizer_config,
                    )
                    log.write(json.dumps({
                        "event": "interrupted",
                        "step": step,
                        "signal": signal.Signals(stop_signal[0]).name,
                        "checkpoint": checkpoint_name,
                    }) + "\n")
                    log.flush()
                    raise SystemExit(128 + stop_signal[0])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    final_metrics = validate(student, teacher, validation)
    save_training_checkpoint(
        args.out / "checkpoint-final.pt",
        student,
        optimizer,
        quantizers,
        args.steps,
        final_metrics,
        capture_run_state(
            augmentation_rng,
            epoch,
            image_cursor,
            time.perf_counter() - started,
        ),
        optimizer_config,
    )
    torch.save(hard_state_dict(student, quantizers), args.out / "swin2sr-ternary-hard.pt")
    (args.out / "summary.json").write_text(json.dumps({
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": args.device,
        "mode": args.mode,
        "quantizer": args.quantizer,
        "rotation": args.rotation,
        "rotation_block": args.rotation_block,
        "optimizer": args.optimizer,
        "prox_rho": args.prox_rho,
        "muon_every": args.muon_every,
        "muon_steps": args.muon_steps,
        "reset_optimizer": args.reset_optimizer,
        "steps": args.steps,
        "schedule_steps": schedule_steps,
        "batch_size": args.batch_size,
        "examples_seen": args.steps * args.batch_size,
        "training_images": len(images),
        "training_roots": [str(path) for path in training_roots],
        "validation_root": str(args.validation_data),
        "quantized_modules": len(quantizers),
        "baseline": baseline,
        "final": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")

if __name__ == "__main__":
    main()
