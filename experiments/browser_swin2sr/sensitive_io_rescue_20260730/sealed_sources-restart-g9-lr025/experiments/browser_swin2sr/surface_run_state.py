"""Checkpoint and resume contracts for full-surface Swin2SR training."""

from __future__ import annotations

from argparse import Namespace

from run_state import atomic_torch_save


def validate_resume_config(
    args: Namespace,
    payload: dict,
    group_size: int,
) -> None:
    previous = payload["surface_training"]
    requested = {
        "group_size": group_size,
        "schedule_steps": args.schedule_steps,
        "loss_recipe": args.loss_recipe,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "conv_quantizer_method": args.conv_quantizer_method,
        "conv_grouping": args.conv_grouping,
        "schedule_profile": args.schedule_profile,
        "hard_shadow_weight_start": args.hard_shadow_weight_start,
        "hard_shadow_weight_end": args.hard_shadow_weight_end,
        "hard_shadow_every": args.hard_shadow_every,
    }
    for name, value in requested.items():
        if previous.get(name) != value:
            raise ValueError(
                f"cannot resume with changed {name}: "
                f"{previous.get(name)!r} -> {value!r}"
            )
    previous_surface_group_sizes = dict(
        previous.get("surface_group_sizes", {}) or {}
    )
    requested_surface_group_sizes = dict(
        getattr(args, "surface_group_sizes", {}) or {}
    )
    if previous_surface_group_sizes != requested_surface_group_sizes:
        raise ValueError(
            "cannot resume with changed surface_group_sizes: "
            f"{previous_surface_group_sizes!r} -> "
            f"{requested_surface_group_sizes!r}"
        )
    previous_conv_lr_multiplier = float(
        previous.get("convolution_learning_rate_multiplier", 0.25)
    )
    requested_conv_lr_multiplier = float(
        getattr(args, "convolution_learning_rate_multiplier", 0.25)
    )
    if previous_conv_lr_multiplier != requested_conv_lr_multiplier:
        raise ValueError(
            "cannot resume with changed convolution_learning_rate_multiplier: "
            f"{previous_conv_lr_multiplier!r} -> "
            f"{requested_conv_lr_multiplier!r}"
        )
    previous_lr = previous["optimizer"]["learning_rate"]
    if previous_lr != args.learning_rate:
        raise ValueError(
            "cannot resume with changed learning_rate: "
            f"{previous_lr!r} -> {args.learning_rate!r}"
        )


def checkpoint_payload(
    student,
    optimizer,
    surfaces,
    *,
    surface_step,
    base_global_step,
    metrics,
    run_state,
    config,
):
    first = next(
        surface.quantizer
        for surface in surfaces
        if surface.module.weight.ndim == 2
    )
    first_conv = next(
        surface.quantizer
        for surface in surfaces
        if surface.module.weight.ndim == 4
    )
    surface_group_sizes = dict(sorted(
        (config.get("surface_group_sizes", {}) or {}).items()
    ))
    return {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_config": config["optimizer"],
        "step": base_global_step + surface_step,
        "surface_step": surface_step,
        "metrics": metrics,
        "quantized_modules": [surface.name for surface in surfaces],
        "quantization": {
            "mode": "ternary",
            "method": first.method,
            "group_size": first.group_size,
            "conv_method": first_conv.method,
            "conv_grouping": config["conv_grouping"],
            "schedule_profile": config["schedule_profile"],
            "rotation": config["core_rotation"],
            "rotation_block": first.rotation_block,
            "scope": "all_matrix_and_conv",
            "surface_group_sizes": surface_group_sizes,
        },
        "run_state": run_state,
        "surface_training": config,
    }


def save_checkpoint(path, *args, **kwargs) -> None:
    atomic_torch_save(checkpoint_payload(*args, **kwargs), path)
