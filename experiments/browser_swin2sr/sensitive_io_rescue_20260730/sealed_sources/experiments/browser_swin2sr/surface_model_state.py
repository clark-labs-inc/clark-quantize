"""Model initialization contracts for full-surface Swin2SR training."""

from __future__ import annotations

import torch
from transformers import Swin2SRForImageSuperResolution

from artifact_state import hard_state_dict
from recovery_controls import add_recovery_parametrizations
from surface_quantization import (
    add_surface_parametrizations,
    apply_surface_group_sizes,
    set_surface_progress,
)
from surface_run_state import validate_resume_config
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


def pretrained(device: str):
    return Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)


def recovery_dense_state(
    payload: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    quantization = payload["quantization"]
    recovery = payload.get("recovery", {})
    model = pretrained(device)
    quantizers = add_recovery_parametrizations(
        model,
        group_size=int(quantization["group_size"]),
        method=quantization["method"],
        rotation=quantization.get("rotation", "none"),
        rotation_block=int(quantization.get("rotation_block", 4)),
        scalar_decomposition=bool(recovery.get("scalar_decomposition", False)),
    )
    model.load_state_dict(payload["model"], strict=True)
    for _name, quantizer, _module in quantizers:
        quantizer.set_progress(1.0)
    return hard_state_dict(model, quantizers)


def load_student(args, payload):
    quantization = payload["quantization"]
    resuming = quantization.get("scope") == "all_matrix_and_conv"
    group_size = args.group_size or int(quantization.get("group_size", 60))
    method = args.quantizer_method or quantization.get("method", "optimal_mse")
    conv_method = (
        getattr(args, "conv_quantizer_method", None)
        or quantization.get("conv_method")
        or method
    )
    conv_grouping = (
        getattr(args, "conv_grouping", None)
        or quantization.get("conv_grouping", "flat")
    )
    schedule_profile = (
        getattr(args, "schedule_profile", None)
        or quantization.get("schedule_profile", "joint_io_v1")
    )
    surface_group_sizes = dict(
        quantization.get("surface_group_sizes", {}) or {}
    )
    args.conv_quantizer_method = conv_method
    args.conv_grouping = conv_grouping
    args.schedule_profile = schedule_profile
    args.surface_group_sizes = surface_group_sizes
    if (
        args.quantizer_method
        and quantization.get("method")
        and args.quantizer_method != quantization["method"]
    ):
        raise ValueError(
            "quantizer method does not match the base checkpoint: "
            f"{args.quantizer_method} != {quantization['method']}"
        )
    rotation = quantization.get("rotation", "none")
    if args.core_rotation is not None and args.core_rotation != rotation:
        raise ValueError(
            "core rotation does not match the base checkpoint: "
            f"{args.core_rotation} != {rotation}"
        )
    rotation_block = int(quantization.get("rotation_block", 4))
    student = pretrained(args.device)
    if resuming:
        validate_resume_config(args, payload, group_size)
        surfaces = add_surface_parametrizations(
            student,
            scope="all_matrix_and_conv",
            group_size=group_size,
            method=method,
            core_rotation=rotation,
            rotation_block=rotation_block,
            conv_method=conv_method,
            conv_grouping=conv_grouping,
            schedule_profile=schedule_profile,
        )
        apply_surface_group_sizes(surfaces, surface_group_sizes)
        student.load_state_dict(payload["model"], strict=True)
        start_step = int(payload["surface_step"])
        base_global_step = int(payload["surface_training"]["base_global_step"])
        hard_core = bool(payload["surface_training"]["hard_core_from_start"])
    elif args.initialization == "fp32":
        surfaces = add_surface_parametrizations(
            student,
            scope="all_matrix_and_conv",
            group_size=group_size,
            method=method,
            core_rotation=rotation,
            rotation_block=rotation_block,
            conv_method=conv_method,
            conv_grouping=conv_grouping,
            schedule_profile=schedule_profile,
        )
        apply_surface_group_sizes(surfaces, surface_group_sizes)
        start_step = 0
        base_global_step = 0
        hard_core = False
    else:
        student.load_state_dict(
            recovery_dense_state(payload, args.device),
            strict=True,
        )
        surfaces = add_surface_parametrizations(
            student,
            scope="all_matrix_and_conv",
            group_size=group_size,
            method=method,
            core_rotation=rotation,
            rotation_block=rotation_block,
            conv_method=conv_method,
            conv_grouping=conv_grouping,
            schedule_profile=schedule_profile,
        )
        apply_surface_group_sizes(surfaces, surface_group_sizes)
        start_step = 0
        base_global_step = int(payload["step"])
        hard_core = True
    set_surface_progress(
        surfaces,
        min(1.0, start_step / max(1, args.schedule_steps)),
        hard_core=hard_core,
    )
    return (
        student.train(),
        surfaces,
        group_size,
        start_step,
        base_global_step,
        resuming,
        hard_core,
    )
