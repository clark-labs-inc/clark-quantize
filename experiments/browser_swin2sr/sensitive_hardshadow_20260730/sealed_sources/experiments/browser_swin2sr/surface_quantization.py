"""Staged full-surface ternary parametrizations for Swin2SR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.utils import parametrize

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from quantizer_controls import ProgressiveLowBit
from rotation_control import block_hadamard
from surface_schedule import surface_targets


CONV_GROUPINGS = ("flat", "output_channel", "kernel")


@dataclass(frozen=True)
class QuantizedSurface:
    name: str
    category: str
    quantizer: ProgressiveLowBit
    module: torch.nn.Module


def surface_group_size(
    module: torch.nn.Module,
    category: str,
    default: int,
    conv_grouping: str,
) -> int:
    if default <= 0:
        raise ValueError("default group size must be positive")
    if conv_grouping not in CONV_GROUPINGS:
        raise ValueError(f"unsupported convolution grouping: {conv_grouping}")
    if not isinstance(module, torch.nn.Conv2d):
        return default
    if category not in {"residual_body_conv", "image_io_conv"}:
        raise ValueError(f"unexpected convolution category: {category}")
    if conv_grouping == "flat":
        return default
    if conv_grouping == "output_channel":
        return module.weight[0].numel()
    kernel_height, kernel_width = module.kernel_size
    kernel_values = kernel_height * kernel_width
    # A one-value "kernel" would carry one scale per weight and therefore
    # would not be a meaningful low-bit representation. Fall back to one
    # group per output filter for 1x1 projections.
    return kernel_values if kernel_values > 1 else module.weight[0].numel()


def add_surface_parametrizations(
    model: torch.nn.Module,
    *,
    scope: str,
    group_size: int,
    method: str = "optimal_mse",
    core_rotation: str = "none",
    rotation_block: int = 4,
    conv_grouping: str = "flat",
    conv_method: Optional[str] = None,
    schedule_profile: str = "joint_io_v1",
) -> list[QuantizedSurface]:
    targets = {
        target.name: target
        for target in surface_targets(model, scope, schedule_profile)
    }
    selected = []
    for name, module in model.named_modules():
        target = targets.get(name)
        if target is None:
            continue
        rotation = (
            core_rotation
            if target.category == "core_linear"
            else "none"
        )
        if rotation == "hadamard":
            if not isinstance(module, torch.nn.Linear):
                raise ValueError("Hadamard shaping is only valid for linear surfaces")
            module.weight.data.copy_(
                block_hadamard(module.weight.data, rotation_block)
            )
        quantizer = ProgressiveLowBit(
            "ternary",
            (
                conv_method
                if isinstance(module, torch.nn.Conv2d) and conv_method
                else method
            ),
            surface_group_size(
                module,
                target.category,
                group_size,
                conv_grouping,
            ),
            target.start_fraction,
            target.end_fraction,
            module.weight,
            rotation,
            rotation_block,
        )
        parametrize.register_parametrization(module, "weight", quantizer)
        if rotation == "hadamard":
            def prepare_inputs(target_module, inputs):
                active = target_module.parametrizations.weight[0]
                values = block_hadamard(inputs[0], active.rotation_block)
                return (values, *inputs[1:])

            module.register_forward_pre_hook(prepare_inputs)
        selected.append(QuantizedSurface(
            name,
            target.category,
            quantizer,
            module,
        ))
    if set(targets) != {surface.name for surface in selected}:
        missing = sorted(set(targets) - {surface.name for surface in selected})
        raise ValueError(f"surface modules are missing: {missing}")
    return selected


def set_surface_progress(
    surfaces: list[QuantizedSurface],
    progress: float,
    *,
    hard_core: bool = True,
) -> None:
    for surface in surfaces:
        if hard_core and surface.category == "core_linear":
            surface.quantizer.amount.fill_(1.0)
        else:
            surface.quantizer.set_progress(progress)


def surface_parameter_groups(
    model: torch.nn.Module,
    surfaces: list[QuantizedSurface],
) -> dict[str, list[torch.nn.Parameter]]:
    model.requires_grad_(False)
    matrix_masters = []
    convolution_masters = []
    corrections = []
    for surface in surfaces:
        master = surface.module.parametrizations.weight.original
        master.requires_grad_(True)
        if master.ndim == 2:
            matrix_masters.append(master)
        else:
            convolution_masters.append(master)
        if surface.module.bias is not None:
            surface.module.bias.requires_grad_(True)
            corrections.append(surface.module.bias)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
                corrections.append(parameter)
    return {
        "matrix_masters": matrix_masters,
        "convolution_masters": convolution_masters,
        "corrections": corrections,
    }


def build_surface_optimizer(
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    learning_rate: float,
    weight_decay: float = 1e-4,
    use_muon: bool = True,
) -> BonsaiQAT8:
    if learning_rate <= 0:
        raise ValueError("surface learning rate must be positive")
    return BonsaiQAT8([
        {
            "params": groups["matrix_masters"],
            "role": "matrix_qat8",
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
            "use_muon": use_muon,
            "muon_every": 16,
            "muon_steps": 2,
        },
        {
            "params": groups["convolution_masters"],
            "role": "adamw",
            "lr": learning_rate * 0.25,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
        },
        {
            "params": groups["corrections"],
            "role": "adamw",
            "lr": learning_rate * 0.10,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
        },
    ])
