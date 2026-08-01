"""Continuous compensation-island policies for Swin2SR ternary recovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from surface_schedule import module_category


POLICIES = ("frozen", "norms", "norms_and_residual")


@dataclass(frozen=True)
class IslandParameter:
    name: str
    role: str
    values: int
    learning_rate_multiplier: float


def island_parameters(model: torch.nn.Module, policy: str) -> list[IslandParameter]:
    if policy not in POLICIES:
        raise ValueError(f"unknown island policy: {policy}")
    if policy == "frozen":
        return []
    selected = []
    seen = set()
    for module_name, module in model.named_modules():
        category = module_category(module_name, module)
        if isinstance(module, torch.nn.LayerNorm):
            role = "normalization"
            multiplier = 0.10
        elif (
            policy == "norms_and_residual"
            and category == "residual_body_conv"
        ):
            role = "residual_body"
            multiplier = 0.05
        else:
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            full_name = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
            selected.append(IslandParameter(
                name=full_name,
                role=role,
                values=parameter.numel(),
                learning_rate_multiplier=multiplier,
            ))
    return selected
