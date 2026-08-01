"""Model-specific staged ternary surfaces for BackdropKit Swin2SR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCOPES = ("core", "core_residual", "all_matrix_and_conv")
SCHEDULE_PROFILES = ("joint_io_v1", "sensitive_io_v2")


@dataclass(frozen=True)
class SurfaceTarget:
    name: str
    category: str
    values: int
    start_fraction: float
    end_fraction: float


def module_category(name: str, module: torch.nn.Module) -> str | None:
    if isinstance(module, torch.nn.Linear):
        if "continuous_position_bias_mlp" in name:
            return "position_bias_linear"
        if (
            ".attention." in name
            or ".intermediate." in name
            or ".output.dense" in name
        ):
            return "core_linear"
        return None
    if isinstance(module, torch.nn.Conv2d):
        if name in {"swin2sr.first_convolution", "upsample.conv"}:
            return "image_io_conv"
        return "residual_body_conv"
    return None


def category_schedule(
    name: str,
    category: str,
    profile: str = "joint_io_v1",
) -> tuple[float, float]:
    if profile not in SCHEDULE_PROFILES:
        raise ValueError(f"unknown surface schedule profile: {profile}")
    if profile == "sensitive_io_v2":
        if category == "core_linear":
            if ".attention." in name:
                return 0.03, 0.15
            return 0.10, 0.30
        if category == "residual_body_conv":
            return 0.28, 0.50
        if category == "position_bias_linear":
            return 0.48, 0.65
        if name == "swin2sr.first_convolution":
            return 0.62, 0.79
        if name == "upsample.conv":
            return 0.80, 0.98
        raise ValueError(f"unknown sensitive surface: {name} ({category})")
    if category == "core_linear":
        if ".attention." in name:
            return 0.00, 0.12
        return 0.08, 0.28
    if category == "residual_body_conv":
        return 0.25, 0.55
    if category == "position_bias_linear":
        return 0.50, 0.72
    if category == "image_io_conv":
        return 0.70, 0.92
    raise ValueError(f"unknown surface category: {category}")


def selected_categories(scope: str) -> set[str]:
    if scope == "core":
        return {"core_linear"}
    if scope == "core_residual":
        return {"core_linear", "residual_body_conv"}
    if scope == "all_matrix_and_conv":
        return {
            "core_linear",
            "residual_body_conv",
            "position_bias_linear",
            "image_io_conv",
        }
    raise ValueError(f"unknown quantization scope: {scope}")


def surface_targets(
    model: torch.nn.Module,
    scope: str,
    schedule_profile: str = "joint_io_v1",
) -> list[SurfaceTarget]:
    selected = selected_categories(scope)
    targets = []
    for name, module in model.named_modules():
        category = module_category(name, module)
        if category not in selected:
            continue
        start, end = category_schedule(name, category, schedule_profile)
        targets.append(SurfaceTarget(
            name=name,
            category=category,
            values=module.weight.numel(),
            start_fraction=start,
            end_fraction=end,
        ))
    if not targets:
        raise ValueError(f"scope selected no model surfaces: {scope}")
    return targets
