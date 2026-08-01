"""Hard-shadow curriculum controls for staged Swin2SR ternary training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qat_objective import distillation_losses
from recovery_controls import localized_feature_loss

if TYPE_CHECKING:
    from surface_quantization import QuantizedSurface
else:
    QuantizedSurface = Any


@dataclass(frozen=True)
class HardShadowPlan:
    weight: float
    transition_fraction: float
    active_surfaces: int
    hard_surfaces: int


def hard_shadow_plan(
    surfaces: list[QuantizedSurface],
    *,
    weight_start: float,
    weight_end: float,
) -> HardShadowPlan:
    """Describe a hard branch for surfaces currently entering ternary."""

    if weight_start < 0 or weight_end < weight_start:
        raise ValueError("hard-shadow weights must satisfy 0 <= start <= end")
    active = []
    hard = 0
    for surface in surfaces:
        amount = float(surface.quantizer.amount.item())
        if amount > 0:
            hard += 1
        if 0 < amount < 1:
            active.append(amount)
    if not active or weight_end == 0:
        return HardShadowPlan(0.0, 0.0, 0, hard)
    transition_fraction = max(active)
    weight = weight_start + transition_fraction * (weight_end - weight_start)
    return HardShadowPlan(weight, transition_fraction, len(active), hard)


def enter_hard_shadow(
    surfaces: list[QuantizedSurface],
) -> list[float]:
    """Make every started surface hard while leaving future stages dense."""

    live_amounts = []
    for surface in surfaces:
        amount = float(surface.quantizer.amount.item())
        live_amounts.append(amount)
        surface.quantizer.amount.fill_(1.0 if amount > 0 else 0.0)
    return live_amounts


def restore_live_amounts(
    surfaces: list[QuantizedSurface],
    amounts: list[float],
) -> None:
    if len(surfaces) != len(amounts):
        raise ValueError("surface and amount counts differ")
    for surface, amount in zip(surfaces, amounts):
        surface.quantizer.amount.fill_(amount)


def hard_shadow_backward(
    student,
    low,
    high,
    teacher_output,
    teacher_reconstruction,
    surfaces,
    weights,
    *,
    loss_recipe: str,
    pixel_objective: str,
    ground_truth_mse_floor: float,
    teacher_mse_floor: float,
    surface_step: int,
    every: int,
    weight_start: float,
    weight_end: float,
) -> dict[str, float]:
    plan = hard_shadow_plan(
        surfaces,
        weight_start=weight_start,
        weight_end=weight_end,
    )
    if plan.weight == 0 or every <= 0 or surface_step % every:
        return {}
    live_amounts = enter_hard_shadow(surfaces)
    try:
        output = student(low, output_hidden_states=True)
        reconstruction = output.reconstruction.clamp(0, 1)
        losses = distillation_losses(
            reconstruction,
            teacher_reconstruction,
            high,
            (output.hidden_states, teacher_output.hidden_states),
            pixel_objective=pixel_objective,
            ground_truth_mse_floor=ground_truth_mse_floor,
            teacher_mse_floor=teacher_mse_floor,
        )
        losses["feature"] = localized_feature_loss(
            output.hidden_states,
            teacher_output.hidden_states,
            None,
            "l2" if loss_recipe == "teacher_heavy" else "layernorm",
        )
        loss = sum(weights[name] * losses[name] for name in weights)
        (plan.weight * loss).backward()
        return {
            "hard_shadow_loss": loss.item(),
            "hard_shadow_weight": plan.weight,
            "hard_shadow_transition_fraction": plan.transition_fraction,
            "hard_shadow_active_surfaces": plan.active_surfaces,
            **{
                f"hard_shadow_{name}": value.item()
                for name, value in losses.items()
            },
        }
    finally:
        restore_live_amounts(surfaces, live_amounts)
