"""Preserve a recovered scale bridge while gradually collapsing it to q1."""

from __future__ import annotations

import torch

from scale_codebook_qat import (
    SCALE_CATEGORY_SCHEDULE,
    ScaleCodebookSurface,
    add_scale_codebooks,
)


STAGED_SCHEDULE_PROFILES = {
    "late_io_v1": SCALE_CATEGORY_SCHEDULE,
    "gentle_io_v1": {
        "core_linear": (0.00, 0.40),
        "residual_body_conv": (0.15, 0.65),
        "position_bias_linear": (0.35, 0.80),
        "image_io_conv": (0.50, 1.00),
    },
}


def require_hard_bridge(surfaces: list[ScaleCodebookSurface]) -> None:
    if not surfaces:
        raise ValueError("staged collapse requires at least one bridge surface")
    for surface in surfaces:
        amount = float(surface.projection.amount.item())
        if amount != 1.0:
            raise ValueError(
                f"scale bridge is not hard before staged collapse: {surface.name}"
            )


def freeze_hard_bridge(
    surfaces: list[ScaleCodebookSurface],
) -> list[torch.nn.Parameter]:
    """Freeze the bridge centers while leaving continuous masters trainable."""

    require_hard_bridge(surfaces)
    parameters = [surface.projection.raw_centers for surface in surfaces]
    for parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    return parameters


def add_staged_collapse(
    ternary_surfaces,
    bridge_surfaces: list[ScaleCodebookSurface],
    *,
    target_bits_by_category: dict[str, int],
) -> list[ScaleCodebookSurface]:
    """Append a gradual projection after an exact, frozen hard bridge."""

    if [surface.name for surface in ternary_surfaces] != [
        surface.name for surface in bridge_surfaces
    ]:
        raise ValueError("ternary and bridge surface order differs")
    freeze_hard_bridge(bridge_surfaces)
    collapse = add_scale_codebooks(
        ternary_surfaces,
        target_bits_by_category=target_bits_by_category,
    )
    if any(float(surface.projection.amount.item()) != 0.0 for surface in collapse):
        raise RuntimeError("new staged-collapse projections must start continuous")
    return collapse


def bridge_summary(surfaces: list[ScaleCodebookSurface]) -> dict:
    return {
        "surfaces": len(surfaces),
        "hard_surfaces": sum(
            float(surface.projection.amount.item()) == 1.0 for surface in surfaces
        ),
        "frozen_center_parameters": sum(
            surface.projection.raw_centers.numel() for surface in surfaces
            if not surface.projection.raw_centers.requires_grad
        ),
        "bits_by_category": {
            category: sorted({
                surface.target_bits for surface in surfaces
                if surface.category == category
            })
            for category in sorted({surface.category for surface in surfaces})
        },
    }


def set_staged_collapse_progress(
    surfaces: list[ScaleCodebookSurface],
    progress: float,
    profile: str,
) -> None:
    if not 0 <= progress <= 1:
        raise ValueError("staged-collapse progress must be between zero and one")
    if profile not in STAGED_SCHEDULE_PROFILES:
        raise ValueError(f"unknown staged-collapse schedule: {profile}")
    schedule = STAGED_SCHEDULE_PROFILES[profile]
    for surface in surfaces:
        start, end = schedule[surface.category]
        amount = min(1.0, max(0.0, (progress - start) / (end - start)))
        surface.projection.amount.fill_(amount)
