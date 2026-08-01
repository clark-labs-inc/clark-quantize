"""Train only packed int7 scales while holding every deployed code fixed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import parametrize

from evaluate_hard_started_checkpoints import load_checkpoint_student
from retained_precision_islands import (
    RSTB_FUSION_FP16,
    symmetric_int7_per_output_components,
)


CHECKPOINT_FORMAT = "backdropkit-swin2sr-fixed-int7-scale-training-v1"


class FixedInt7ScaleProjection(torch.nn.Module):
    """Decode frozen int7 codes with 240 trainable FP16 scale values."""

    def __init__(self, weight: torch.Tensor, scale_method: str) -> None:
        super().__init__()
        codes, scales = symmetric_int7_per_output_components(
            weight.detach(),
            scale_method,
        )
        self.register_buffer("codes", codes)
        self.register_buffer("initial_scales", scales.to(torch.float16))
        self.log_scale_delta = torch.nn.Parameter(
            torch.zeros_like(scales, dtype=torch.float32)
        )

    def deployment_scales(self, dtype: torch.dtype) -> torch.Tensor:
        bounded = self.log_scale_delta.clamp(-0.25, 0.25)
        continuous = self.initial_scales.float() * torch.exp(bounded)
        rounded = continuous.to(torch.float16).to(dtype)
        return continuous.to(dtype) + (rounded - continuous.to(dtype)).detach()

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        scales = self.deployment_scales(weight.dtype)
        decoded = self.codes.to(weight.dtype) * scales[:, None]
        return decoded.reshape_as(weight)


@dataclass(frozen=True)
class FixedScaleSurface:
    name: str
    projection: FixedInt7ScaleProjection


def configure_fixed_int7_scales(model, surfaces, lloyd_mask: int) -> list[FixedScaleSurface]:
    if not 0 <= lloyd_mask < 16:
        raise ValueError("four-layer Lloyd mask must be between 0 and 15")
    by_name = {surface.name: surface for surface in surfaces}
    model.requires_grad_(False)
    configured = []
    for stage, (name, expected_values) in enumerate(RSTB_FUSION_FP16.modules):
        surface = by_name.get(name)
        if surface is None:
            raise ValueError(f"RSTB fusion surface is missing: {name}")
        module = surface.module
        parent = module.weight.detach().clone()
        if parent.numel() != expected_values:
            raise ValueError(f"RSTB fusion surface shape changed: {name}")
        method = "mse_iterative" if lloyd_mask & (1 << stage) else "max"
        projection = FixedInt7ScaleProjection(parent, method)
        parametrize.remove_parametrizations(
            module,
            "weight",
            leave_parametrized=False,
        )
        module.weight.data.copy_(parent)
        parametrize.register_parametrization(module, "weight", projection)
        module.parametrizations.weight.original.requires_grad_(False)
        projection.log_scale_delta.requires_grad_(True)
        expected_codes, expected_scales = symmetric_int7_per_output_components(
            parent,
            method,
        )
        expected = (
            expected_codes.to(parent.dtype) * expected_scales[:, None]
        ).reshape_as(parent)
        if not torch.equal(module.weight.detach(), expected):
            raise RuntimeError(f"fixed-scale initialization changed {name}")
        configured.append(FixedScaleSurface(name, projection))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != 240:
        raise RuntimeError("fixed int7 scale recovery must expose exactly 240 values")
    return configured


def load_fixed_int7_scale_student(
    parent: Path,
    lloyd_mask: int,
    device: str,
    checkpoint: Path | None = None,
):
    student, surfaces, parent_payload = load_checkpoint_student(parent, device)
    fixed = configure_fixed_int7_scales(student, surfaces, lloyd_mask)
    payload = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("fixed-scale checkpoint has the wrong format")
        if int(payload["fixed_int7_scale_training"]["lloyd_mask"]) != lloyd_mask:
            raise ValueError("fixed-scale checkpoint has a different Lloyd mask")
        student.load_state_dict(payload["model"], strict=True)
    return student.eval(), surfaces, fixed, parent_payload, payload


def trainable_scale_parameters(fixed: list[FixedScaleSurface]) -> list[torch.nn.Parameter]:
    return [surface.projection.log_scale_delta for surface in fixed]


@torch.no_grad()
def clamp_scale_deltas(fixed: list[FixedScaleSurface]) -> None:
    for surface in fixed:
        surface.projection.log_scale_delta.clamp_(-0.25, 0.25)


def scale_receipt(fixed: list[FixedScaleSurface]) -> dict:
    deltas = torch.cat([
        surface.projection.log_scale_delta.detach().float().cpu()
        for surface in fixed
    ])
    return {
        "parameters": int(deltas.numel()),
        "mean_log_delta": float(deltas.mean().item()),
        "max_abs_log_delta": float(deltas.abs().max().item()),
        "at_bound": int((deltas.abs() >= 0.25).sum().item()),
    }
