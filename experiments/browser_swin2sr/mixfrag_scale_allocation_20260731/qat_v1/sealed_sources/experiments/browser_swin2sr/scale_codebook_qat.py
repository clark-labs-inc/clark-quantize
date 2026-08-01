"""A second gradual QAT stage that learns browser-storable ternary scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import parametrize

from scale_codebook import group_nonzero_counts, quantize_scale_codebook


SCALE_CATEGORY_SCHEDULE = {
    "core_linear": (0.00, 0.40),
    "residual_body_conv": (0.20, 0.60),
    "position_bias_linear": (0.40, 0.75),
    "image_io_conv": (0.65, 1.00),
}


class CodebookScaleSTE(torch.autograd.Function):
    """Hard scale-codebook forward with identity weight and learned-center gradients."""

    @staticmethod
    def forward(
        ctx,
        projected: torch.Tensor,
        centers: torch.Tensor,
        amount: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        if group_size <= 0:
            raise ValueError("scale-codebook group size must be positive")
        flat = projected.reshape(-1)
        pad = (-flat.numel()) % group_size
        padded = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
        groups = padded.reshape(-1, group_size)
        scales = groups.detach().abs().amax(dim=1)
        safe = scales.clamp_min(torch.finfo(projected.dtype).eps)
        codes = torch.round(groups.detach() / safe[:, None]).clamp(-1, 1)
        codes = torch.where(scales[:, None] == 0, torch.zeros_like(codes), codes)
        assignments = torch.abs(scales[:, None] - centers.detach()[None, :]).argmin(dim=1)
        quantized = codes * centers[assignments, None]
        blended = groups + amount * (quantized - groups)
        ctx.save_for_backward(codes, assignments, amount)
        ctx.group_size = group_size
        ctx.center_count = centers.numel()
        ctx.original_values = flat.numel()
        return blended.reshape(-1)[: flat.numel()].reshape_as(projected)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        codes, assignments, amount = ctx.saved_tensors
        flat = grad_output.reshape(-1)
        pad = (-flat.numel()) % ctx.group_size
        padded = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
        group_gradients = (padded.reshape(-1, ctx.group_size) * codes).sum(dim=1)
        grad_centers = grad_output.new_zeros(ctx.center_count)
        grad_centers.scatter_add_(0, assignments, amount * group_gradients)
        return grad_output, grad_centers, None, None


class ScaleCodebookProjection(torch.nn.Module):
    def __init__(
        self,
        initial_weight: torch.Tensor,
        group_size: int,
        target_bits: int,
    ) -> None:
        super().__init__()
        if target_bits < 1 or target_bits > 8:
            raise ValueError("target scale bits must be between one and eight")
        flat = initial_weight.detach().float().cpu().numpy().reshape(-1)
        pad = (-flat.size) % group_size
        padded = np.pad(flat, (0, pad)) if pad else flat
        groups = padded.reshape(-1, group_size)
        scales = np.max(np.abs(groups), axis=1).astype(np.float32)
        safe = np.maximum(scales, np.finfo(np.float32).eps)
        codes = np.rint(groups / safe[:, None]).clip(-1, 1).astype(np.int8)
        codes[scales == 0] = 0
        fitted = quantize_scale_codebook(
            scales,
            group_nonzero_counts(codes.reshape(-1), group_size),
            target_bits,
        )
        self.group_size = group_size
        self.target_bits = target_bits
        centers = torch.from_numpy(fitted.centers.astype(np.float32)).to(
            initial_weight.device
        ).clamp_min(1e-8)
        self.raw_centers = torch.nn.Parameter(torch.log(torch.expm1(centers)))
        self.register_buffer("amount", torch.tensor(0.0), persistent=True)

    @property
    def centers(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_centers)

    @property
    def stored_centers(self) -> torch.Tensor:
        """Use the exact FP16 values the browser artifact will store, with STE."""

        centers = self.centers
        rounded = centers.to(torch.float16).to(centers.dtype)
        return centers + (rounded - centers).detach()

    @torch.no_grad()
    def set_centers(self, values: torch.Tensor) -> None:
        positive = values.to(self.raw_centers).clamp_min(1e-8)
        self.raw_centers.copy_(torch.log(torch.expm1(positive)))

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        return CodebookScaleSTE.apply(
            projected,
            self.stored_centers,
            self.amount,
            self.group_size,
        )


@dataclass(frozen=True)
class ScaleCodebookSurface:
    name: str
    category: str
    target_bits: int
    projection: ScaleCodebookProjection
    module: torch.nn.Module


def add_scale_codebooks(
    surfaces,
    *,
    target_bits_by_category: dict[str, int],
) -> list[ScaleCodebookSurface]:
    unknown = set(target_bits_by_category) - set(SCALE_CATEGORY_SCHEDULE)
    if unknown:
        raise ValueError(f"unknown scale-codebook categories: {sorted(unknown)}")
    target_bits_by_surface = {
        surface.name: int(target_bits_by_category[surface.category])
        for surface in surfaces
    }
    return add_per_surface_scale_codebooks(
        surfaces,
        target_bits_by_surface=target_bits_by_surface,
    )


def add_per_surface_scale_codebooks(
    surfaces,
    *,
    target_bits_by_surface: dict[str, int],
) -> list[ScaleCodebookSurface]:
    names = {surface.name for surface in surfaces}
    if set(target_bits_by_surface) != names:
        missing = sorted(names - set(target_bits_by_surface))
        extra = sorted(set(target_bits_by_surface) - names)
        raise ValueError(
            f"per-surface scale map differs: missing={missing[:3]} extra={extra[:3]}"
        )
    selected = []
    for surface in surfaces:
        amount = float(surface.quantizer.amount.item())
        if amount != 1.0:
            raise ValueError(f"ternary surface is not hard before scale QAT: {surface.name}")
        target_bits = int(target_bits_by_surface[surface.name])
        if target_bits < 1 or target_bits > 8:
            raise ValueError(f"target scale bits are invalid: {surface.name}")
        projection = ScaleCodebookProjection(
            surface.module.weight.detach(),
            surface.quantizer.group_size,
            target_bits,
        )
        parametrize.register_parametrization(surface.module, "weight", projection)
        selected.append(ScaleCodebookSurface(
            name=surface.name,
            category=surface.category,
            target_bits=target_bits,
            projection=projection,
            module=surface.module,
        ))
    return selected


def set_scale_codebook_progress(
    surfaces: list[ScaleCodebookSurface],
    progress: float,
) -> None:
    if not 0 <= progress <= 1:
        raise ValueError("scale-codebook progress must be between zero and one")
    for surface in surfaces:
        start, end = SCALE_CATEGORY_SCHEDULE[surface.category]
        amount = min(1.0, max(0.0, (progress - start) / max(1e-6, end - start)))
        surface.projection.amount.fill_(amount)


def scale_codebook_parameter_group(
    surfaces: list[ScaleCodebookSurface],
) -> list[torch.nn.Parameter]:
    parameters = [surface.projection.raw_centers for surface in surfaces]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def scale_codebook_summary(surfaces: list[ScaleCodebookSurface]) -> dict:
    values = [float(surface.projection.amount.item()) for surface in surfaces]
    return {
        "surfaces": len(surfaces),
        "transitioning_surfaces": sum(0 < value < 1 for value in values),
        "fully_codebook_surfaces": sum(value == 1 for value in values),
        "codebook_parameters": sum(
            surface.projection.raw_centers.numel() for surface in surfaces
        ),
        "target_bits_by_category": {
            category: sorted({
                surface.target_bits for surface in surfaces
                if surface.category == category
            })
            for category in sorted({surface.category for surface in surfaces})
        },
        "target_bit_counts": {
            str(bits): sum(surface.target_bits == bits for surface in surfaces)
            for bits in sorted({surface.target_bits for surface in surfaces})
        },
    }
