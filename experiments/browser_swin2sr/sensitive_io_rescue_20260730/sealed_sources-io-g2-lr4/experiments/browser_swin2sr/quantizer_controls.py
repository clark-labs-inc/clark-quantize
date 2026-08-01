"""Low-bit quantizer controls for the Swin2SR research ladder."""

from __future__ import annotations

import math

import torch

from clark_air.quant.binary import quantize_binary
from clark_air.quant.optimal_ternary import (
    TernaryQuantized,
    _pad_flat,
    quantize_ternary_mse,
)


@torch.no_grad()
def quantize_ternary_absmean(
    weight: torch.Tensor,
    group_size: int,
    eps: float = 1e-12,
) -> TernaryQuantized:
    """Groupwise BitNet-style absmean ternary control."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat, pad = _pad_flat(weight.detach().float(), group_size)
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().mean(dim=1).clamp_min(eps)
    codes = torch.round(groups / scales[:, None]).clamp(-1, 1).to(torch.int8)
    return TernaryQuantized(
        codes=codes.reshape(-1),
        scales=scales,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        pad=pad,
    )


class ParetoSeq(torch.autograd.Function):
    """Exact ternary SEQ forward/backward from Meta's ParetoQ reference."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        safe_alpha = alpha.clamp_min(1e-5)
        normalized = weight / safe_alpha
        clipped = normalized.clamp(-0.99, 0.99)
        levels = torch.round(clipped * 1.5) / 1.5
        ctx.save_for_backward(normalized, safe_alpha)
        return levels * safe_alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        normalized, _safe_alpha = ctx.saved_tensors
        middle = (normalized >= -0.99) & (normalized <= 0.99)
        small = normalized < -0.99
        big = normalized > 0.99
        levels = torch.round(normalized.clamp(-0.99, 0.99) * 1.5) / 1.5
        alpha_term = (
            -small.to(normalized.dtype)
            + big.to(normalized.dtype)
            + middle.to(normalized.dtype) * (levels - normalized)
        )
        gradient_scale = 1.0 / math.sqrt(normalized.numel())
        grad_alpha = (
            alpha_term * grad_output * gradient_scale
        ).sum(dim=-1, keepdim=True)
        grad_weight = middle.to(grad_output.dtype) * grad_output
        return grad_weight, grad_alpha


class ProgressiveLowBit(torch.nn.Module):
    """Scheduled hard projection for preregistered quantizer controls."""

    def __init__(
        self,
        mode: str,
        method: str,
        group_size: int,
        start_fraction: float,
        end_fraction: float,
        initial_weight: torch.Tensor,
        rotation: str = "none",
        rotation_block: int = 4,
    ) -> None:
        super().__init__()
        if mode != "ternary" and method != "optimal_mse":
            raise ValueError(f"{method} is only defined for ternary mode")
        self.mode = mode
        self.method = method
        self.group_size = group_size
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.rotation = rotation
        self.rotation_block = rotation_block
        self.register_buffer("amount", torch.tensor(0.0), persistent=False)
        if method == "paretoq_seq":
            row_max = initial_weight.detach().abs().amax(dim=-1, keepdim=True)
            self.alpha = torch.nn.Parameter(row_max.clamp_min(1e-5))
        else:
            self.register_parameter("alpha", None)

    def set_progress(self, progress: float) -> None:
        span = max(1e-6, self.end_fraction - self.start_fraction)
        amount = min(1.0, max(0.0, (progress - self.start_fraction) / span))
        self.amount.fill_(amount)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":
            return quantize_binary(weight, group_size=self.group_size).dequantize()
        if self.method == "optimal_mse":
            return quantize_ternary_mse(
                weight,
                group_size=self.group_size,
            ).dequantize()
        if self.method == "absmean":
            return quantize_ternary_absmean(
                weight,
                group_size=self.group_size,
            ).dequantize()
        if self.method == "paretoq_seq":
            return ParetoSeq.apply(weight, self.alpha)
        raise ValueError(f"unsupported quantizer method: {self.method}")

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        projected = self.project(weight)
        if self.method == "paretoq_seq":
            return weight + self.amount * (projected - weight)
        return weight + self.amount * (projected - weight).detach()
