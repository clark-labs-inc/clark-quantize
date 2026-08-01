"""Groupwise binary quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryQuantized:
    """Groupwise binary codes and positive scales."""

    codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    pad: int

    def dequantize(self) -> torch.Tensor:
        values = self.codes.reshape(-1, self.group_size).float() * self.scales.reshape(-1, 1).float()
        values = values.reshape(-1)
        if self.pad:
            values = values[:-self.pad]
        return values.reshape(self.original_shape)


@torch.no_grad()
def quantize_binary(weight: torch.Tensor, group_size: int = 112, eps: float = 1e-12) -> BinaryQuantized:
    """Return groupwise sign codes with MSE-optimal scale for fixed signs."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    flat = weight.detach().float().reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    groups = flat.reshape(-1, group_size)
    codes = torch.where(groups >= 0, torch.ones_like(groups), -torch.ones_like(groups)).to(torch.int8)
    scales = groups.abs().mean(dim=1).clamp_min(eps)
    return BinaryQuantized(
        codes=codes.reshape(-1),
        scales=scales,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        pad=pad,
    )
