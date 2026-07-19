"""Codebook-native weight geometry regularizers."""

from __future__ import annotations

import torch


def codebook_attraction_loss(master: torch.Tensor, quantized: torch.Tensor, scales: torch.Tensor | None = None) -> torch.Tensor:
    """Pull master weights toward their current hard codebook values."""

    diff = master.float() - quantized.detach().float()
    if scales is None:
        denom = master.detach().float().square().mean().clamp_min(1e-8)
    else:
        denom = scales.detach().float().square().mean().clamp_min(1e-8)
    return diff.square().mean() / denom


def threshold_margin_loss(master: torch.Tensor, threshold: torch.Tensor | float, temperature: float = 0.15) -> torch.Tensor:
    """Discourage weights from hovering near a codebook support threshold."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    threshold_t = torch.as_tensor(threshold, device=master.device, dtype=master.dtype).detach()
    distance = (master.float().abs() - threshold_t.float()).abs()
    scale = master.detach().abs().mean().clamp_min(1e-8)
    return torch.exp(-(distance / scale) / temperature).mean()
