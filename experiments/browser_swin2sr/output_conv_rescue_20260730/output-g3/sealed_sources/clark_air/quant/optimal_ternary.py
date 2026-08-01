"""MSE-optimal groupwise ternary quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TernaryQuantized:
    """Groupwise ternary codes and positive scales."""

    codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    pad: int

    def dequantize(self) -> torch.Tensor:
        flat_codes = self.codes.reshape(-1, self.group_size).float()
        flat_scales = self.scales.reshape(-1, 1).float()
        values = (flat_codes * flat_scales).reshape(-1)
        if self.pad:
            values = values[:-self.pad]
        return values.reshape(self.original_shape)


def _pad_flat(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat, pad


@torch.no_grad()
def quantize_ternary_mse(weight: torch.Tensor, group_size: int = 112, eps: float = 1e-12) -> TernaryQuantized:
    """Return MSE-optimal ternary codes for each group.

    For a selected support S in one group, the optimal scale is:

        s = mean(abs(w_i), i in S)

    We scan all top-k absolute-value supports and choose the k with minimum
    squared error. This is the Clark/Sana PTQ initializer that survived where
    naive absmean thresholding collapsed.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    flat, pad = _pad_flat(weight.detach().float(), group_size)
    groups = flat.reshape(-1, group_size)
    abs_groups = groups.abs()
    sorted_abs, sorted_idx = torch.sort(abs_groups, dim=1, descending=True)
    cumsum_abs = sorted_abs.cumsum(dim=1)
    cumsum_sq = sorted_abs.square().cumsum(dim=1)
    total_sq = abs_groups.square().sum(dim=1, keepdim=True)
    k = torch.arange(1, group_size + 1, device=groups.device, dtype=groups.dtype).reshape(1, -1)
    support_error = total_sq - cumsum_abs.square() / k
    best_k_idx = support_error.argmin(dim=1)
    best_k = best_k_idx + 1
    best_sum_abs = cumsum_abs.gather(1, best_k_idx.reshape(-1, 1)).squeeze(1)
    scales = (best_sum_abs / best_k.to(groups.dtype)).clamp_min(eps)

    ranks = torch.empty_like(sorted_idx)
    ranks.scatter_(1, sorted_idx, torch.arange(group_size, device=groups.device).expand_as(sorted_idx))
    support = ranks < best_k.reshape(-1, 1)
    codes = torch.where(support, groups.sign(), torch.zeros_like(groups)).to(torch.int8)
    return TernaryQuantized(
        codes=codes.reshape(-1),
        scales=scales,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        pad=pad,
    )
