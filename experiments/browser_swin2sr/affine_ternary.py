"""Groupwise affine ternary projection for a bounded Swin2SR screen."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clark_air.quant.optimal_ternary import _pad_flat, quantize_ternary_mse


@dataclass(frozen=True)
class AffineTernaryQuantized:
    """Ternary codes with one shift and one positive scale per group."""

    codes: torch.Tensor
    shifts: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    pad: int

    def dequantize(self) -> torch.Tensor:
        groups = self.codes.reshape(-1, self.group_size).float()
        values = (
            self.shifts.reshape(-1, 1).float()
            + self.scales.reshape(-1, 1).float() * groups
        ).reshape(-1)
        if self.pad:
            values = values[:-self.pad]
        return values.reshape(self.original_shape)


def _fit_continuous(
    groups: torch.Tensor,
    codes: torch.Tensor,
    valid: torch.Tensor,
    eps: float,
    *,
    allow_code_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve shift and scale exactly for fixed codes in every group."""

    mask = valid.to(groups.dtype)
    values = groups * mask
    ternary = codes.to(groups.dtype) * mask
    count = mask.sum(dim=1).clamp_min(1)
    sum_w = values.sum(dim=1)
    sum_t = ternary.sum(dim=1)
    sum_tt = ternary.square().sum(dim=1)
    sum_tw = (ternary * values).sum(dim=1)
    denominator = count * sum_tt - sum_t.square()
    regular = denominator.abs() > eps
    safe_denominator = torch.where(
        regular,
        denominator,
        torch.ones_like(denominator),
    )
    scale = torch.where(
        regular,
        (count * sum_tw - sum_t * sum_w) / safe_denominator,
        torch.zeros_like(sum_w),
    )
    shift = (sum_w - scale * sum_t) / count
    negative = scale < 0
    if allow_code_flip:
        codes = torch.where(negative[:, None], -codes, codes)
        scale = scale.abs()
    degenerate = ~regular | (scale <= eps)
    if allow_code_flip:
        codes = torch.where(degenerate[:, None], torch.zeros_like(codes), codes)
    scale = torch.where(degenerate, torch.zeros_like(scale), scale)
    shift = torch.where(degenerate, sum_w / count, shift)
    return codes, shift, scale


def _nearest_codes(
    groups: torch.Tensor,
    shifts: torch.Tensor,
    scales: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    levels = torch.stack((shifts - scales, shifts, shifts + scales), dim=1)
    distances = (groups[:, :, None] - levels[:, None, :]).square()
    codes = distances.argmin(dim=2).to(torch.int8) - 1
    return torch.where(valid, codes, torch.zeros_like(codes))


def _refine(
    groups: torch.Tensor,
    initial_codes: torch.Tensor,
    valid: torch.Tensor,
    max_iterations: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = initial_codes.to(torch.int8)
    for _ in range(max_iterations):
        codes, shifts, scales = _fit_continuous(
            groups,
            codes,
            valid,
            eps,
            allow_code_flip=True,
        )
        updated = _nearest_codes(groups, shifts, scales, valid)
        if torch.equal(updated, codes):
            break
        codes = updated
    return _fit_continuous(
        groups,
        codes,
        valid,
        eps,
        allow_code_flip=True,
    )


def _group_error(
    groups: torch.Tensor,
    codes: torch.Tensor,
    shifts: torch.Tensor,
    scales: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    restored = shifts[:, None] + scales[:, None] * codes.to(groups.dtype)
    return ((groups - restored).square() * valid).sum(dim=1)


@torch.no_grad()
def fit_affine_ternary_codes(
    weight: torch.Tensor,
    codes: torch.Tensor,
    group_size: int,
    pad: int,
    eps: float = 1e-12,
) -> AffineTernaryQuantized:
    """Fit only shift and scale while preserving an exact ternary code map."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat, inferred_pad = _pad_flat(weight.detach().float(), group_size)
    if inferred_pad != pad:
        raise ValueError(f"pad mismatch: {pad} != {inferred_pad}")
    if codes.numel() != flat.numel():
        raise ValueError("code count does not match padded weight")
    groups = flat.reshape(-1, group_size)
    grouped_codes = codes.detach().to(torch.int8).reshape_as(groups)
    if not bool(((grouped_codes >= -1) & (grouped_codes <= 1)).all()):
        raise ValueError("codes must be ternary")
    valid = torch.ones_like(groups, dtype=torch.bool)
    if pad:
        valid[-1, -pad:] = False
        grouped_codes = grouped_codes.clone()
        grouped_codes[-1, -pad:] = 0
    fitted_codes, shifts, scales = _fit_continuous(
        groups,
        grouped_codes,
        valid,
        eps,
        allow_code_flip=False,
    )
    return AffineTernaryQuantized(
        codes=fitted_codes.reshape(-1),
        shifts=shifts,
        scales=scales,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        pad=pad,
    )


@torch.no_grad()
def quantize_affine_ternary_mse(
    weight: torch.Tensor,
    group_size: int = 112,
    max_iterations: int = 32,
    eps: float = 1e-12,
) -> AffineTernaryQuantized:
    """Alternating least-squares affine ternary projection.

    The symmetric MSE-optimal code pattern is always one initialization, so the
    FP32 affine result cannot have higher weight MSE than that symmetric fit.
    A mean-centered initialization helps when a group has a material offset.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    flat, pad = _pad_flat(weight.detach().float(), group_size)
    groups = flat.reshape(-1, group_size)
    valid = torch.ones_like(groups, dtype=torch.bool)
    if pad:
        valid[-1, -pad:] = False

    symmetric = quantize_ternary_mse(weight, group_size=group_size)
    initial_symmetric = symmetric.codes.reshape_as(groups)
    first = _refine(groups, initial_symmetric, valid, max_iterations, eps)

    count = valid.sum(dim=1).clamp_min(1).to(groups.dtype)
    means = (groups * valid).sum(dim=1) / count
    centered = torch.where(valid, groups - means[:, None], torch.zeros_like(groups))
    centered_fit = quantize_ternary_mse(
        centered.reshape(-1),
        group_size=group_size,
    )
    second = _refine(
        groups,
        centered_fit.codes.reshape_as(groups),
        valid,
        max_iterations,
        eps,
    )

    first_error = _group_error(groups, *first, valid)
    second_error = _group_error(groups, *second, valid)
    choose_second = second_error < first_error
    codes = torch.where(choose_second[:, None], second[0], first[0])
    shifts = torch.where(choose_second, second[1], first[1])
    scales = torch.where(choose_second, second[2], first[2])
    return AffineTernaryQuantized(
        codes=codes.reshape(-1),
        shifts=shifts,
        scales=scales,
        original_shape=tuple(weight.shape),
        group_size=group_size,
        pad=pad,
    )
