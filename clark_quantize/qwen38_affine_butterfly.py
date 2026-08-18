"""Core algebra for the Qwen3.8-27B affine-base3 butterfly experiment.

The representation has exactly three values per group::

    weight[j] = mu[group] + alpha[group] * trit[j]

and an optional shared orthogonal activation transform represented as a product
of sparse Givens stages.  These helpers deliberately contain no sampled-model
or synthetic promotion gate; they are the exact primitives consumed by the
complete-block experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ternary import ternary_codes_scales


@dataclass(frozen=True)
class AffineTernary:
    codes: torch.Tensor
    mu: torch.Tensor
    alpha: torch.Tensor
    group_size: int

    def reconstruct(self) -> torch.Tensor:
        if self.codes.ndim != 2:
            raise ValueError("affine ternary codes must be a matrix")
        rows, columns = self.codes.shape
        groups = (columns + self.group_size - 1) // self.group_size
        if self.mu.shape != (rows, groups) or self.alpha.shape != (rows, groups):
            raise ValueError("affine ternary parameter dimensions changed")
        pad = groups * self.group_size - columns
        codes = self.codes.float()
        if pad:
            codes = torch.nn.functional.pad(codes, (0, pad))
        rebuilt = self.mu.unsqueeze(-1) + self.alpha.unsqueeze(-1) * codes.reshape(
            rows, groups, self.group_size
        )
        return rebuilt.reshape(rows, -1)[:, :columns].contiguous()


def deterministic_givens_pairings(
    dimension: int, *, stages: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Generate reproducible perfect matchings; the matchings need no storage."""
    if dimension <= 0 or dimension % 2 or stages <= 0 or seed < 0:
        raise ValueError("Givens dimensions, stages, and seed are invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pairings = torch.stack(
        [torch.randperm(dimension, generator=generator).reshape(-1, 2) for _ in range(stages)]
    )
    return pairings.to(device=device)


def apply_givens_product(
    values: torch.Tensor,
    angles: torch.Tensor,
    pairings: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a product of disjoint-pair Givens stages on the last dimension."""
    if values.ndim == 0 or angles.ndim != 2 or pairings.ndim != 3:
        raise ValueError("invalid Givens operand ranks")
    stages, pairs = angles.shape
    dimension = values.shape[-1]
    if pairings.shape != (stages, pairs, 2) or pairs * 2 != dimension:
        raise ValueError("Givens angles and pairings do not cover the activation")
    if pairings.device != values.device or angles.device != values.device:
        raise ValueError("Givens values, angles, and pairings must share a device")
    expected = torch.arange(dimension, device=values.device)
    for stage in pairings:
        if not torch.equal(torch.sort(stage.reshape(-1)).values, expected):
            raise ValueError("each Givens stage must be a perfect matching")

    result = values
    order = range(stages - 1, -1, -1) if inverse else range(stages)
    for index in order:
        first = pairings[index, :, 0]
        second = pairings[index, :, 1]
        theta = -angles[index] if inverse else angles[index]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        left = result.index_select(-1, first)
        right = result.index_select(-1, second)
        stage_result = torch.empty_like(result)
        stage_result.index_copy_(-1, first, cosine * left - sine * right)
        stage_result.index_copy_(-1, second, sine * left + cosine * right)
        result = stage_result
    return result


def _solve_affine(
    groups: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.to(groups.dtype)
    count = valid.sum(-1)
    trits = codes.to(groups.dtype) * valid
    sum_t = trits.sum(-1)
    sum_t2 = trits.square().sum(-1)
    sum_w = (groups * valid).sum(-1)
    sum_tw = (groups * trits).sum(-1)
    determinant = count * sum_t2 - sum_t.square()
    safe = determinant.abs() > 1e-12
    mu = torch.where(
        safe,
        (sum_w * sum_t2 - sum_tw * sum_t) / determinant.clamp_min(1e-12),
        sum_w / count.clamp_min(1),
    )
    alpha = torch.where(
        safe,
        (count * sum_tw - sum_t * sum_w) / determinant.clamp_min(1e-12),
        torch.zeros_like(sum_w),
    )
    negative = alpha < 0
    alpha = alpha.abs()
    codes = torch.where(negative.unsqueeze(-1), -codes, codes)
    return mu, alpha, codes


@torch.no_grad()
def initialize_affine_ternary(
    weight: torch.Tensor, *, group_size: int = 128, iterations: int = 4
) -> AffineTernary:
    """Fit a deterministic Euclidean initializer that cannot lose to direct ternary.

    Activation-metric relocation is performed by the complete-block runner.  This
    initializer supplies its exact three-state starting point and falls back per
    group to the frozen optimal symmetric solution if Lloyd relocation is worse.
    """
    if weight.ndim != 2 or not weight.is_floating_point() or group_size <= 0 or iterations <= 0:
        raise ValueError("affine ternary initialization expects a floating matrix")
    rows, columns = weight.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("affine ternary initialization expects non-empty weights")
    source = weight.float().contiguous()
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    padded = torch.nn.functional.pad(source, (0, pad)) if pad else source
    grouped = padded.reshape(rows, groups, group_size)
    mask = torch.ones_like(grouped, dtype=torch.bool)
    if pad:
        mask[:, -1, -pad:] = False

    direct_codes, direct_scales = ternary_codes_scales(source, group=group_size, method="optimal")
    padded_direct = torch.nn.functional.pad(direct_codes, (0, pad)) if pad else direct_codes
    direct_grouped = padded_direct.reshape(rows, groups, group_size)
    codes = direct_grouped.clone()

    for _ in range(iterations):
        mu, alpha, codes = _solve_affine(grouped, codes, mask)
        levels = torch.stack((mu - alpha, mu, mu + alpha), dim=-1)
        distances = (grouped.unsqueeze(-1) - levels.unsqueeze(-2)).square()
        codes = distances.argmin(-1).to(torch.int8) - 1
        codes = torch.where(mask, codes, torch.zeros_like(codes))
    mu, alpha, codes = _solve_affine(grouped, codes, mask)

    candidate = mu.unsqueeze(-1) + alpha.unsqueeze(-1) * codes.float()
    direct = direct_scales.float().unsqueeze(-1) * direct_grouped.float()
    candidate_error = ((candidate - grouped).square() * mask).sum(-1)
    direct_error = ((direct - grouped).square() * mask).sum(-1)
    use_candidate = candidate_error <= direct_error
    selected_codes = torch.where(use_candidate.unsqueeze(-1), codes, direct_grouped)
    selected_mu = torch.where(use_candidate, mu, torch.zeros_like(mu))
    selected_alpha = torch.where(use_candidate, alpha, direct_scales.float())
    return AffineTernary(
        codes=selected_codes.reshape(rows, -1)[:, :columns].to(torch.int8).contiguous(),
        mu=selected_mu.contiguous(),
        alpha=selected_alpha.contiguous(),
        group_size=group_size,
    )


def affine_ternary_linear(inputs: torch.Tensor, quantized: AffineTernary) -> torch.Tensor:
    """Evaluate the fused group-sum plus ternary-dot CPU algebra."""
    if inputs.shape[-1] != quantized.codes.shape[1]:
        raise ValueError("activation width does not match affine ternary weights")
    rows, columns = quantized.codes.shape
    groups = (columns + quantized.group_size - 1) // quantized.group_size
    pad = groups * quantized.group_size - columns
    values = inputs.float()
    codes = quantized.codes.float()
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
        codes = torch.nn.functional.pad(codes, (0, pad))
    activation_groups = values.reshape(*values.shape[:-1], groups, quantized.group_size)
    code_groups = codes.reshape(rows, groups, quantized.group_size)
    trit_dot = torch.einsum("...gk,ogk->...og", activation_groups, code_groups)
    group_sum = activation_groups.sum(-1).unsqueeze(-2)
    return (
        quantized.alpha.float() * trit_dot + quantized.mu.float() * group_sum
    ).sum(-1)
