"""Build mixed-rate candidate tables from Fisher damages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clark_air.quant import quantize_binary, quantize_ternary_mse
from clark_air.rate.fisher_diag import fisher_quadratic_damage
from clark_air.rate.waterfill import RateCandidate


@dataclass(frozen=True)
class GroupPolicyInput:
    """One tensor/group to consider for binary versus ternary assignment."""

    name: str
    weight: torch.Tensor
    fisher_diag: torch.Tensor
    group_size: int = 112
    scale_bits: int = 8
    ternary_code_bits: float = 64.0 / 40.0


def candidates_for_group(group: GroupPolicyInput) -> list[RateCandidate]:
    """Create binary and ternary rate candidates for one tensor group."""

    weight = group.weight.detach().float().cpu()
    fisher = group.fisher_diag.detach().float().cpu()
    ternary = quantize_ternary_mse(weight, group_size=group.group_size).dequantize().cpu()
    binary = quantize_binary(weight, group_size=group.group_size).dequantize().cpu()
    n = float(weight.numel())
    ternary_bits = n * (group.ternary_code_bits + group.scale_bits / group.group_size)
    binary_bits = n * (1.0 + group.scale_bits / group.group_size)
    return [
        RateCandidate(
            group=group.name,
            rate="ternary",
            bits=ternary_bits,
            damage=fisher_quadratic_damage(ternary - weight, fisher),
        ),
        RateCandidate(
            group=group.name,
            rate="binary",
            bits=binary_bits,
            damage=fisher_quadratic_damage(binary - weight, fisher),
        ),
    ]


def candidates_for_groups(groups: list[GroupPolicyInput]) -> list[RateCandidate]:
    candidates: list[RateCandidate] = []
    for group in groups:
        candidates.extend(candidates_for_group(group))
    return candidates
