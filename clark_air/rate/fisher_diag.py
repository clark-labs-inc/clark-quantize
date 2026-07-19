"""Empirical diagonal Fisher utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

import torch


@dataclass
class DiagonalFisher:
    """Running diagonal empirical Fisher estimate keyed by parameter name."""

    values: dict[str, torch.Tensor] = field(default_factory=dict)
    count: int = 0

    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        for name, param in named_parameters:
            if param.grad is None:
                continue
            grad2 = param.grad.detach().float().square().cpu()
            if name not in self.values:
                self.values[name] = grad2
            else:
                self.values[name].add_(grad2)
        self.count += 1

    def mean(self) -> dict[str, torch.Tensor]:
        denom = max(self.count, 1)
        return {name: value / denom for name, value in self.values.items()}


def fisher_quadratic_damage(delta: torch.Tensor, fisher_diag: torch.Tensor) -> float:
    """Compute 0.5 * delta^T diag(F) delta."""

    return float(0.5 * (delta.detach().float().cpu().square() * fisher_diag.float().cpu()).sum().item())
