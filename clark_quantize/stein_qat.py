"""Stein helpers for distributional low-bit QAT probes.

These utilities are deliberately model-agnostic. The Sana trainer supplies
the scheduler convention and teacher score; this module only builds the
kernel Stein transport field and simple rate-distortion choices.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


def tensor_rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    x = x.float()
    return torch.sqrt(torch.mean(x * x) + eps)


def flow_velocity_to_score(
    sample: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Convert Sana flow velocity to a score for use-flow-sigmas schedulers.

    Diffusers' Sana scheduler uses ``prediction_type="flow_prediction"`` and
    ``x_sigma = (1 - sigma) * x0 + sigma * eps``. Its model output is
    ``v = eps - x0`` and ``x0_pred = x - sigma * v``. The corresponding score
    is ``-(x + (1 - sigma) * v) / sigma``.
    """
    sig = torch.as_tensor(sigma, device=sample.device, dtype=sample.float().dtype)
    while sig.ndim < sample.ndim:
        sig = sig.view(*sig.shape, *([1] * (sample.ndim - sig.ndim)))
    sig = sig.clamp_min(eps)
    alpha = 1.0 - sig
    return -((sample.float() + alpha * velocity.float()) / sig)


def make_rademacher_projection(
    input_dim: int,
    output_dim: int,
    *,
    device: torch.device | str,
    seed: int = 0,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    projection = torch.empty(
        int(input_dim),
        int(output_dim),
        device=device,
        dtype=torch.float32,
    )
    projection.bernoulli_(0.5, generator=generator)
    projection.mul_(2.0).sub_(1.0)
    projection.div_(math.sqrt(float(output_dim)))
    projection.requires_grad_(False)
    return projection


@torch.no_grad()
def stein_imq_field(
    states: Tensor,
    target_scores: Tensor,
    projection: Tensor,
    *,
    kernel_offset: float = 1.0,
    beta: float = -0.5,
    normalize_field: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute a stop-gradient IMQ Stein/SVGD transport field.

    All particles in ``states`` should be samples for the same prompt. The
    returned field has the same shape as ``states`` and should be treated as a
    detached transport direction.
    """
    batch = states.shape[0]
    if batch <= 0:
        raise ValueError("states must contain at least one particle")
    x = states.detach().float().flatten(1)
    score = target_scores.detach().float().flatten(1)
    if x.shape != score.shape:
        raise ValueError(f"state/score shape mismatch: {tuple(x.shape)} vs {tuple(score.shape)}")
    if projection.shape[0] != x.shape[1]:
        raise ValueError(f"projection input {projection.shape[0]} != state dim {x.shape[1]}")

    state_rms = x.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    z = (x / state_rms) @ projection.float()

    diff = z[:, None, :] - z[None, :, :]
    dist2 = diff.square().sum(dim=-1)
    base = kernel_offset * kernel_offset + dist2
    kernel = base.pow(beta)

    attraction = kernel.transpose(0, 1) @ score
    attraction.div_(float(batch))

    grad_z = 2.0 * beta * base.pow(beta - 1.0)[..., None] * diff
    repulsion_z = grad_z.sum(dim=0).div_(float(batch))
    repulsion_x = (repulsion_z @ projection.float().transpose(0, 1)) / state_rms

    field = attraction + repulsion_x
    if normalize_field:
        field_rms = field.square().mean(dim=1, keepdim=True).sqrt()
        score_rms = score.square().mean(dim=1, keepdim=True).sqrt()
        field = field * (score_rms / field_rms.clamp_min(eps))
    return field.reshape_as(states)


def stein_transport_loss(differentiable_states: Tensor, stein_field: Tensor) -> Tensor:
    """Gradient descent on this loss moves states along ``+stein_field``."""
    return -(
        differentiable_states.float().flatten(1)
        * stein_field.detach().float().flatten(1)
    ).sum(dim=1).mean()


def choose_binary_ternary_rates(
    groups: Sequence[dict],
    *,
    target_bits: float,
    tolerance: float = 1e-4,
    max_lambda: float = 1e6,
) -> tuple[list[str], float, float]:
    """Rate-distortion water filling for binary/ternary group choices.

    Each group must provide binary/ternary damage and bits:
    ``binary_damage``, ``ternary_damage``, ``binary_bits``, ``ternary_bits``.
    Returns ``(choices, total_bits, lambda)``.
    """
    if not groups:
        return [], 0.0, 0.0
    lo, hi = 0.0, float(max_lambda)
    best_choices: list[str] | None = None
    best_bits = float("inf")
    best_lam = hi
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        choices: list[str] = []
        total_bits = 0.0
        for group in groups:
            cost_binary = float(group["binary_damage"]) + lam * float(group["binary_bits"])
            cost_ternary = float(group["ternary_damage"]) + lam * float(group["ternary_bits"])
            if cost_binary <= cost_ternary:
                rate = "binary"
            else:
                rate = "ternary"
            choices.append(rate)
            total_bits += float(group[f"{rate}_bits"])
        if total_bits > target_bits:
            lo = lam
        else:
            hi = lam
            best_choices = choices
            best_bits = total_bits
            best_lam = lam
        if abs(total_bits - target_bits) <= tolerance:
            best_choices = choices
            best_bits = total_bits
            best_lam = lam
            break
    if best_choices is None:
        best_choices = choices
        best_bits = total_bits
        best_lam = lam
    return best_choices, best_bits, best_lam


def fisher_diagonal_damage(delta: Tensor, fisher_diag: Tensor) -> Tensor:
    """Second-order local damage estimate: 0.5 * delta^T F delta."""
    return 0.5 * (delta.float().square() * fisher_diag.float()).sum()


def accumulate_fisher_diagonal(
    params: Iterable[Tensor],
    *,
    decay: float = 0.95,
    existing: Sequence[Tensor] | None = None,
) -> list[Tensor]:
    """Accumulate an empirical Fisher diagonal from current gradients."""
    out: list[Tensor] = []
    old = list(existing) if existing is not None else []
    for idx, p in enumerate(params):
        g = torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.detach().float()
        cur = g.square()
        if idx < len(old):
            cur = float(decay) * old[idx].to(cur.device).float() + (1.0 - float(decay)) * cur
        out.append(cur.detach())
    return out
