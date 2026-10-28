"""Exact block-orthogonal transforms for CPU-first Qwen3.8 ternary fitting."""
from __future__ import annotations

import math

import torch


def deterministic_rademacher(
    columns: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return reproducible {-1,+1} signs without persistent model payload."""
    if columns <= 0 or seed < 0:
        raise ValueError("columns must be positive and seed nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (columns,), generator=generator, dtype=torch.int8)
    return (bits.to(device=device, dtype=torch.float32) * 2.0 - 1.0).contiguous()


def normalized_hadamard(values: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform along the last axis."""
    if not values.is_floating_point() or values.ndim == 0:
        raise ValueError("Hadamard input must be a floating-point tensor")
    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    output = values.to(torch.float32).clone()
    step = 1
    while step < width:
        shaped = output.reshape(*output.shape[:-1], -1, step * 2)
        left = shaped[..., :step].clone()
        right = shaped[..., step:].clone()
        shaped[..., :step] = left + right
        shaped[..., step:] = left - right
        step *= 2
    return (output / math.sqrt(width)).contiguous()


def block_rht(
    values: torch.Tensor,
    *,
    group: int = 128,
    seed: int = 0,
    signs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-multiply row vectors by block-diagonal ``D H / sqrt(group)``.

    Applying the same function to activation row vectors produces the matching
    coordinates.  Consequently ``block_rht(W) @ block_rht(x) == W @ x`` when
    the first operand is interpreted as a matrix and the second as a vector.
    """
    if values.ndim < 1 or not values.is_floating_point():
        raise ValueError("block RHT expects a floating-point tensor")
    columns = values.shape[-1]
    if group <= 0 or group & (group - 1) or columns % group:
        raise ValueError("group must be a power of two dividing the last dimension")
    if signs is None:
        signs = deterministic_rademacher(columns, seed=seed, device=values.device)
    else:
        signs = signs.detach().to(device=values.device, dtype=torch.float32)
        if signs.ndim != 1 or signs.numel() != columns:
            raise ValueError("Rademacher signs must match the last dimension")
        if not torch.all((signs == -1) | (signs == 1)):
            raise ValueError("Rademacher signs must be exactly -1 or +1")
    signed = values.to(torch.float32) * signs
    grouped = signed.reshape(*signed.shape[:-1], columns // group, group)
    transformed = normalized_hadamard(grouped).reshape(values.shape)
    return transformed.contiguous(), signs.contiguous()


def inverse_block_rht(
    values: torch.Tensor,
    signs: torch.Tensor,
    *,
    group: int = 128,
) -> torch.Tensor:
    """Invert :func:`block_rht` for row vectors."""
    columns = values.shape[-1]
    if columns % group:
        raise ValueError("group must divide the last dimension")
    transformed = normalized_hadamard(
        values.to(torch.float32).reshape(*values.shape[:-1], columns // group, group)
    ).reshape(values.shape)
    return (transformed * signs.to(values.device, torch.float32)).contiguous()
