"""Function-preserving orthogonal shaping controls."""

from __future__ import annotations

import math

import torch


def validate_block_size(features: int, block_size: int) -> None:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    if features % block_size:
        raise ValueError(
            f"feature dimension {features} is not divisible by block size {block_size}"
        )


def block_hadamard(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply a normalized Sylvester Hadamard matrix along the final dimension."""

    validate_block_size(tensor.shape[-1], block_size)
    original_shape = tensor.shape
    block_shape = (*original_shape[:-1], original_shape[-1] // block_size)
    values = tensor.reshape(*block_shape, block_size)
    width = 1
    while width < block_size:
        paired = values.reshape(*block_shape, -1, 2, width)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        values = torch.cat((left + right, left - right), dim=-1).reshape(
            *block_shape,
            block_size,
        )
        width *= 2
    return values.reshape(original_shape) / math.sqrt(block_size)
