from __future__ import annotations

import pytest
import torch

from clark_quantize.qwen38_calibration_span_residual import (
    fit_float_residual_adapter,
    normalized_mse,
    quantize_residual_adapter,
)


def synthetic_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260819)
    rows, inputs, outputs, rank = 64, 40, 32, 4
    hidden = torch.randn(rows, inputs, generator=generator)
    left = torch.randn(inputs, rank, generator=generator) * 0.2
    right = torch.randn(rank, outputs, generator=generator) * 0.2
    bias = torch.randn(outputs, generator=generator) * 0.05
    candidate = torch.randn(rows, outputs, generator=generator)
    target = candidate + hidden.matmul(left).matmul(right) + bias
    return hidden, candidate, target


def test_float_calibration_span_recovers_low_rank_error() -> None:
    hidden, candidate, target = synthetic_problem()
    adapter = fit_float_residual_adapter(
        hidden,
        candidate,
        target,
        rank=4,
        ridge_fraction=1e-8,
    )
    corrected = candidate + adapter.correction(hidden)
    assert adapter.rank == 4
    assert normalized_mse(corrected, target) < 1e-10


def test_int8_factors_preserve_most_of_the_geometric_correction() -> None:
    hidden, candidate, target = synthetic_problem()
    direct_nmse = normalized_mse(candidate, target)
    adapter = quantize_residual_adapter(
        fit_float_residual_adapter(
            hidden,
            candidate,
            target,
            rank=4,
            ridge_fraction=1e-8,
        )
    )
    corrected_nmse = normalized_mse(candidate + adapter.correction(hidden), target)
    assert corrected_nmse < direct_nmse * 0.001
    assert adapter.persistent_bytes == (
        adapter.left_codes.numel()
        + adapter.right_codes.numel()
        + 2 * adapter.left_scales.numel()
        + 2 * adapter.right_scales.numel()
        + 2 * adapter.bias.numel()
    )


def test_fit_rejects_invalid_rank_and_zero_energy() -> None:
    hidden, candidate, target = synthetic_problem()
    with pytest.raises(ValueError, match="rank must be"):
        fit_float_residual_adapter(
            hidden, candidate, target, rank=64, ridge_fraction=1e-4
        )
    with pytest.raises(ValueError, match="zero centered energy"):
        fit_float_residual_adapter(
            torch.ones_like(hidden),
            candidate,
            target,
            rank=4,
            ridge_fraction=1e-4,
        )
