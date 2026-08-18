from __future__ import annotations

import pytest
import torch

from clark_quantize.qwen38_activation_dual_binary import (
    activation_dual_outputs,
    fit_activation_dual_binary_residual,
    ridge_lift_activation_features,
)
from clark_quantize.qwen38_binary_low_rank_residual import (
    binary_right_values,
    quantize_binary_rows,
)


def test_dual_lift_recovers_features_when_activations_have_full_column_rank() -> None:
    generator = torch.Generator().manual_seed(4121)
    activations = torch.randn(96, 24, generator=generator)
    right = torch.randn(5, 24, generator=generator)
    features = activations @ right.T
    lifted = ridge_lift_activation_features(
        activations, features, relative_ridge=1e-8
    )
    reconstructed = activations @ lifted.T
    assert torch.allclose(reconstructed, features, rtol=2e-4, atol=2e-4)


def test_activation_dual_fit_recovers_shared_binary_output_subspace() -> None:
    generator = torch.Generator().manual_seed(4122)
    activations = torch.randn(128, 32, generator=generator)
    true_right = quantize_binary_rows(torch.randn(4, 32, generator=generator))
    features = activations @ binary_right_values(true_right).T
    left = (
        torch.randint(-6, 7, (13, 4), generator=generator).to(torch.float32)
        * torch.linspace(0.02, 0.08, 13)[:, None],
        torch.randint(-6, 7, (11, 4), generator=generator).to(torch.float32)
        * torch.linspace(0.03, 0.07, 11)[:, None],
    )
    targets = tuple(features @ values.T for values in left)
    fit = fit_activation_dual_binary_residual(
        activations,
        targets,
        rank=4,
        iterations=4,
        pca_iterations=2,
        seed=4123,
        relative_ridge=1e-7,
    )
    predicted = activation_dual_outputs(
        activations, fit.left_coefficients, fit.right
    )
    source_error = sum(
        float(target.square().sum(dtype=torch.float64).item())
        for target in targets
    )
    error = sum(
        float((target - observed).square().sum(dtype=torch.float64).item())
        for target, observed in zip(targets, predicted)
    )
    assert error / source_error < 0.08
    assert min(fit.objective_history) == fit.objective_history[fit.best_iteration - 1]
    assert fit.initial_squared_error == pytest.approx(source_error)
    assert [values.shape for values in predicted] == [values.shape for values in targets]


def test_activation_dual_fit_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(4124)
    activations = torch.randn(40, 18, generator=generator)
    targets = (torch.randn(40, 15, generator=generator),)
    first = fit_activation_dual_binary_residual(
        activations, targets, rank=3, iterations=3, seed=4125
    )
    second = fit_activation_dual_binary_residual(
        activations, targets, rank=3, iterations=3, seed=4125
    )
    assert first.objective_history == second.objective_history
    assert torch.equal(first.right.signs, second.right.signs)
    assert torch.equal(first.right.scales, second.right.scales)
    assert torch.equal(first.left_codes[0], second.left_codes[0])
    assert torch.equal(first.left_scales[0], second.left_scales[0])


def test_activation_dual_fit_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="invalid activation-dual"):
        fit_activation_dual_binary_residual(
            torch.ones(4, 8), (torch.ones(3, 7),), rank=2
        )
