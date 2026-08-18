from __future__ import annotations

import torch
import torch.nn.functional as functional

from clark_quantize.qwen38_affine_butterfly import (
    activation_metric_affine_ternary,
    affine_ternary_linear,
    apply_givens_product,
    deterministic_givens_pairings,
    fit_shared_affine_rotation,
    initialize_affine_ternary,
)
from clark_quantize.ternary import ternary_codes_scales


def test_givens_product_is_orthogonal_invertible_and_differentiable() -> None:
    torch.manual_seed(4)
    source = torch.randn(3, 16)
    angles = (torch.randn(4, 8) * 0.1).requires_grad_()
    pairings = deterministic_givens_pairings(16, stages=4, seed=20260818)
    rotated = apply_givens_product(source, angles, pairings)
    restored = apply_givens_product(rotated, angles, pairings, inverse=True)
    torch.testing.assert_close(rotated.square().sum(-1), source.square().sum(-1))
    torch.testing.assert_close(restored, source, atol=2e-6, rtol=2e-6)
    rotated.square().sum().backward()
    assert angles.grad is not None
    assert bool(torch.isfinite(angles.grad).all())


def test_shared_rotation_preserves_exact_linear_map_before_quantization() -> None:
    torch.manual_seed(5)
    weight = torch.randn(7, 16)
    inputs = torch.randn(11, 16)
    angles = torch.randn(3, 8) * 0.2
    pairings = deterministic_givens_pairings(16, stages=3, seed=99)
    rotated_weight = apply_givens_product(weight, angles, pairings)
    rotated_inputs = apply_givens_product(inputs, angles, pairings)
    torch.testing.assert_close(
        functional.linear(rotated_inputs, rotated_weight),
        functional.linear(inputs, weight),
        atol=3e-5,
        rtol=3e-5,
    )


def test_affine_initializer_has_three_states_and_never_loses_to_symmetric() -> None:
    torch.manual_seed(6)
    weight = torch.randn(9, 260) * 0.4 + 0.7
    candidate = initialize_affine_ternary(weight, group_size=128)
    rebuilt = candidate.reconstruct()
    direct_codes, direct_scales = ternary_codes_scales(weight, group=128, method="optimal")
    pad = 124
    direct = (
        torch.nn.functional.pad(direct_codes.float(), (0, pad)).reshape(9, 3, 128)
        * direct_scales.float().unsqueeze(-1)
    ).reshape(9, -1)[:, :260]
    assert candidate.codes.dtype == torch.int8
    assert set(candidate.codes.unique().tolist()) <= {-1, 0, 1}
    assert bool((candidate.alpha >= 0).all())
    assert float((rebuilt - weight).square().sum()) <= float((direct - weight).square().sum()) + 1e-4


def test_affine_group_sum_runtime_matches_reconstructed_linear_for_partial_group() -> None:
    torch.manual_seed(7)
    weight = torch.randn(5, 133) + 0.25
    inputs = torch.randn(4, 133)
    candidate = initialize_affine_ternary(weight, group_size=128)
    observed = affine_ternary_linear(inputs, candidate)
    expected = functional.linear(inputs, candidate.reconstruct())
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)


def test_activation_metric_relocation_visits_all_coordinates_and_reduces_error() -> None:
    torch.manual_seed(71)
    inputs = torch.randn(9, 19)
    # Correlation makes the exact activation metric materially different from
    # independent Euclidean weight error.
    inputs[:, 8:16] = inputs[:, :8] * 0.8 + inputs[:, 8:16] * 0.2
    weight = torch.randn(5, 19) * 0.3 + 0.2
    initial = initialize_affine_ternary(weight, group_size=8)
    relocated = activation_metric_affine_ternary(weight, inputs, group_size=8)
    target = functional.linear(inputs, weight)
    initial_error = (functional.linear(inputs, initial.reconstruct()) - target).square().sum()
    relocated_error = (
        functional.linear(inputs, relocated.reconstruct()) - target
    ).square().sum()
    assert relocated.codes.shape == weight.shape
    assert set(relocated.codes.unique().tolist()) <= {-1, 0, 1}
    assert bool((relocated.alpha >= 0).all())
    assert float(relocated_error) <= float(initial_error) + 1e-4


def test_shared_rotation_fit_covers_every_consumer_row_and_keeps_validation_disjoint() -> None:
    torch.manual_seed(8)
    weights = {
        "q_proj": torch.randn(4, 128) * 0.2 + 0.1,
        "k_proj": torch.randn(3, 128) * 0.3 - 0.2,
    }
    calibration = torch.randn(6, 128)
    validation = torch.randn(5, 128)
    pairings = deterministic_givens_pairings(128, stages=2, seed=8)
    fitted = fit_shared_affine_rotation(
        weights,
        calibration,
        validation,
        pairings=pairings,
        steps=2,
        learning_rate=2e-3,
        row_shard=2,
        group_size=128,
    )
    assert fitted.metrics["consumer_count"] == 2
    assert fitted.metrics["rows"] == 7
    assert fitted.metrics["parameters"] == 7 * 128
    assert fitted.metrics["calibration_rows"] == 6
    assert fitted.metrics["validation_rows"] == 5
    assert all(row["processed_rows"] == 7 for row in fitted.metrics["history"])
    assert any(row["gradient_norm_before_clip"] > 1e-8 for row in fitted.metrics["history"])
    assert fitted.metrics["finite"] is True
    assert set(fitted.quantized) == set(weights)
    torch.testing.assert_close(
        fitted.angles,
        fitted.angles.to(torch.float16).to(torch.float32),
        rtol=0,
        atol=0,
    )
