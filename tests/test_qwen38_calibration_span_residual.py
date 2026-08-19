from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from clark_quantize.qwen38_calibration_span_residual import (
    fit_float_residual_adapter,
    normalized_mse,
    quantize_residual_adapter,
)


REPO = Path(__file__).resolve().parents[1]
PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_calibration_span_residual_v36.json"
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


def test_v36_prereg_keeps_selection_out_of_fresh_confirmation() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["data"]["calibration_split"] == {
        "fit_rows": 48,
        "validation_rows": 16,
        "validation_rule": "Within the frozen ordered calibration-row vector, ordinals congruent to 3 modulo 4 are validation; all others are fit.",
        "same_split_all_layers": True,
    }
    assert prereg["training"]["candidate_ranks"] == [8, 16, 32, 48]
    assert prereg["training"]["candidate_ridge_fractions"] == [
        0.0001,
        0.001,
        0.01,
    ]
    fresh = prereg["data"]["fresh_confirmation"]
    assert fresh["collected_after_preregistration"] is True
    assert fresh["checkpoint_or_hyperparameter_selection_forbidden"] is True
    assert fresh["prior_confirmation_cache_forbidden"] is True
    assert prereg["training"]["per_layer_hyperparameter_selection_forbidden"] is True


def test_v36_rank_32_projection_matches_storage_and_mac_formula() -> None:
    prereg = json.loads(PREREG.read_text())
    hidden = prereg["scope"]["hidden_size"]
    rank = 32
    per_layer_bytes = 2 * hidden * rank + 4 * rank + 2 * hidden
    per_layer_macs = 2 * hidden * rank
    projection = prereg["storage_and_compute"][
        "rank_32_complete_64_layer_projection"
    ]
    assert projection["persistent_bytes"] == 64 * per_layer_bytes
    assert projection["macs_per_token"] == 64 * per_layer_macs
    assert projection["weight_traffic_fraction_vs_base_compute_image"] < 0.002
