from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from clark_quantize.qwen38_calibration_span_residual import (
    expected_int8_adapter_bytes,
    fit_float_residual_adapter,
    load_int8_adapter,
    normalized_mse,
    quantize_residual_adapter,
    write_int8_adapter,
)


REPO = Path(__file__).resolve().parents[1]
PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_calibration_span_residual_v36.json"
)
RUNTIME = PREREG.parent
SHELL_RUNNER = RUNTIME / "run_qwen38_27b_calibration_span_residual_v36.sh"
sys.path.insert(0, str(RUNTIME))

import run_qwen38_27b_calibration_span_residual_v36 as runner


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


def test_int8_payload_roundtrip_is_exact(tmp_path: Path) -> None:
    hidden, candidate, target = synthetic_problem()
    adapter = quantize_residual_adapter(
        fit_float_residual_adapter(
            hidden,
            candidate,
            target,
            rank=4,
            ridge_fraction=1e-4,
        )
    )
    path = tmp_path / "adapter.csr"
    write_int8_adapter(path, adapter)
    restored = load_int8_adapter(
        path,
        input_features=hidden.shape[1],
        output_features=target.shape[1],
        rank=4,
        ridge_fraction=1e-4,
    )
    assert path.stat().st_size == expected_int8_adapter_bytes(40, 32, 4)
    assert path.stat().st_size == adapter.persistent_bytes
    assert torch.equal(restored.left_codes, adapter.left_codes)
    assert torch.equal(restored.left_scales, adapter.left_scales)
    assert torch.equal(restored.right_codes, adapter.right_codes)
    assert torch.equal(restored.right_scales, adapter.right_scales)
    assert torch.equal(restored.bias, adapter.bias)


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


def test_v36_context_replay_slices_to_sequences_containing_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        layer_types = ["full_attention"]

    captured: dict[str, tuple[int, ...]] = {}

    def fake_context(config, hidden, attention_mask, position_ids):
        captured["hidden"] = tuple(hidden.shape)
        captured["attention_mask"] = tuple(attention_mask.shape)
        captured["position_ids"] = tuple(position_ids.shape)
        return "positions", {"full_attention": "mask"}

    monkeypatch.setattr(runner.qat, "make_context", fake_context)
    cache = {
        "layers": {
            "0": {
                "input": torch.zeros(12, 64, 2),
                "output": torch.ones(12, 64, 2),
            }
        },
        "attention_mask": torch.ones(12, 64),
        "text_position_ids": torch.arange(64).repeat(12, 1),
    }
    hidden, parent, positions, mask, position_ids = runner.load_context(
        Config(), cache, 0, selected=torch.tensor([0, 7 * 64 + 11])
    )
    assert hidden.shape == parent.shape == (8, 64, 2)
    assert position_ids.shape == (8, 64)
    assert positions == "positions"
    assert mask == "mask"
    assert captured == {
        "hidden": (8, 64, 2),
        "attention_mask": (8, 64),
        "position_ids": (8, 64),
    }


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
    assert prereg["training"]["candidate_ranks"] == [8, 16, 32, 47]
    assert prereg["pre_execution_correction"]["measurement_state"].startswith(
        "no v36"
    )
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


def test_v36_shell_runner_preserves_holdout_sequence_and_disk_bounds() -> None:
    source = SHELL_RUNNER.read_text()
    calibration = source.index("--stage calibration")
    holdout = source.index(
        'taskset -c 0,2,4,6 "$python_bin" "$collector"', calibration
    )
    confirmation = source.index("--stage confirmation")
    assert calibration < holdout < confirmation
    assert "EXPECTED_REPOSITORY_COMMIT" in source
    assert "EXPERIMENT_COMPLETE" in source
    assert "artifacts.sha256" in source
    assert "! -name artifacts.sha256.tmp" in source
    assert "another llama.cpp workload is active" in source
    assert "pgrep -f '^/workspace/venvs/quant-cpu/bin/python" in source
