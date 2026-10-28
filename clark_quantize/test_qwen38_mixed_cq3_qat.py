from __future__ import annotations

import inspect

import pytest
import torch

from clark_quantize.qwen38_layer0_qat import (
    RowShardTrainingConfig,
    pack_cq3_rows,
    unpack_cq3_rows,
)
from clark_quantize.qwen38_mixed_cq3_qat import (
    aggregate_calibration_metrics,
    train_row_shard_calibration_only,
)


def test_mixed_cq3_training_api_has_no_confirmation_or_validation_input() -> None:
    signature = inspect.signature(train_row_shard_calibration_only)
    assert "validation_inputs" not in signature.parameters
    assert "confirmation_inputs" not in signature.parameters
    assert "calibration_inputs" in signature.parameters


def test_mixed_cq3_selects_a_hard_calibration_state_and_roundtrips() -> None:
    torch.manual_seed(7)
    source = torch.randn(4, 256) * 0.02
    inputs = torch.randn(12, 256)
    config = RowShardTrainingConfig(
        steps=2,
        learning_rate=1e-4,
        gradient_clip_norm=1.0,
        weight_reconstruction_weight=0.01,
        group_size=128,
    )
    codes, scales, metrics = train_row_shard_calibration_only(
        source, inputs, config=config
    )
    assert codes.shape == source.shape
    assert scales.shape == (4, 2)
    assert codes.dtype == torch.int8
    assert metrics["selection_data"] == "calibration_only"
    assert metrics["selected_step"] in (0, 1, 2)
    assert metrics["selected_calibration_nmse"] <= metrics[
        "step_zero_calibration_nmse"
    ]
    assert metrics["steps_completed"] == 2
    payload = pack_cq3_rows(codes, scales)
    rebuilt_codes, rebuilt_scales = unpack_cq3_rows(payload, rows=4, columns=256)
    assert torch.equal(codes, rebuilt_codes)
    assert torch.equal(scales, rebuilt_scales)


def test_mixed_cq3_supports_distinct_teacher_inputs_for_down_projection() -> None:
    torch.manual_seed(11)
    source = torch.randn(2, 256) * 0.03
    candidate_inputs = torch.randn(8, 256)
    teacher_inputs = candidate_inputs + 0.01 * torch.randn_like(candidate_inputs)
    codes, scales, metrics = train_row_shard_calibration_only(
        source,
        candidate_inputs,
        teacher_calibration_inputs=teacher_inputs,
        config=RowShardTrainingConfig(steps=1),
    )
    assert codes.shape == source.shape
    assert scales.shape == (2, 2)
    assert metrics["distinct_teacher_inputs"] is True
    assert metrics["finite"] is True


def test_mixed_cq3_aggregate_uses_exact_energy_denominator() -> None:
    shards = [
        {
            "rows": 2,
            "accepted": True,
            "finite": True,
            "step_zero_calibration_squared_error_sum": 4.0,
            "selected_calibration_squared_error_sum": 3.0,
            "calibration_target_squared_sum": 20.0,
            "selected_code_churn_fraction": 0.02,
        },
        {
            "rows": 6,
            "accepted": False,
            "finite": True,
            "step_zero_calibration_squared_error_sum": 6.0,
            "selected_calibration_squared_error_sum": 6.0,
            "calibration_target_squared_sum": 30.0,
            "selected_code_churn_fraction": 0.0,
        },
    ]
    result = aggregate_calibration_metrics(shards)
    assert result["step_zero_calibration_nmse"] == 10.0 / 50.0
    assert result["selected_calibration_nmse"] == 9.0 / 50.0
    assert result["relative_calibration_nmse_improvement"] == pytest.approx(0.1)
    assert result["weighted_selected_code_churn_fraction"] == 0.005
    assert result["selection_data"] == "calibration_only"
