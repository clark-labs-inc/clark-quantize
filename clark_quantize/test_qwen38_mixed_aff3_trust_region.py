from __future__ import annotations

import inspect

import pytest
import torch

from clark_quantize.qwen38_affine_qat import hard_fp16_ste
from clark_quantize.qwen38_mixed_aff3_trust_region import (
    calibration_only_hard_forward_trust_region,
)


def test_aff3_trust_region_api_has_no_holdout_input() -> None:
    parameters = inspect.signature(
        calibration_only_hard_forward_trust_region
    ).parameters
    assert "validation" not in " ".join(parameters)
    assert "confirmation" not in " ".join(parameters)
    assert "loss_closure" in parameters


def test_aff3_trust_region_is_monotone_and_uses_the_first_proposal() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    def loss() -> torch.Tensor:
        return hard_fp16_ste(parameter).square().sum()

    result = calibration_only_hard_forward_trust_region(
        [parameter],
        loss,
        lambda: None,
        outer_steps=3,
        global_l2_radius=0.2,
        backtracking_factors=(1.0, 0.5, 0.25),
    )
    accepted = [
        item["accepted_hard_forward_calibration_nmse"] for item in result["history"]
    ]
    assert accepted == sorted(accepted, reverse=True)
    assert all(item["accepted_factor"] == 1.0 for item in result["history"])
    assert result["final_hard_forward_calibration_nmse"] < result[
        "initial_hard_forward_calibration_nmse"
    ]
    assert result["outer_steps_completed"] == 3
    assert result["selection_data"] == "calibration_only"


def test_aff3_trust_region_restores_a_rejected_discrete_proposal() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.1]))

    def loss() -> torch.Tensor:
        hard = parameter + (torch.sign(parameter) - parameter).detach()
        return (hard - 0.5).square().sum()

    before = parameter.detach().clone()
    result = calibration_only_hard_forward_trust_region(
        [parameter],
        loss,
        lambda: None,
        outer_steps=1,
        global_l2_radius=2.0,
        backtracking_factors=(1.0,),
    )
    assert torch.equal(parameter, before)
    assert result["accepted_steps"] == 0
    assert result["history"][0]["proposals"][0]["accepted"] is False
    assert result["final_hard_forward_calibration_nmse"] == pytest.approx(0.25)


def test_aff3_trust_region_rejects_bad_schedules() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="configuration"):
        calibration_only_hard_forward_trust_region(
            [parameter],
            lambda: parameter.square().sum(),
            lambda: None,
            outer_steps=1,
            global_l2_radius=0.1,
            backtracking_factors=(0.5, 1.0),
        )
