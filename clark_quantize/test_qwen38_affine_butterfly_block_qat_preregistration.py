from __future__ import annotations

import json
from pathlib import Path


PREREG = (
    Path(__file__).resolve().parents[1]
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_affine_butterfly_block_qat_preregistration_v1.json"
)


def test_qat_prereg_is_qwen27b_only_and_frozen_before_parent_terminal() -> None:
    data = json.loads(PREREG.read_text())
    assert data["status"] == "frozen_before_parent_ptq_terminal_result"
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["qwen_4b_compute_forbidden"] is True
    assert data["scope"]["pilot_layers"] == [0, 32, 63]
    assert data["training"]["steps"] == 8
    assert data["training"]["checkpoint_selection"].startswith("The eighth")
    assert "no_sweep" in data["training"]


def test_qat_only_trains_state_that_survives_into_the_file() -> None:
    data = json.loads(PREREG.read_text())
    representation = data["trained_representation"]
    for required in ("row anchors", "codebooks", "Givens angles"):
        assert required in representation["trainable_continuous_state"]
    for required in ("Ternary assignments", "4-bit alpha/mu indices"):
        assert required in representation["trainable_discrete_state"]
    for forbidden in ("LoRA", "dense residual", "latent shadow weight"):
        assert forbidden in representation["forbidden_state"]


def test_qat_gate_is_heldout_storage_honest_and_sensitive_site_complete() -> None:
    data = json.loads(PREREG.read_text())
    sensitivity = data["sensitivity_protocol"]
    assert "depthwise_convolution" in sensitivity["sites_linear_attention"]
    assert "validation" in sensitivity["selection_rule"]
    assert "No higher-precision exception" in sensitivity["mixed_mode_boundary"]
    gate = data["falsification_gate"]
    assert gate["minimum_relative_validation_nmse_reduction_vs_direct_ternary"] == 0.3
    assert gate["maximum_regression_vs_parent_ptq_candidate"] == 0.0
    assert gate["maximum_projected_complete_text_file_bpw"] == 1.7
    budget = data["whole_model_budget"]
    assert budget["base_projected_complete_bundle_bpw"] < budget["absolute_ceiling_bpw"]
    assert budget["remaining_bytes_below_ceiling"] == 109754911
