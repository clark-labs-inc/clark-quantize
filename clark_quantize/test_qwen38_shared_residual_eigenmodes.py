from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_shared_residual_eigenmodes_preregistration_v1.json"
METHOD = RUNTIME / "QWEN38_SHARED_RESIDUAL_EIGENMODES_V1.md"


def _payload() -> dict:
    return json.loads(PREREG.read_text())


def test_shared_eigenmode_pilot_is_material_and_qwen38_only() -> None:
    payload = _payload()
    assert payload["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["pilot_layers"] == [0, 32, 63]
    assert payload["scope"]["candidate_weight_scope"] == [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    assert payload["scope"]["complete_block_output_required"] is True
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True


def test_shared_eigenmode_pilot_has_exact_cpu_deployment_state() -> None:
    payload = _payload()
    approximation = payload["approximation"]
    assert approximation["rank"] == 64
    assert approximation["base_type"] == "exact upstream TQ2_0 blocks"
    assert approximation["stored_factor_type"].startswith("symmetric INT8")
    assert approximation["hard_forward_training"] is True
    assert "unquantized factor matrices" in approximation["forbidden_deployed_state"]
    assert "runtime reconstruction of a dense weight" in approximation["forbidden_deployed_state"]


def test_shared_eigenmode_gate_is_quality_and_runtime_material() -> None:
    payload = _payload()
    gate = payload["falsification_gate"]
    assert gate["minimum_complete_block_validation_nmse_reduction_vs_raw_tq2_mlp"] >= 0.70
    assert gate["maximum_candidate_complete_block_validation_nmse"] <= 1e-4
    assert gate["minimum_same_shape_decode_kernel_speedup_vs_q4"] >= 1.25
    assert gate["maximum_projected_residual_kernel_fraction_of_block_wall_time"] <= 0.10
    assert gate["fixed_bits_per_weight_ceiling"] is None


def test_shared_eigenmode_method_does_not_claim_known_math_as_novel() -> None:
    source = METHOD.read_text()
    payload = _payload()
    assert "weighted Eckart-Young solution" in source
    assert "It is not itself a novelty claim" in source
    assert len(payload["novelty_boundary"]["known_adjacent_work"]) >= 7
    assert payload["novelty_boundary"]["claim"].startswith("No novelty")


def test_shared_eigenmode_is_subordinate_to_100tps_program() -> None:
    payload = _payload()
    source = METHOD.read_text()
    assert "qwen38_27b_amd_100tps_goal_v1" in payload["program_role"]
    assert "not the primary route" in source
    assert "single-token kernel pass is not evidence for 100 tok/s" in source
    assert "100 accepted tokens per second" in payload["claim_boundary"]
