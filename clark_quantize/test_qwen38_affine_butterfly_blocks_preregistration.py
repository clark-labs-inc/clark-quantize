from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_affine_butterfly_blocks_preregistration_v1.json"
)


def load_preregistration() -> dict:
    return json.loads(PREREGISTRATION.read_text())


def test_affine_butterfly_gate_is_complete_block_and_qwen38_only() -> None:
    payload = load_preregistration()
    scope = payload["scope"]
    assert scope["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert scope["text_graph_parameters"] == 26_895_998_464
    assert scope["pilot_layers"] == [0, 32, 63]
    assert scope["pilot_layer_types"] == [
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert scope["qwen_4b_compute_forbidden"] is True
    assert "Every rank-2-or-higher weight" in scope["coverage"]


def test_affine_base3_has_honest_whole_model_storage_and_cpu_algebra() -> None:
    payload = load_preregistration()
    representation = payload["representation"]
    projection = representation["complete_text_projection"]
    assert "exactly three reconstructed values" in representation["weight_equation"]
    assert "203-bit base-3 integer" in representation["ternary_packing"]
    assert "five-trits-per-byte radix-5" in payload["pre_execution_correction"]["change"]
    assert payload["pre_execution_correction"]["execution_had_started"] is False
    assert "sum(x[g])" in representation["runtime_equation"]
    assert projection["maximum_projected_complete_bundle_bpw"] <= 1.70
    assert projection["maximum_rotation_and_auxiliary_bpw"] <= 0.10
    assert projection["maximum_projected_complete_bundle_bytes"] * 8 / 26_895_998_464 == projection[
        "maximum_projected_complete_bundle_bpw"
    ]


def test_affine_butterfly_is_one_shot_falsifiable() -> None:
    payload = load_preregistration()
    fit = payload["fit"]
    gate = payload["falsification_gate"]
    assert fit["rotation_stages"] == 6
    assert fit["rotation_steps"] == 8
    assert fit["no_sweep"].startswith("Layer set")
    assert gate["minimum_relative_validation_nmse_reduction_vs_direct_ternary"] == 0.30
    assert gate["requires_reduction_on_every_selected_block"] is True
    assert gate["reject_action"].startswith("Terminate affine-butterfly")
    assert payload["whole_model_promotion_gates"]["runtime"][
        "minimum_paired_end_to_end_speedup"
    ] == 1.15
