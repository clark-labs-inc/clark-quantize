from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1.json"
)


def payload() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def test_mixed_v1_is_full_qwen38_scope_and_not_a_small_demo() -> None:
    data = payload()
    assert data["status"] == "frozen_after_terminal_qat_before_mixed_execution"
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["text_layers"] == 64
    assert data["scope"]["qwen_4b_compute_forbidden"] is True
    representation = data["representation"]
    assert representation["converted_parameters"] == 19_126_026_240
    assert representation["converted_fraction_of_text_graph"] >= 0.70
    assert representation["mlp"]["tensors"] == 192
    assert representation["attention_output"]["tensors"] == 64


def test_mixed_v1_mode_map_follows_calibration_only_sensitivity() -> None:
    data = payload()["calibration_only_mode_selection"]
    values = data["aggregate_calibration_nmse_across_layers_0_32_63"]
    assert min(values, key=values.get) == "attention_output"
    assert max(values, key=values.get) == "attention_input"
    modes = data["selected_modes"]
    assert modes["attention_output"] == "AFF3_X16 register-lookup v18"
    assert modes["mlp_input"].startswith("CQ3_0x16")
    assert modes["mlp_down"].startswith("CQ3_0x16")
    assert modes["attention_input"] == "unchanged Q4_K_M control types"
    assert data["selection_was_frozen_before_new_confirmation_collection"] is True


def test_mixed_v1_counts_every_runtime_byte() -> None:
    data = payload()
    storage = data["complete_runtime_storage_accounting"]
    assert storage["q4_control_file_bytes"] == 18_973_870_432
    assert storage["selected_control_payload_bytes"] == 11_643_125_760
    assert storage["selected_candidate_compute_image_bytes"] == 5_068_296_192
    assert storage["projected_complete_compute_image_bytes_before_alignment"] == 12_399_040_864
    assert storage["maximum_complete_compute_image_bytes_after_alignment"] == 12_500_000_000
    assert storage["all_layouts_anchors_codebooks_rotations_alignment_metadata_and_unchanged_tensors_count"] is True
    assert data["representation"]["mlp"]["effective_runtime_bpw"] == 2.125
    assert data["representation"]["attention_output"][
        "runtime_bpw_including_anchors_codebooks_and_rotations"
    ] < 2.08


def test_mixed_v1_uses_a_fresh_sealed_confirmation_split() -> None:
    confirmation = payload()["data"]["fresh_confirmation"]
    assert len(confirmation["texts"]) == 4
    assert confirmation["selected_rows_per_layer"] == 32
    assert confirmation["target_layers"] == [0, 32, 63]
    assert confirmation["collected_after_preregistration"] is True
    assert confirmation["may_score_only_the_final_frozen_candidate"] is True
    assert confirmation["checkpoint_or_hyperparameter_selection_forbidden"] is True
    assert "CUTE" in confirmation["evaluation_contamination_forbidden"]


def test_mixed_v1_training_is_hard_forward_and_confirmation_blind() -> None:
    training = payload()["training"]
    assert training["mlp_cq3"]["steps_per_row_shard"] == 6
    assert training["mlp_cq3"]["group_size"] == 128
    assert "calibration-loss" in training["mlp_cq3"]["selection"]
    assert training["attention_output_aff3"]["frozen_state"].startswith(
        "ternary codes"
    )
    assert training["attention_output_aff3"]["outer_steps"] == 8
    assert training["attention_output_aff3"]["initial_global_l2_radius"] == 0.02
    assert "fresh confirmation remains sealed" in training[
        "attention_output_aff3"
    ]["checkpoint_selection"]
    assert "One mode map" in training["no_sweep"]


def test_mixed_v1_has_block_kernel_quality_performance_and_energy_gates() -> None:
    gates = payload()["sequential_gates"]
    block = gates["gate_1_trained_three_block_numeric"]
    assert block["minimum_mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed"] >= 0.01
    assert block["maximum_final_fresh_confirmation_complete_block_nmse_per_layer"] <= 0.05
    kernel = gates["gate_2_native_kernel"]
    assert kernel["minimum_cq3_x8_over_x16_speedup"] >= 1.08
    assert kernel["small_batch_widths"] == [1, 2, 4, 6, 8]
    quality = gates["gate_3_complete_model_quality"]
    assert quality["cute_correct_of_20_minimum"] >= 9
    assert quality["math500_correct_of_32_minimum"] >= 8
    assert quality["mmlu_correct_of_64_minimum"] >= 50
    assert quality["wikitext_maximum_perplexity_ratio_vs_q4"] <= 1.05
    performance = gates["gate_4_complete_model_performance"]
    assert performance["breakthrough_minimum_decode_tokens_per_second"] >= 2.75
    assert performance["breakthrough_minimum_decode_speedup_vs_q4"] >= 1.4
    assert performance["stretch_decode_tokens_per_second"] >= 3.2
    assert performance["energy_measurement_required_for_breakthrough_claim"] is True


def test_mixed_v1_cannot_promote_partial_evidence() -> None:
    data = payload()
    assert "complete 64-layer artifact" in data["claim_boundary"]
    rules = data["stop_rules"]
    assert any("Do not infer an unrun score" in rule for rule in rules)
    assert any("Infrastructure-invalid" in rule for rule in rules)
    assert any("silently fall back" in rule for rule in rules)
