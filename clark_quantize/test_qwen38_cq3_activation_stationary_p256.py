from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_activation_stationary_p256_v20.json"
PATCH = RUNTIME / "kernels/llama_cpp_cq3_activation_stationary_p256_v20.patch"
RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_cq3_activation_stationary_p256_v20/terminal-decision.json"
)


def test_v20_is_storage_neutral_qwen38_27b_only() -> None:
    prereg = json.loads(PREREG.read_text())
    scope = prereg["scope"]
    assert scope["model"] == "Qwen/Qwen3.8-27B"
    assert scope["qwen_4b_compute_forbidden"] is True
    assert scope["weight_bits_per_weight"] == 2.125
    layout = prereg["runtime_layout"]
    assert layout["rows_per_panel"] == 256
    assert layout["source_bytes_per_matrix"] == layout["candidate_bytes_per_matrix"]
    assert layout["storage_expansion_fraction"] == 0.0
    assert 17_408 % layout["rows_per_panel"] == 0


def test_v20_changes_reuse_axis_materially() -> None:
    prereg = json.loads(PREREG.read_text())
    cost = prereg["static_cost_model"]
    assert cost["x8_activation_broadcasts_per_matrix"] == 11_141_120
    assert cost["candidate_activation_broadcasts_per_matrix"] == 69_632
    assert cost["broadcast_construction_reduction"] == 160.0
    assert cost["candidate_int32_scratch_bytes"] == 1_024
    assert cost["not_a_speed_claim"] is True


def test_v20_freezes_exact_paired_speed_and_stability_gates() -> None:
    prereg = json.loads(PREREG.read_text())
    gate = prereg["sequential_gates"]["gate_1_exact_shape_paired_microbenchmark"]
    assert gate["minimum_median_x8_over_candidate_pairwise_ratio"] == 1.10
    assert gate["minimum_median_ordinary_over_candidate_pairwise_ratio"] == 1.43
    assert gate["maximum_x8_over_candidate_pairwise_ratio_mad_fraction"] == 0.03
    assert gate["maximum_ordinary_over_candidate_pairwise_ratio_mad_fraction"] == 0.03
    assert prereg["measurement"]["measured_pairs_per_comparison"] == 21


def test_v20_native_patch_has_panel_major_kernel_when_present() -> None:
    if not PATCH.exists():
        return
    source = PATCH.read_text()
    assert "CQ3_AS_PANEL_ROWS 256" in source
    assert "ggml_repack_cq3_0_to_cq3_0p256" in source
    assert "ggml_gemv_cq3_0p256_cq8_0" in source
    assert "scratch" in source
    assert "_mm256_maddubs_epi16" in source


def test_v20_receipt_obeys_frozen_decision_if_present() -> None:
    if not RECEIPT.exists():
        return
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["model"] == "Qwen/Qwen3.8-27B"
    checks = receipt["checks"]
    passed = all(
        (
            checks["x8_candidate_bit_exact"],
            checks["median_x8_over_candidate_pairwise_ratio_1_10_met"],
            checks["median_ordinary_over_candidate_pairwise_ratio_1_43_met"],
            checks["x8_over_candidate_pairwise_ratio_mad_fraction_0_03_met"],
            checks["ordinary_over_candidate_pairwise_ratio_mad_fraction_0_03_met"],
            checks["all_frequency_samples_positive"],
        )
    )
    assert receipt["passed"] is passed
    assert receipt["kernel_promoted"] is passed
