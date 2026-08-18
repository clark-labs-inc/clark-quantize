from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_avx2_dual_chain_v11.json"
PATCH = RUNTIME / "kernels/llama_cpp_cq3_avx2_dual_chain_v11.patch"
RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_cq3_avx2_dual_chain_v11/terminal-decision.json"
)
V12_PREREG = RUNTIME / "qwen38_27b_cq3_avx2_dual_chain_v12.json"
V12_HARNESS_PATCH = (
    RUNTIME / "kernels/llama_cpp_cq3_avx2_dual_chain_tolerant_harness_v12.patch"
)
V12_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_cq3_avx2_dual_chain_v12/terminal-decision.json"
)
V13_PREREG = (
    RUNTIME / "qwen38_27b_cq3_avx2_dual_chain_characterization_v13.json"
)
V13_PATCH = (
    RUNTIME / "kernels/llama_cpp_cq3_avx2_dual_chain_characterization_v13.patch"
)
V13_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_cq3_avx2_dual_chain_characterization_v13/terminal-decision.json"
)
V14_PREREG = RUNTIME / "qwen38_27b_cq3_avx2_dual_decode_v14.json"
V14_PATCH = RUNTIME / "kernels/llama_cpp_cq3_avx2_dual_decode_v14.patch"


def test_dual_chain_prereg_targets_only_qwen38_27b_cq3() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["scope"]["weight_bits_per_weight"] == 2.125
    gate = prereg["sequential_gates"]["gate_2_exact_shape_microbenchmark"]
    assert gate["rows"] == 17_408
    assert gate["columns"] == 5_120
    assert gate["repetitions"] == 9
    assert gate["minimum_control_over_candidate_speedup"] >= 1.08


def test_dual_chain_patch_preserves_integer_dot_and_splits_fp_dependencies() -> None:
    source = PATCH.read_text()
    assert "ggml_cq3_dot_scaled_group_dual_chain_v11" in source
    assert "sumf_group0" in source
    assert "sumf_group1" in source
    assert "_mm256_maddubs_epi16(digits0, values0)" in source
    assert "_mm256_maddubs_epi16(digits3, values3)" in source
    assert "activation->sum" in source
    assert "weight->d[group]" in source
    assert "_mm256_add_ps(sumf_group0, sumf_group1)" in source


def test_v11_stops_before_timing_on_frozen_pair_exactness_gate() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    assert receipt["classification"] == (
        "numerical_contract_failure_not_a_performance_score"
    )
    assert receipt["checks"]["existing_pair_bit_exactness_passed"] is False
    assert receipt["performance"]["ran"] is False
    assert "No zero" in receipt["performance"]["reason"]


def test_v12_replaces_bit_identity_with_bounded_error_before_timing() -> None:
    prereg = json.loads(V12_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["implementation"]["kernel_change_from_v11"] == "none"
    correctness = prereg["sequential_gates"]["gate_1_build_and_correctness"]
    assert correctness["maximum_scaled_error"] <= 2e-5
    performance = prereg["sequential_gates"]["gate_2_exact_shape_microbenchmark"]
    assert performance["minimum_control_over_candidate_speedup"] >= 1.08
    harness = V12_HARNESS_PATCH.read_text()
    assert "max_scaled_delta <= 2e-5f" in harness
    assert "exact_control_candidate ? \"true\" : \"false\"" in harness
    assert "max_abs_delta" in harness


def test_v12_stops_before_timing_on_full_shape_error_bound() -> None:
    receipt = json.loads(V12_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    assert receipt["checks"]["assembly_has_two_independent_accumulator_registers"] is True
    assert receipt["checks"]["full_shape_max_scaled_delta_at_most_2e_5"] is False
    assert receipt["failure"]["population"]["total_outputs"] == 34_816
    assert receipt["performance"]["ran"] is False
    assert "No zero" in receipt["performance"]["reason"]


def test_v13_is_nonpromotable_and_reports_full_error_distribution() -> None:
    prereg = json.loads(V13_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["classification"] == "non_promotable_diagnostic"
    assert prereg["measurement_contract"]["total_outputs"] == 34_816
    assert prereg["measurement_contract"]["timed_repetitions"] == 9
    assert prereg["decision_contract"]["promotable"] is False
    assert prereg["decision_contract"][
        "no_numerical_threshold_will_be_selected_from_this_run"
    ] is True
    source = V13_PATCH.read_text()
    assert "scaled_delta_p50" in source
    assert "scaled_delta_p99" in source
    assert "std::isfinite(max_scaled_delta)" in source


def test_v13_rejects_stable_dual_chain_regression_without_model_evals() -> None:
    receipt = json.loads(V13_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    measurement = receipt["measurement"]
    assert measurement["total_outputs"] == 34_816
    assert measurement["single_chain_over_dual_chain_speedup"] < 1.0
    assert measurement["dual_chain_sample_range_fraction"] < 0.05
    assert measurement["scaled_delta_p99"] < measurement["max_scaled_delta"]
    assert receipt["checks"]["model_integration_required"] is False
    assert receipt["quality"]["ran"] is False


def test_v14_preserves_fp_order_and_targets_integer_decode_overlap() -> None:
    prereg = json.loads(V14_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["scope"]["weight_bits_per_weight"] == 2.125
    correctness = prereg["sequential_gates"]["gate_1_build_and_bit_exactness"]
    assert correctness["full_shape_exact_control_candidate"] is True
    performance = prereg["sequential_gates"]["gate_2_exact_shape_microbenchmark"]
    assert performance["minimum_control_over_candidate_speedup"] >= 1.05
    source = V14_PATCH.read_text()
    first = source.index("accumulator0), scale0, sumf")
    second = source.index("accumulator1), scale1, sumf")
    assert first < second
    assert "ggml_cq3_integer_group_dual_decode_v14" in source
    assert "sumf_group0" not in source
