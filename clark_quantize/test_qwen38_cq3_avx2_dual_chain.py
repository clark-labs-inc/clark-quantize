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
