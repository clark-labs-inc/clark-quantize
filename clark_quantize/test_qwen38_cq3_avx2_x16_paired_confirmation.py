from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_avx2_x16_paired_confirmation_v19.json"
HARNESS = RUNTIME / "kernels/qwen38_27b_cq3_x16_paired_bench_v19.inc"
HARNESS_PATCH = RUNTIME / "kernels/llama_cpp_cq3_x16_paired_harness_v19.patch"
RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_cq3_avx2_x16_paired_confirmation_v19/terminal-decision.json"
)


def test_v19_freezes_candidate_and_changes_only_harness() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    candidate = prereg["frozen_candidate"]
    assert candidate["candidate_patch_sha256"] == (
        "8e8ad180217a761c8f4931dd90036164330a090e95b8f61716e2611fc970c6d5"
    )
    assert candidate["kernel_code_changes_allowed"] is False
    assert candidate["runtime_layout_changes_allowed"] is False
    assert candidate["harness_only_change_allowed"] is True


def test_v19_uses_frozen_ab_ba_pairwise_statistics() -> None:
    prereg = json.loads(PREREG.read_text())
    measurement = prereg["measurement"]
    assert measurement["measured_pairs_per_comparison"] == 21
    assert "Even indices use AB and odd indices BA" in measurement["schedule"]
    assert measurement["raw_time_medians_are_diagnostic_only"] is True
    gate = prereg["sequential_gates"]["gate_1_paired_confirmation"]
    assert gate["minimum_median_x8_over_x16_pairwise_ratio"] == 1.08
    assert gate["minimum_median_ordinary_over_x16_pairwise_ratio"] == 1.4
    assert gate["maximum_x8_over_x16_pairwise_ratio_mad_fraction"] == 0.03
    assert gate["maximum_ordinary_over_x16_pairwise_ratio_mad_fraction"] == 0.03


def test_v19_harness_records_pairs_frequency_and_no_kernel_code() -> None:
    source = HARNESS.read_text()
    harness_patch = HARNESS_PATCH.read_text()
    assert "measured_pairs = 21" in source
    assert "control/candidate" in source
    assert "qwen38_cq3_x19_frequency_mhz" in source
    assert "x8_over_x16_pairwise_ratio_mad_fraction" in source
    assert "ggml_gemv_cq3_0x16_cq8_0" in source
    assert "ggml/src" not in harness_patch
    assert harness_patch.count("diff --git") == 1


def test_v19_receipt_obeys_frozen_decision_if_present() -> None:
    if not RECEIPT.exists():
        return
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["model"] == "Qwen/Qwen3.8-27B"
    measurement = receipt["measurement"]
    assert len(measurement["x8_over_x16_pairwise_ratios"]) == 21
    assert len(measurement["ordinary_over_x16_pairwise_ratios"]) == 21
    checks = receipt["checks"]
    passed = all(
        (
            checks["median_x8_over_x16_pairwise_ratio_1_08_met"],
            checks["median_ordinary_over_x16_pairwise_ratio_1_40_met"],
            checks["x8_over_x16_pairwise_ratio_mad_fraction_0_03_met"],
            checks["ordinary_over_x16_pairwise_ratio_mad_fraction_0_03_met"],
            checks["all_frequency_samples_positive"],
        )
    )
    assert receipt["passed"] is passed
    assert receipt["kernel_promoted"] is passed
