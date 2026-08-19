from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_calibration_residual_cpu_v38.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v38_all_frozen_inputs_are_byte_exact() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["status"] == "frozen_before_authoritative_amd_execution"
    for descriptor in prereg["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_v38_uses_both_exact_positive_v37_payloads() -> None:
    prereg = json.loads(PREREG.read_text())
    parent_path = REPO / prereg["parent"]["terminal_decision_path"]
    parent = json.loads(parent_path.read_text())
    assert sha256(parent_path) == prereg["parent"]["terminal_decision_sha256"]
    assert parent["decision"] == "admit_exact_algebra_to_native_kernel_gate_only"
    assert parent["scientific_gate_passed"] is True
    assert [case["layer"] for case in prereg["cases"]] == [0, 63]
    for case in prereg["cases"]:
        payload = REPO / case["payload_path"]
        assert payload.stat().st_size == case["payload_bytes"] == 174144
        assert sha256(payload) == case["payload_sha256"]


def test_v38_kernel_contract_forbids_high_precision_materialization() -> None:
    prereg = json.loads(PREREG.read_text())
    algebra = prereg["runtime_algebra"]
    assert algebra["rank"] == 16
    assert algebra["rank_scratch_bytes_per_call"] == 64
    assert algebra["high_precision_factor_materialization_forbidden"] is True
    assert algebra["dense_5120_by_5120_residual_materialization_forbidden"] is True
    source = (REPO / prereg["frozen_inputs"]["native_c"]["path"]).read_text()
    assert "malloc(" not in source
    assert "calloc(" not in source
    assert "_mm256_cvtepi8_epi32" in source
    assert "_mm256_cvtph_ps" in source


def test_v38_timing_and_promotion_denominators_are_frozen() -> None:
    prereg = json.loads(PREREG.read_text())
    measurement = prereg["measurement"]
    assert measurement["timing_order"] == [
        "scalar_a",
        "avx2_a",
        "avx2_b",
        "scalar_b",
    ]
    assert measurement["warmups_per_half"] == 5
    assert measurement["repetitions_per_half"] == 16
    assert measurement["inner_iterations_per_sample"] == 128
    assert measurement["samples_per_arm_per_payload"] == 32
    gate = prereg["gate"]
    assert gate["maximum_absolute_error"] == 0.0005
    assert gate["minimum_avx2_speedup_vs_scalar"] == 1.25
    assert gate["maximum_projected_all_64_layers_avx2_ns"] == 7272727
    assert gate["maximum_candidate_sample_range_fraction"] == 0.2


def test_v38_benchmark_does_not_open_capability_evals() -> None:
    prereg = json.loads(PREREG.read_text())
    source = (REPO / prereg["frozen_inputs"]["benchmark"]["path"]).read_text()
    for forbidden in ("CUTE", "MATH-500", "WikiText", "MMLU", "HumanEval"):
        assert forbidden not in source
    assert "load_int8_adapter" in source
    assert "cq3_calibration_residual_f32_avx2_trusted" in source
