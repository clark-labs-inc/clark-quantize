from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_calibration_residual_ggml_v40.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def test_v40_is_qwen38_only_and_frozen_before_timing() -> None:
    data = preregistration()
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["qwen_4b_compute_forbidden"] is True
    assert data["status"] == (
        "frozen_after_validation_only_preflight_before_any_timing_execution"
    )
    assert data["disclosed_preflight"]["timing_samples_collected"] == 0
    assert data["disclosed_preflight"]["thresholds_adjusted_from_timing"] is False


def test_v40_inherits_the_admitted_v39_kernel_without_erasing_v38() -> None:
    data = preregistration()
    assert data["parent"]["required_decision"] == (
        "admit_unchanged_exact_residual_kernel_to_llama_cpp_integration_gate"
    )
    assert data["parent"]["required_gate_passed"] is True
    assert data["parent"]["v38_remains_terminally_rejected"] is True
    parent = REPO / data["parent"]["terminal_decision_path"]
    assert sha256(parent) == data["parent"]["terminal_decision_sha256"]


def test_v40_uses_a_clean_pinned_ggml_graph_node() -> None:
    data = preregistration()
    assert data["llama_cpp"]["revision"] == (
        "4df29be4f4c3673f428170fda944a5b19f743bb8"
    )
    assert data["llama_cpp"]["source_must_be_clean"] is True
    assert data["llama_cpp"]["model_loader_change_in_v40"] is False
    source = (RUNTIME / "qwen38_calibration_residual_ggml_v40.cpp").read_text()
    assert "ggml_map_custom2(" in source
    assert "ggml_backend_graph_compute" in source
    assert "cq3_calibration_residual_f32_avx2_trusted(" in source
    assert "float rank_scratch[kRank]" in source
    assert "ggml_mul_mat(" not in source
    assert "float_factors" not in source


def test_v40_gate_has_decode_microbatch_parity_and_stability() -> None:
    gate = preregistration()["gate"]
    assert gate["required_exact_payloads"] == 2
    assert gate["required_token_counts"] == [1, 8]
    assert gate["maximum_absolute_error_vs_direct"] == 0.0
    assert gate["maximum_decode_graph_overhead_ratio_vs_direct"] == 4.0
    assert gate["maximum_projected_all_64_layers_decode_ns"] == 7_272_727
    assert gate["maximum_graph_block_median_range_fraction"] == 0.1
    assert gate["maximum_eight_token_per_token_vs_decode_ratio"] == 0.5
    assert gate["required_payload_bytes_each"] == 174_144
    assert gate["required_rank_scratch_bytes_per_task"] == 64


def test_v40_retains_all_raw_timing_denominators() -> None:
    measurement = preregistration()["measurement"]
    assert measurement["blocks_per_arm_per_case"] == 7
    assert measurement["samples_per_arm_per_block"] == 9
    assert measurement["inner_batch_calls_per_sample"] == 128
    assert "504 batch timing samples" in measurement["raw_samples"]
    runner = (RUNTIME / "run_qwen38_calibration_residual_ggml_v40.py").read_text()
    assert '"raw_ns_per_batch_blocks": blocks' in runner
    assert "statistics.median(block)" in runner
    assert "atomic_json(args.receipt, receipt)" in runner
    assert "atomic_json(args.decision, decision)" in runner
    assert "atomic_text(" in runner


def test_v40_frozen_local_inputs_match_hashes() -> None:
    for descriptor in preregistration()["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_v40_external_executable_and_libraries_are_frozen() -> None:
    external = preregistration()["frozen_external_binaries"]
    assert set(external) == {"harness", "ggml_cpu", "ggml_base"}
    for descriptor in external.values():
        assert descriptor["build_path"].startswith("/workspace/")
        assert len(descriptor["sha256"]) == 64


def test_v40_claim_boundary_forbids_whole_model_promotion() -> None:
    data = preregistration()
    assert "whole-model quality or speed" in data["claim_boundary"]
    rules = "\n".join(data["promotion_rules"])
    assert "adapter-file loading" in rules
    assert "WikiText" in rules
    assert "CUTE" in rules
    assert "MATH-500" in rules
    assert "MMLU" in rules
    assert "cannot establish the 2.75 tok/s target" in rules
