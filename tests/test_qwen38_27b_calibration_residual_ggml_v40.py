from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_calibration_residual_ggml_v40.json"
RECEIPTS = RUNTIME / "receipts/qwen38_27b_calibration_residual_ggml_v40"


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


def test_v40_terminal_receipt_passes_every_frozen_gate() -> None:
    receipt = json.loads((RECEIPTS / "receipt.json").read_text())
    assert receipt["complete"] is True
    assert receipt["terminal"] is True
    assert receipt["passed"] is True
    assert len(receipt["cases"]) == 4
    assert all(receipt["gate"]["checks"].values())
    assert sum(
        len(block)
        for case in receipt["cases"]
        for arm in ("direct", "ggml")
        for block in case[arm]["raw_ns_per_batch_blocks"]
    ) == 504
    assert max(
        case["maximum_absolute_error_vs_direct"] for case in receipt["cases"]
    ) == 0.0
    decode = [case for case in receipt["cases"] if case["tokens"] == 1]
    assert max(case["ggml_overhead_ratio_vs_direct"] for case in decode) == (
        1.0181524704377736
    )
    assert max(case["ggml_ns_per_token"] * 64 for case in decode) == 706_165.0
    assert max(
        row["eight_token_per_token_vs_decode_ratio"]
        for row in receipt["microbatch_ratios"]
    ) == 0.2999111452637459


def test_v40_completion_marker_was_written_after_receipt_and_decision() -> None:
    receipt_path = RECEIPTS / "receipt.json"
    decision_path = RECEIPTS / "terminal-decision.json"
    marker = json.loads((RECEIPTS / "complete.json").read_text())
    decision = json.loads(decision_path.read_text())
    assert marker["terminal"] is marker["passed"] is True
    assert marker["receipt_sha256"] == sha256(receipt_path)
    assert marker["decision_sha256"] == sha256(decision_path)
    assert decision["receipt_sha256"] == sha256(receipt_path)
    assert decision["decision"] == (
        "admit_exact_compact_node_to_llama_adapter_loader_integration_only"
    )


def test_v40_execution_receipt_authenticates_the_bounded_run() -> None:
    execution = json.loads((RECEIPTS / "execution.json").read_text())
    assert execution["repository_commit"] == (
        "0949961e7bebb816cd640e3879e12790e41aa23e"
    )
    assert execution["llama_cpp_revision"] == (
        "4df29be4f4c3673f428170fda944a5b19f743bb8"
    )
    assert execution["cpu_affinity"] == [0, 2, 4, 6]
    assert execution["memory_swap_max_bytes"] == 0
    assert execution["restarts"] == 0
    assert execution["result"] == "success"
    assert execution["exit_code"] == 0
    assert execution["retained_raw_timing_samples"] == 504
    assert execution["receipt_sha256"] == sha256(RECEIPTS / "receipt.json")
    assert execution["terminal_decision_sha256"] == sha256(
        RECEIPTS / "terminal-decision.json"
    )
    assert execution["completion_marker_sha256"] == sha256(
        RECEIPTS / "complete.json"
    )
