from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
GOAL = RUNTIME / "qwen38_27b_amd_100tps_goal_v1.json"
PROGRAM = RUNTIME / "QWEN38_AMD_100TPS_PROGRAM.md"
AFFINE_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_affine_butterfly_layer0_diagnostic_v1/summary.json"
)


def _payload() -> dict:
    return json.loads(GOAL.read_text())


def test_goal_counts_only_single_stream_target_accepted_tokens() -> None:
    contract = _payload()["throughput_contract"]
    assert contract["minimum_single_stream_accepted_tokens_per_second"] == 100.0
    assert "accepted output tokens" in contract["primary_metric"]
    assert "rejected draft tokens" in contract["excluded_from_numerator"]
    assert contract["aggregate_server_tokens_per_second_is_secondary_and_not_substitutable"]
    assert contract["prefill_tokens_per_second_is_secondary_and_not_substitutable"]
    assert contract["microkernel_or_draft_only_tokens_per_second_is_not_a_model_result"]


def test_goal_is_qwen38_27b_cpu_only_and_allows_its_mtp() -> None:
    scope = _payload()["scope"]
    assert scope["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert scope["directly_associated_mtp_or_draft_components_allowed"] is True
    assert scope["qwen_4b_compute_forbidden"] is True
    assert scope["cpu_deployment_only"] is True
    assert scope["amd_final_validation_required"] is True


def test_goal_records_physical_roofline_and_requires_amortization() -> None:
    payload = _payload()
    roofline = payload["observed_roofline"]
    assert roofline["q4_k_m"]["minimum_weight_read_bandwidth_bytes_per_second_at_100_dense_decode"] == 1_897_387_043_200
    assert roofline["tq2_0"]["minimum_weight_read_bandwidth_bytes_per_second_at_100_dense_decode"] == 1_138_504_742_400
    assert payload["hardware_contract"]["current_ccx33_cannot_validate_100_tokens_per_second"] is True
    assert payload["primary_architecture"]["proposal_goal_mean_accepted_tokens_per_target_call"] >= 4.0


def test_goal_requires_semantic_parity_and_nearly_lossless_quality() -> None:
    payload = _payload()
    assert payload["semantic_correctness"]["greedy"]["require_exact_committed_token_id_parity_with_non_speculative_target"] is True
    quality = payload["nearly_lossless_quality_gate"]
    assert quality["minimum_cute_correct_of_20"] >= 8
    assert quality["minimum_math500_correct_of_8"] >= 4
    assert quality["minimum_code_correct_of_6"] >= 4
    assert quality["minimum_mmlu_correct_of_64"] >= 50
    assert quality["maximum_wikitext_ppl_ratio_vs_q4"] <= 1.05


def test_program_refuses_surrogate_100tps_claims() -> None:
    source = PROGRAM.read_text()
    assert "100 accepted, end-to-end output tokens per second" in source
    assert "Draft tokens, prompt tokens, batched requests, and" in source
    assert "exact token-ID" in source
    assert "parity is the first gate" in source
    assert "Neither a projected roofline nor an MTP acceptance" in source


def test_obsolete_affine_lane_is_closed_with_a_terminal_receipt() -> None:
    receipt = json.loads(AFFINE_RECEIPT.read_text())
    assert receipt["remote_service_status_after_capture"] == "inactive"
    assert receipt["layer_0_terminal"]["local_preregistered_gate_passed"] is True
    assert receipt["layer_0_terminal"]["candidate_complete_block_validation_nmse"] > 0.02
    assert receipt["interpretation"]["quality_or_speed_claim_authorized"] is False
    assert "layers 32 and 63 intentionally not executed" in receipt["interpretation"]["status"]
