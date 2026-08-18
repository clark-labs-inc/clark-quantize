from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_q4_k_fused_ragged_avx2_v2.json"
PATCH = RUNTIME / "kernels/llama_cpp_q4_k_fused_ragged_avx2_v2.patch"
SUMMARIZER = RUNTIME / "summarize_qwen38_27b_q4_k_fused_ragged_avx2_v2.py"
MTP2_PREREG = RUNTIME / "qwen38_27b_q4_k_fused_ragged_mtp2_system_v1.json"
MTP2_SUMMARIZER = RUNTIME / "summarize_qwen38_27b_q4_k_fused_ragged_mtp2_system_v1.py"
CUTE_RUNNER = RUNTIME / "run_qwen38_27b_q4_k_fused_ragged_cute_v1.sh"
CUTE_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_fused_ragged_cute_v1/terminal-decision.json"
)
PADDED_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_ragged3_gemm4_v1/terminal-decision.json"
)
FOUR_ROW_PREREG = RUNTIME / "qwen38_27b_q4_k_fused_four_row_avx2_v3.json"
FOUR_ROW_PATCH = (
    RUNTIME / "kernels/llama_cpp_q4_k_fused_four_row_avx2_v3.patch"
)
FOUR_ROW_SUMMARIZER = (
    RUNTIME / "summarize_qwen38_27b_q4_k_fused_four_row_avx2_v3.py"
)
FOUR_ROW_RUNNER = RUNTIME / "run_qwen38_27b_q4_k_fused_four_row_avx2_v3.sh"
FOUR_ROW_PROFILE = (
    RUNTIME
    / "receipts/qwen38_27b_mtp3_four_row_profile_v1/terminal-decision.json"
)
FOUR_ROW_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_fused_four_row_avx2_v3/terminal-decision.json"
)
FOUR_ROW_V4_PREREG = (
    RUNTIME / "qwen38_27b_q4_k_fused_four_row_specialized_avx2_v4.json"
)
FOUR_ROW_V4_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_fused_four_row_specialized_avx2_v4/terminal-decision.json"
)
FOUR_ROW_V5_PREREG = (
    RUNTIME / "qwen38_27b_q4_k_fused_four_row_rowmajor_avx2_v5.json"
)
FOUR_ROW_V5_PATCH = (
    RUNTIME / "kernels/llama_cpp_q4_k_fused_four_row_rowmajor_avx2_v5.patch"
)
FOUR_ROW_V5_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_fused_four_row_rowmajor_avx2_v5/terminal-decision.json"
)
Q4_K_GEMM_V6_PREREG = (
    RUNTIME / "qwen38_27b_q4_k_gemm_half_permute_avx2_v6.json"
)
Q4_K_GEMM_V6_PATCH = (
    RUNTIME / "kernels/llama_cpp_q4_k_gemm_half_permute_avx2_v6.patch"
)
Q4_K_GEMM_V6_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_gemm_half_permute_avx2_v6/terminal-decision.json"
)
Q8_TINYBLAS_PROFILE_PREREG = (
    RUNTIME / "qwen38_27b_q8_q8_tinyblas_shape_profile_v1.json"
)
Q8_TINYBLAS_PROFILE_PATCH = (
    RUNTIME / "kernels/llama_cpp_q8_q8_shape_profile_v1.patch"
)
Q8_TINYBLAS_PROFILE_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q8_q8_tinyblas_shape_profile_v1/terminal-decision.json"
)
Q8_N4_ORIENTATION_PREREG = (
    RUNTIME / "qwen38_27b_q8_q8_n4_orientation_avx2_v1.json"
)
Q8_N4_ORIENTATION_PATCH = (
    RUNTIME / "kernels/llama_cpp_q8_q8_n4_orientation_avx2_v1.patch"
)
Q8_N4_EXACT_HARNESS = (
    RUNTIME / "verify_qwen38_27b_q8_q8_n4_exact_v1.cpp"
)
Q8_N4_ORIENTATION_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q8_q8_n4_orientation_avx2_v1/terminal-decision.json"
)
Q8_FAMILY_Q4_PREREG = (
    RUNTIME / "qwen38_27b_q8_family_q4_k_transcode_v1.json"
)
Q8_FAMILY_Q4_VERIFIER = (
    RUNTIME / "verify_qwen38_27b_selective_gguf_v1.py"
)
Q8_FAMILY_Q4_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q8_family_q4_k_transcode_v1/terminal-decision.json"
)
Q8_FAMILY_Q40_PREREG = (
    RUNTIME / "qwen38_27b_q8_family_q4_0_fastpath_v2.json"
)
Q8_FAMILY_Q40_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q8_family_q4_0_fastpath_v2/terminal-decision.json"
)
Q4_K_ROWMAJOR_QUALITY_V7_PREREG = (
    RUNTIME / "qwen38_27b_q4_k_rowmajor_quality_gated_v7.json"
)


def test_fused_ragged_scope_and_whole_model_gate() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["scope"]["llama_cpp_revision"] == (
        "4df29be4f4c3673f428170fda944a5b19f743bb8"
    )
    correctness = payload["promotion_gate"]["correctness"]
    performance = payload["promotion_gate"]["performance"]
    assert correctness["prompts"] == 4
    assert correctness["tokens_per_prompt"] == 96
    assert performance["minimum_whole_model_mtp3_accepted_tps_speedup_vs_control"] >= 1.05
    assert performance["maximum_target_only_tps_regression_fraction"] <= 0.02


def test_q8_shape_profiler_is_bounded_thread_deduplicated_and_non_scoreable() -> None:
    prereg = json.loads(Q8_TINYBLAS_PROFILE_PREREG.read_text())
    source = Q8_TINYBLAS_PROFILE_PATCH.read_text()
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["method"]["maximum_distinct_shapes"] == 128
    assert "#ifdef CLARK_Q8_SHAPE_PROFILE" in source
    assert "CLARK_Q8_PROFILE_MAX_SHAPES = 128" in source
    assert "if (params->ith == 0)" in source
    assert "clark_record_q8_shape(m, n, k, lda, ldb, ldc, params->nth)" in source
    assert "CLARK_Q8_SHAPE" in source
    assert "CLARK_Q8_SUMMARY" in source
    assert "++clark_q8_profile.overflow" in source
    assert prereg["correctness_boundary"].startswith(
        "The profile build is not a candidate runtime"
    )


def test_q8_shape_profile_selects_material_n4_family_without_scoring_timing() -> None:
    receipt = json.loads(Q8_TINYBLAS_PROFILE_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["counter_receipt"]["overflow"] == 0
    assert receipt["counter_receipt"]["logical_dispatches"] == 8352
    n4 = next(family for family in receipt["shape_families"] if family["n"] == 4)
    assert n4["calls"] == 7488
    assert n4["call_fraction"] >= 0.70
    assert n4["tile_fraction"] >= 0.90
    assert receipt["amdahl"]["n4_zero_cost_ceiling_from_tile_fraction"] > 1.10
    assert any("not_scoreable" in key for key in receipt["workload"])
    assert receipt["decision"] == "select_q8_q8_n4_orientation_kernel"


def test_q8_n4_orientation_candidate_is_material_and_exactness_gated() -> None:
    prereg = json.loads(Q8_N4_ORIENTATION_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["profile_anchor"]["n4_call_fraction"] >= 0.70
    assert prereg["profile_anchor"]["n4_zero_cost_whole_request_ceiling"] > 1.10
    required = prereg["candidate"]["required_scope"]
    assert any("TA=block_q8_0" in item for item in required)
    assert any("exact remaining n == 4" in item for item in required)
    gates = prereg["sequential_gates"]
    assert gates["gate_2_real_model_four_token_graph"][
        "minimum_candidate_over_control_speedup"
    ] >= 1.08
    assert gates["gate_3_terminal_generation"][
        "minimum_candidate_over_control_mtp3_accepted_tps"
    ] >= 1.08
    assert gates["gate_4_quality"]["math_or_mmlu_style_gate_required_before_promotion"]


def test_q8_n4_orientation_patch_is_type_shape_and_isa_scoped() -> None:
    source = Q8_N4_ORIENTATION_PATCH.read_text()
    assert "std::is_same_v<TA, block_q8_0>" in source
    assert "std::is_same_v<TB, block_q8_0>" in source
    assert "if (n - n0 == 4)" in source
    assert "mc = 2" in source
    assert "nc = 4" in source
    assert "gemmMx4<2>(m0, m, n0, n)" in source
    assert "defined(__AVX2__) && defined(__F16C__)" in source
    assert source.count("gemmMx4<2>(m0, m, n0, n)") == 1


def test_q8_n4_exact_harness_covers_all_profiled_weight_shapes() -> None:
    source = Q8_N4_EXACT_HARNESS.read_text()
    for m, k in (
        (10240, 160),
        (48, 160),
        (6144, 160),
        (5120, 192),
        (12288, 160),
        (1024, 160),
    ):
        assert f"{{{m}, {k}}}" in source
    assert "constexpr int64_t n = 4" in source
    assert "constexpr int nth = 4" in source
    assert "GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32" in source
    assert "write_all(output, result.data()" in source


def test_q8_n4_orientation_is_terminal_exact_negative() -> None:
    receipt = json.loads(Q8_N4_ORIENTATION_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    assert receipt["exactness"]["byte_identical"] is True
    assert receipt["real_model_four_token_graph"]["control_over_candidate"] < 1.08
    assert receipt["checks"]["graph_speedup_at_least_1_08"] is False
    assert receipt["terminal_generation"]["ran"] is False
    assert receipt["cute"]["ran"] is False
    assert receipt["q8_tensor_inventory"]["tensor_count"] == 288
    assert receipt["q8_tensor_inventory"]["weight_count"] == 6734479360


def test_q8_family_q4_screen_is_material_disk_bounded_and_quality_gated() -> None:
    prereg = json.loads(Q8_FAMILY_Q4_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    artifact = prereg["artifact_contract"]
    assert artifact["selected_tensor_count"] == 288
    assert artifact["selected_weight_count"] == 6734479360
    assert artifact["minimum_artifact_bytes_saved"] >= 3_300_000_000
    assert artifact["maximum_retained_candidates"] == 1
    gates = prereg["sequential_gates"]
    assert gates["gate_3_real_model_four_token_graph"][
        "minimum_candidate_over_control_speedup"
    ] >= 1.08
    quality = gates["gate_5_quality"]
    assert quality["cute_quick_denominator"] == 20
    assert quality["maximum_wikitext_ppl_ratio"] <= 1.05
    assert quality["mmlu_denominator"] == 64
    assert quality["paired_frozen_math500_subset_required"] is True
    assert "QAT" in " ".join(prereg["recovery_if_quality_fails"])


def test_selective_gguf_verifier_hashes_nonselected_payloads_and_exact_scope() -> None:
    source = Q8_FAMILY_Q4_VERIFIER.read_text()
    assert 'parser.add_argument("--expected-target-type", default="Q4_K")' in source
    assert 'before_type != "Q8_0" or after_type != args.expected_target_type' in source
    assert 'candidate_types.get(args.expected_target_type, 0)' in source
    assert 'candidate_types.get("Q8_0", 0) == 0' in source
    assert "control_nonselected_sha256 == candidate_nonselected_sha256" in source
    assert "selected_weight_count == 6_734_479_360" in source
    assert "actual_bytes_saved == args.expected_bytes_saved" in source
    assert "selected_payload_saved == args.expected_bytes_saved" in source
    assert "all(checks.values())" in source


def test_q8_family_q4_k_is_terminal_mixed_negative() -> None:
    receipt = json.loads(Q8_FAMILY_Q4_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    assert receipt["artifact"]["nonselected_payloads_byte_identical"] is True
    assert receipt["free_generation_canary"]["total_tokens"] == 384
    assert receipt["real_model_four_token_graph"]["control_over_candidate"] >= 1.08
    assert receipt["paired_serving"]["candidate_over_control_target"] > 1.18
    assert receipt["paired_serving"]["candidate_over_control_mtp3"] < 1.08
    assert receipt["quality"]["cute_quick"]["candidate_correct"] == 8
    assert receipt["quality"]["cute_quick"]["candidate_denominator"] == 20
    assert receipt["quality"]["wikitext_quick"]["candidate_over_control"] <= 1.05
    assert receipt["quality"]["mmlu"]["ran"] is False


def test_q4_0_fastpath_must_beat_control_and_prior_q4_k_candidate() -> None:
    prereg = json.loads(Q8_FAMILY_Q40_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["artifact_contract"]["selected_tensor_count"] == 288
    assert prereg["artifact_contract"]["maximum_retained_candidates"] == 1
    graph = prereg["sequential_gates"]["gate_3_real_model_four_token_graph"]
    assert graph["minimum_candidate_over_control_speedup"] >= 1.10
    assert graph["minimum_candidate_over_prior_q4_k_speedup"] >= 1.03
    serving = prereg["sequential_gates"]["gate_4_target_and_mtp3"]
    assert serving["minimum_candidate_over_control_target_tps"] >= 1.20
    assert serving["minimum_candidate_over_control_mtp3_tps"] >= 1.08
    assert serving["minimum_candidate_over_prior_q4_k_mtp3_tps"] >= 1.03
    quality = prereg["sequential_gates"]["gate_5_quality"]
    assert quality["cute_quick_denominator"] == 20
    assert quality["maximum_wikitext_ppl_ratio"] <= 1.05


def test_rowmajor_v7_replaces_token_identity_with_frozen_quality_gates() -> None:
    prereg = json.loads(Q4_K_ROWMAJOR_QUALITY_V7_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["artifact_contract"]["selected_tensor_count"] == 288
    assert prereg["artifact_contract"]["maximum_retained_candidates"] == 1
    graph = prereg["sequential_gates"]["gate_2_same_weights_graph"]
    assert graph["minimum_c_over_b_speedup"] >= 1.05
    generation = prereg["sequential_gates"]["gate_3_free_generation"]
    assert generation["exact_cross_runtime_token_parity_required"] is False
    assert generation["trajectory_differences_must_be_recorded"] is True
    serving = prereg["sequential_gates"]["gate_4_paired_serving"]
    assert serving["minimum_c_over_a_mtp3_tps"] >= 1.08
    assert serving["minimum_c_over_b_mtp3_tps"] >= 1.03
    quality = prereg["sequential_gates"]["gate_5_quick_quality"]
    assert quality["cute_quick_denominator"] == 20
    assert quality["minimum_candidate_correct"] >= 9
    assert quality["maximum_candidate_over_control_ppl"] <= 1.05
    capability = prereg["sequential_gates"]["gate_6_capability"]
    assert capability["mmlu_denominator"] == 64
    assert capability["paired_frozen_math500_subset_required"] is True


def test_q4_0_fastpath_is_terminally_rejected_with_quick_eval() -> None:
    receipt = json.loads(Q8_FAMILY_Q40_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is False
    assert receipt["artifact"]["nonselected_payloads_byte_identical"] is True
    assert receipt["free_generation_canary"]["total_tokens"] == 384
    graph = receipt["real_model_four_token_graph"]
    assert graph["control_over_candidate"] < 1.10
    assert graph["prior_q4_k_over_candidate"] < 1.03
    cute = receipt["quick_quality_diagnostic"]["cute_quick"]
    assert cute["candidate_correct"] == 8
    assert cute["candidate_denominator"] == 20


def test_fused_patch_shares_q4_decode_and_preserves_row_accumulators() -> None:
    source = PATCH.read_text()
    assert "Decode each Q4_K weight block once" in source
    assert "__m256 acc_rows[3]" in source
    assert "__m256i iacc_b[3]" in source
    assert "for (int64_t y = 0; y < nr; ++y)" in source
    q4_decode_end = source.index("mins_01")
    row_loop = source.index("for (int64_t y = 0; y < nr; ++y)", q4_decode_end)
    assert q4_decode_end < row_loop
    assert "iacc_b[y]" in source
    assert "s + y * bs + x * 8" in source


def test_summarizer_uses_real_model_samples_and_exact_parity() -> None:
    source = SUMMARIZER.read_text()
    assert 'set(samples) != {1, 2, 3}' in source
    assert 'graph["1"]["speedup"] >= 1 / 1.02' in source
    assert 'graph["2"]["speedup"] >= 1.08' in source
    assert 'graph["3"]["speedup"] >= 1.10' in source
    assert "mtp3_ratio >= 1.05" in source
    assert "token_rows(args.control_mtp, \"target\")" in source
    assert "token_rows(args.control_mtp, \"mtp3\")" in source
    assert '"passed": all(checks.values())' in source


def test_mtp2_gate_matches_three_row_kernel_and_separates_serving_winner() -> None:
    payload = json.loads(MTP2_PREREG.read_text())
    assert payload["shape_correction"]["mtp_draft_width"] == 2
    assert payload["shape_correction"]["target_verification_rows"] == 3
    assert payload["promotion_gate"]["paired_mtp2"][
        "minimum_candidate_over_control_accepted_tps"
    ] >= 1.05
    source = MTP2_SUMMARIZER.read_text()
    assert "paired_mtp2_ratio >= 1.05" in source
    assert '"kernel_promoted": all(kernel_checks.values())' in source
    assert '"deployment_winner": deployment_winner and all(kernel_checks.values())' in source


def test_padded_gemm4_receipt_excludes_unexercised_synthetic_timing() -> None:
    payload = json.loads(PADDED_RECEIPT.read_text())
    assert payload["terminal"] is True
    assert payload["decision"] == "reject_padded_gemm4"
    assert payload["invalid_diagnostic"]["classification"] == (
        "not a score for this candidate"
    )
    assert payload["gates"]["candidate_dispatch_proven"] is True
    assert payload["gates"]["promotion_passed"] is False


def test_cute_runner_is_candidate_pinned_restartable_and_nonoverlapping() -> None:
    source = CUTE_RUNNER.read_text()
    assert "qwen38-27b-q4-k-fused-ragged-avx2-v2" in source
    assert "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34" in source
    assert 'if [[ -s "$state/CUTE_COMPLETE" && -s "$state/cute.json" ]]' in source
    assert "pgrep -x llama-server" in source
    assert 'http://127.0.0.1:${port}/health' in source
    assert "eval_llama_cpp_cute_quick.py" in source


def test_cute_receipt_has_exact_denominator_and_excludes_invalid_attempt() -> None:
    payload = json.loads(CUTE_RECEIPT.read_text())
    assert payload["benchmark"]["candidate_correct"] == 9
    assert payload["benchmark"]["candidate_denominator"] == 20
    assert payload["benchmark"]["control_correct"] == 9
    assert payload["benchmark"]["control_denominator"] == 20
    assert payload["prior_attempt"]["classification"] == "infrastructure-invalid"
    assert payload["prior_attempt"]["scored"] is False
    assert payload["gates"]["item_level_output_parity"] is True
    assert payload["gates"]["passed"] is True


def test_four_row_candidate_is_profile_selected_and_preregistered() -> None:
    payload = json.loads(FOUR_ROW_PREREG.read_text())
    profile = json.loads(FOUR_ROW_PROFILE.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["profile_anchor"]["four_row_q4_k_cycle_fraction"] == 0.4498
    assert payload["promotion_gate"]["performance"][
        "minimum_real_model_four_row_graph_speedup"
    ] >= 1.10
    assert payload["promotion_gate"]["performance"][
        "minimum_whole_model_mtp3_accepted_tps_speedup"
    ] >= 1.05
    assert profile["terminal"] is True
    assert profile["decision"] == "prioritize_four_row_q4_k_target_kernel"
    assert profile["exclusive_cycle_samples"]["ggml_gemm_q4_K_8x8_q8_K"] > 0.40
    assert profile["exclusive_cycle_samples"]["draft_q4_0_gemv"] < 0.10


def test_four_row_patch_is_exact_shape_scoped() -> None:
    source = FOUR_ROW_PATCH.read_text()
    assert "assert(nr > 0 && nr <= 4)" in source
    assert "__m256 acc_rows[4]" in source
    assert "__m256i iacc_b[4]" in source
    assert "const bool fused_four_rows = q4_k_8x8 && nrows == 4" in source
    assert "nrows > 3 && !fused_four_rows" in source
    assert "fused_four_rows ? 0" in source


def test_four_row_summarizer_enforces_terminal_gates() -> None:
    source = FOUR_ROW_SUMMARIZER.read_text()
    assert "expected 14 paired graph samples" in source
    assert "graph_speedup >= 1.10" in source
    assert "mtp3_ratio >= 1.05" in source
    assert "target_ratio >= 0.98" in source
    assert "cross_runtime_mtp3_exact_token_parity" in source
    assert '"draft_behavior_unchanged"' in source
    assert '"cute_required": all(checks.values())' in source


def test_four_row_runner_is_restartable_pinned_and_exact_denominator() -> None:
    source = FOUR_ROW_RUNNER.read_text()
    assert "Qwen3.8-27B-Q4_K_M.gguf" in source
    assert "mtp-Qwen3.8-27B-Q4_0.gguf" in source
    assert "4df29be4f4c3673f428170fda944a5b19f743bb8" in source
    assert "-p 4 -n 0 -r 7" in source
    assert "--widths 0,3 --n-predict 96" in source
    assert 'if [[ -s "$state_dir/terminal-decision.json"' in source
    assert "pgrep -x llama-server" in source
    assert "summarize_qwen38_27b_q4_k_fused_four_row_avx2_v3.py" in source


def test_four_row_v3_is_terminal_numerical_rejection() -> None:
    payload = json.loads(FOUR_ROW_RECEIPT.read_text())
    assert payload["terminal"] is True
    assert payload["infrastructure_valid"] is True
    assert payload["passed"] is False
    assert payload["checks"]["four_row_graph_speedup_at_least_1_10"] is True
    assert payload["checks"]["cross_runtime_target_exact_token_parity"] is False
    assert payload["checks"]["whole_model_mtp3_speedup_at_least_1_05"] is False
    assert payload["whole_model"]["candidate_draft_n"] == 1116
    assert payload["whole_model"]["candidate_draft_n_accepted"] == 0
    assert payload["token_diagnostics"]["candidate_target_unique_token_ids"] == [0]
    assert payload["token_diagnostics"]["candidate_target_zero_token_fraction"] == 1.0
    assert payload["provenance"]["four_row_patch_sha256"] == payload["provenance"][
        "executed_four_row_patch_sha256"
    ]
    assert payload["provenance"]["repository_normalized_four_row_patch_sha256"] == (
        "41fac22043304854d4b21562d3b5fb6c5bbbc880b339a4af9c7aae8bb849c5ab"
    )


def test_four_row_v4_prereg_requires_separate_specialization_and_canaries() -> None:
    payload = json.loads(FOUR_ROW_V4_PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["negative_anchor"]["classification"] == (
        "numerical-contract failure"
    )
    structure = payload["implementation"]["required_structure"]
    assert any("MAX_ROWS=3" in item for item in structure)
    assert any("MAX_ROWS=4" in item for item in structure)
    gates = payload["sequential_gates"]
    assert gates["gate_1_ordinary_decode_canary"][
        "control_vs_candidate_exact_tokens"
    ] is True
    assert gates["gate_2_backend_and_graph"][
        "minimum_real_model_four_row_graph_speedup"
    ] >= 1.10
    assert gates["gate_3_mtp3_canary"][
        "candidate_draft_acceptance_must_equal_control"
    ] is True


def test_four_row_v4_is_cancelled_on_proven_layout_mismatch() -> None:
    payload = json.loads(FOUR_ROW_V4_RECEIPT.read_text())
    assert payload["terminal"] is True
    assert payload["build_attempted"] is False
    assert payload["benchmark_attempted"] is False
    assert payload["passed"] is False
    assert payload["evidence"]["produced_type"]["type"] == "block_q8_Kx4"
    assert payload["evidence"]["incorrect_consumer"]["cast"] == (
        "const block_q8_K *"
    )
    assert payload["classification"].endswith("not a performance score")


def test_four_row_v5_prereg_is_single_change_and_sequentially_gated() -> None:
    payload = json.loads(FOUR_ROW_V5_PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert "ne11 == 4" in payload["implementation"]["single_independent_change"]
    assert "v3 exact-four-row GEMV arithmetic" in payload["implementation"][
        "unchanged"
    ]
    gates = payload["sequential_gates"]
    assert gates["gate_2_ordinary_decode_canary"][
        "control_vs_candidate_exact_tokens"
    ] is True
    assert gates["gate_3_real_model_four_row_graph"][
        "minimum_candidate_over_control_speedup"
    ] >= 1.10
    assert gates["gate_5_terminal_mtp3"][
        "minimum_candidate_over_control_accepted_tps"
    ] >= 1.05


def test_four_row_v5_patch_changes_only_exact_four_row_q8_packing() -> None:
    source = FOUR_ROW_V5_PATCH.read_text()
    assert "const bool row_major_four_rows = q4_k_8x8 && ne11 == 4" in source
    assert "row_major_four_rows ? 0 : ne11 - ne11 % 4" in source
    assert "i11_processed + ith" in source
    assert "from_float" in source
    assert "ggml_quantize_mat_t" in source
    assert "arch/x86/repack.cpp" not in source


def test_four_row_v5_is_rejected_despite_mtp3_speed_diagnostic() -> None:
    payload = json.loads(FOUR_ROW_V5_RECEIPT.read_text())
    assert payload["terminal"] is True
    assert payload["infrastructure_valid"] is True
    assert payload["passed"] is False
    assert payload["whole_model"]["candidate_over_control_mtp3"] > 1.12
    assert payload["checks"]["terminal_target_4x96_exact_token_parity"] is False
    assert payload["checks"]["terminal_mtp3_4x96_exact_token_parity"] is False
    assert payload["checks"][
        "predeclared_14_sample_graph_speedup_at_least_1_10"
    ] is False
    assert payload["cute"]["ran"] is False


def test_q4_k_gemm_v6_prereg_preserves_algebra_and_gates_stability() -> None:
    payload = json.loads(Q4_K_GEMM_V6_PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert "floating-point accumulation order" in payload["implementation"][
        "forbidden_changes"
    ]
    gates = payload["sequential_gates"]
    assert gates["gate_1_static_and_backend"][
        "byte_permutation_equivalence_exhaustive"
    ] is True
    assert gates["gate_2_real_model_four_row_graph"][
        "include_all_28_measured_samples"
    ] is True
    assert gates["gate_2_real_model_four_row_graph"][
        "maximum_candidate_process_median_range_fraction"
    ] <= 0.05
    assert gates["gate_3_terminal_generation"][
        "target_and_mtp3_exact_control_parity"
    ] is True


def test_q4_k_gemm_v6_half_permute_is_exhaustively_lane_equivalent() -> None:
    source = Q4_K_GEMM_V6_PATCH.read_text()
    assert source.count("_mm256_permute2x128_si256") == 8
    assert source.count("0x20") == 4
    assert source.count("0x31") == 4
    assert source.count("-_mm256_blend_epi32") == 0
    for active_source_lane in range(16):
        a = [int(index == active_source_lane) for index in range(8)]
        b = [int(index + 8 == active_source_lane) for index in range(8)]
        swap_a = a[4:] + a[:4]
        swap_b = b[4:] + b[:4]
        old_0145 = a[:4] + swap_b[4:]
        old_2367 = swap_a[:4] + b[4:]
        new_0145 = a[:4] + b[:4]
        new_2367 = a[4:] + b[4:]
        assert old_0145 == new_0145
        assert old_2367 == new_2367


def test_q4_k_gemm_v6_is_stable_but_below_speed_gate() -> None:
    payload = json.loads(Q4_K_GEMM_V6_RECEIPT.read_text())
    assert payload["terminal"] is True
    assert payload["infrastructure_valid"] is True
    assert payload["passed"] is False
    assert payload["checks"]["byte_permutation_equivalence_exhaustive"] is True
    assert payload["checks"]["graph_speedup_at_least_1_03"] is False
    assert payload["checks"][
        "candidate_process_median_range_at_most_0_05"
    ] is True
    graph = payload["real_model_four_row_graph"]
    assert graph["control_sample_count"] == 28
    assert graph["candidate_sample_count"] == 28
    assert 1.02 < graph["control_over_candidate"] < 1.03
    assert payload["terminal_generation"]["ran"] is False
    assert payload["cute"]["ran"] is False


def test_q8_tinyblas_profile_is_large_share_and_shape_gated() -> None:
    payload = json.loads(Q8_TINYBLAS_PROFILE_PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["profile_anchor"]["mtp3_exclusive_cycle_fraction"] > 0.23
    assert "gemm4xN<2>" in payload["profile_anchor"]["symbol"]
    assert "m, n, k, lda, ldb, ldc" in payload["method"]["required_fields"]
    assert payload["method"]["maximum_distinct_shapes"] == 128
    assert "70%" in payload["selection_rule"]
