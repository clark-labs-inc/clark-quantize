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
