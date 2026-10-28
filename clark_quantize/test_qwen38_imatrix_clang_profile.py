from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
V32_PREREG = RUNTIME / "qwen38_27b_q4_k_clang21_full_graph_profile_v32.json"
V32_AMENDMENT = (
    RUNTIME / "qwen38_27b_q4_k_clang21_full_graph_profile_v32_analysis_amendment.json"
)
V32_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_q4_k_clang21_full_graph_profile_v32/terminal-decision.json"
)
V33_PREREG = RUNTIME / "qwen38_27b_imatrix_q4_k_clang21_graph_profile_v33.json"
V33_RUNNER = RUNTIME / "run_qwen38_27b_imatrix_q4_k_clang21_graph_profile_v33.sh"
V33_SUMMARIZER = (
    RUNTIME / "summarize_qwen38_27b_imatrix_q4_k_clang21_graph_profile_v33.py"
)
V33_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_imatrix_q4_k_clang21_graph_profile_v33/terminal-decision.json"
)
V34_PREREG = RUNTIME / "qwen38_27b_clang21_libomp_longrun_v34.json"
V34_RUNNER = RUNTIME / "run_qwen38_27b_clang21_libomp_longrun_v34.sh"
V34_SUMMARIZER = RUNTIME / "summarize_qwen38_27b_clang21_libomp_longrun_v34.py"
V34_RECEIPT = (
    RUNTIME
    / "receipts/qwen38_27b_clang21_libomp_longrun_v34/terminal-decision.json"
)
V35_PREREG = RUNTIME / "qwen38_27b_clang21_serving_transfer_v35.json"
V35_RUNNER = RUNTIME / "run_qwen38_27b_clang21_serving_transfer_v35.sh"
V35_SUMMARIZER = RUNTIME / "summarize_qwen38_27b_clang21_serving_transfer_v35.py"


def test_v32_stock_profile_is_terminal_and_does_not_select_from_stock_q8() -> None:
    prereg = json.loads(V32_PREREG.read_text())
    amendment = json.loads(V32_AMENDMENT.read_text())
    receipt = json.loads(V32_RECEIPT.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["profiles"]["candidate_clang21_v30"]["perf_samples"] >= 100_000
    assert receipt["profiles"]["candidate_clang21_v30"]["lost_samples"] == 0
    assert receipt["paired_bench_parent_over_candidate"] < 1.05
    assert receipt["amdahl"]["required_q8_speedup_given_confirmed_q4_for_1_40x"] > 3.8
    assert "288 Q8_0" in receipt["next_lane"]
    assert "infrastructure-invalid" in amendment["reason"]
    assert "No new profile or benchmark samples." in amendment["forbidden_changes"]
    assert receipt["invalid_attempt"]["classification"].startswith(
        "infrastructure-invalid"
    )


def test_v33_profiles_the_quality_anchor_and_freezes_identity() -> None:
    prereg = json.loads(V33_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["scope"]["artifact_sha256"] == (
        "6176472b7d039a6d911b34eea78dd4507256b4a22b3003d40b8587b711014f2a"
    )
    assert prereg["quality_anchor"]["cute"] == "9/20 versus stock 9/20"
    assert prereg["quality_anchor"]["wikitext_ratio_vs_stock"] < 1.0
    assert prereg["quality_anchor"]["math500"] == "5/8"
    assert prereg["analysis"]["required_speedup_from_v23_anchor"] > 1.20
    assert "v32 stock-model shares" in prereg["analysis"]["forbidden_inference"]


def test_v33_runner_is_bounded_pinned_and_deletes_raw_profiles() -> None:
    source = V33_RUNNER.read_text()
    assert "Qwen3.8-27B-Q4_K_M-imatrix-Q4_K-v10.gguf" in source
    assert "sha256sum \"$model\"" in source
    assert "-p 4 -n 0 -r 32" in source
    assert "taskset -c 0,2,4,6" in source
    assert "pgrep -f '/llama-(server|bench|perplexity)( |$)'" in source
    assert "rm -f \"$data\"" in source
    assert "sealed-inputs.sha256" in source


def test_v33_reducer_uses_exclusive_cycles_and_materiality_gate() -> None:
    source = V33_SUMMARIZER.read_text()
    assert "--no-children" in V33_RUNNER.read_text()
    assert '"q4_k_verifier"' in source
    assert '"q8_x_q8_tinyblas"' in source
    assert "share >= 0.10" in source
    assert "single_group_zero_cost_ceilings" in source
    assert "required_speedup_from_v23_anchor" in source
    assert "artifact_and_binary_identities_exact" in source


def test_v33_receipt_retains_profile_but_blocks_speed_claim_on_phase_change() -> None:
    receipt = json.loads(V33_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["profiles"]["candidate_clang21_v30"][
        "exclusive_cycle_fractions"
    ]["q8_x_q8_tinyblas"] == 0.0
    assert receipt["profiles"]["candidate_clang21_v30"][
        "exclusive_cycle_fractions"
    ]["q4_k_verifier"] > 0.66
    phase = receipt["post_profile_phase_change_diagnostic"]
    assert phase["candidate_post_over_pre"] > 1.16
    assert receipt["aggregate_diagnostic"]["speed_claim_authorized"] is False
    assert "isolate Clang code generation" in receipt["next_lane"]


def test_v34_preregisters_long_process_runtime_isolation() -> None:
    prereg = json.loads(V34_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["protocol"]["samples_per_process"] == 32
    assert prereg["protocol"]["run_order"] == [
        "parent_a",
        "clang_default_a",
        "clang_active_a",
        "clang_active_b",
        "clang_default_b",
        "parent_b",
    ]
    assert prereg["protocol"]["clang_active_environment"] == {
        "KMP_BLOCKTIME": "infinite",
        "OMP_WAIT_POLICY": "active",
        "KMP_LIBRARY": "turnaround",
    }
    assert prereg["analysis"]["stable_process_maximum_changepoint_ratio"] <= 1.05
    assert prereg["analysis"]["minimum_parent_over_candidate_arm_median"] >= 1.04


def test_v34_runner_and_reducer_preserve_every_longrun_sample() -> None:
    runner = V34_RUNNER.read_text()
    reducer = V34_SUMMARIZER.read_text()
    assert "-p 4 -n 0 -r 32" in runner
    assert runner.count("run_one ") == 6
    assert "KMP_BLOCKTIME=infinite" in runner
    assert "OMP_WAIT_POLICY=active" in runner
    assert "pgrep -x perf" in runner
    assert "for split in range(8, 25)" in reducer
    assert '"samples": samples' in reducer
    assert "all_32_samples_count" in V34_PREREG.read_text()
    assert "select_clang_active_policy_for_serving_validation" in reducer


def test_v34_receipt_selects_stable_active_libomp_for_serving_only() -> None:
    receipt = json.loads(V34_RECEIPT.read_text())
    assert receipt["terminal"] is True
    assert receipt["infrastructure_valid"] is True
    assert receipt["passed"] is True
    assert receipt["causal_diagnostic"]["default_phase_reproduced_without_perf"] is False
    assert all(
        arm["maximum_changepoint_ratio"] < 1.02
        for arm in receipt["arms"].values()
    )
    assert receipt["arms"]["clang_active_v30"]["parent_over_arm_median"] >= 1.06
    assert receipt["decision"] == "select_clang_active_policy_for_serving_validation"
    assert receipt["target_analysis"]["serving_gate_already_established"] is False
    assert "Run an AB/BA llama-server validation" in receipt["next_lane"]


def test_v35_preregisters_same_model_abba_serving_and_capability_gates() -> None:
    prereg = json.loads(V35_PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["protocol"]["run_order"] == [
        "parent_a",
        "candidate_a",
        "candidate_b",
        "parent_b",
    ]
    assert prereg["protocol"]["configurations_per_run"] == ["target", "mtp3"]
    assert prereg["protocol"]["prompts_per_configuration"] == 4
    assert prereg["protocol"]["generated_tokens_per_prompt"] == 96
    assert prereg["protocol"]["candidate_environment"]["KMP_BLOCKTIME"] == "infinite"
    assert prereg["gates"]["serving_transfer"][
        "minimum_candidate_over_parent_conventional_tps"
    ] >= 1.04
    assert prereg["gates"]["always_run_quick_quality"] == {
        "cute_quick_minimum_correct": 9,
        "cute_quick_denominator": 20,
        "math500_minimum_correct": 5,
        "math500_denominator": 8,
        "code_trace_minimum_correct": 5,
        "code_trace_denominator": 6,
        "maximum_wikitext_ppl_ratio_vs_stock": 1.01,
        "wikitext_chunks": 4,
    }
    assert prereg["gates"]["conditional_mmlu"]["denominator"] == 64


def test_v35_runner_is_restartable_bounded_and_hash_pinned() -> None:
    source = V35_RUNNER.read_text()
    assert "EXPECTED_PREREG_SHA256" in source
    assert "EXPECTED_SUMMARIZER_SHA256" in source
    assert "EXPECTED_RUNNER_SHA256" in source
    assert "run_mtp parent_a parent" in source
    assert "run_mtp candidate_a candidate" in source
    assert "run_mtp candidate_b candidate" in source
    assert "run_mtp parent_b parent" in source
    assert "KMP_BLOCKTIME=infinite" in source
    assert "--widths 0,3 --n-predict 96" in source
    assert "eval_llama_cpp_cute_quick.py" in source
    assert "eval_llama_cpp_math_code_canary.py" in source
    assert "eval_llama_cpp_wikitext_quick.py" in source
    assert "conditional_mmlu_start" in source
    assert "find \"$state_dir\" \"$run_dir\"" in source
    assert "no model" not in source.lower()


def test_v35_reducer_keeps_program_targets_diagnostic() -> None:
    source = V35_SUMMARIZER.read_text()
    assert '"candidate_over_parent_conventional"' in source
    assert '"candidate_over_parent_mtp3"' in source
    assert '"candidate_at_least_100_tps"' in source
    assert '"cross_runtime_target_token_parity_diagnostic"' in source
    assert '"candidate_target_mtp3_exact_token_parity_every_prompt"' in source
    assert '"advance_to_mmlu_64"' in source
    assert '"retain_clang_active_as_representation_kernel_component"' in source
