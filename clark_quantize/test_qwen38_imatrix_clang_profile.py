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
