from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_q3_k_m_baseline_v1.sh"
)
Q3_K_L_PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_q3_k_l_baseline_v1.json"
)
Q3_K_M_TERMINAL = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_q3_k_m_baseline_v1/terminal-decision.json"
)
Q3_K_L_TERMINAL = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_q3_k_l_baseline_v1/terminal-decision.json"
)


def test_q3_baseline_is_qwen38_only_and_pins_the_bf16_parent() -> None:
    source = SCRIPT.read_text()
    assert "Qwen3.8-27B-BF16.gguf" in source
    assert "Qwen3.8-27B-Q3_K_M.gguf" in source
    assert "5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e" in source
    assert "quant_type=${QUANT_TYPE:-Q3_K_M}" in source
    assert '"$quant_type" 4' in source
    assert "baseline_variant=${BASELINE_VARIANT:-q3_k_m}" in source
    assert "qwen35|qwen3[.]5|4b" in source


def test_q3_baseline_requires_quality_before_speed_and_mmlu_after_screen() -> None:
    source = SCRIPT.read_text()
    quick = source.index('printf \'complete\\n\' > "$state_dir/QUICK_QUALITY_COMPLETE"')
    speed = source.index("run_screen_row q4")
    mmlu = source.index('if [[ ! -s "$state_dir/mmlu.json" ]]')
    assert quick < speed < mmlu
    assert '"cute": cute_correct >= 7' in source
    assert '"math500": scores["math500"] >= 4' in source
    assert '"code": scores["code_trace"] >= 4' in source
    assert 'ppl / q4["wikitext_ppl"] <= 1.10' in source
    assert 'mmlu_correct >= 49' in source


def test_q3_baseline_has_material_cpu_speed_screen() -> None:
    source = SCRIPT.read_text()
    assert 'taskset -c 0,2,4,6 "$llama_build/bin/llama-bench"' in source
    assert 'rows["q3"]["decode"]["median_tokens_per_second"] >= 2.25' in source
    assert "decode_speedup >= 1.15" in source
    assert "prefill_fraction >= 0.90" in source
    assert "not a Clark research breakthrough" in source


def test_q3_k_l_is_one_adjacent_qwen38_control_with_a_stop_rule() -> None:
    import json

    payload = json.loads(Q3_K_L_PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["scope"]["quantization"] == "Q3_K_L"
    assert payload["scope"]["one_adjacent_stock_format_only"] is True
    assert payload["scope"]["no_iq3_or_additional_stock_three_bit_sweep_after_this_run"] is True
    assert payload["trigger"]["q3_k_m_cute_correct_of_20"] == 6
    assert "stop stock three-bit" in payload["stop_rules"][0]


def test_q3_k_m_terminal_receipt_preserves_exact_negative_denominators() -> None:
    import json

    payload = json.loads(Q3_K_M_TERMINAL.read_text())
    assert payload["terminal"] is True
    assert payload["infrastructure_valid"] is True
    assert payload["decision"] == "reject_before_speed_and_mmlu"
    assert payload["quality"]["cute"] == {
        "correct": 6,
        "n": 20,
        "required": 7,
        "passed": False,
        "breakdown": {
            "contains_char": "4/4",
            "ins_char": "0/4",
            "del_char": "2/4",
            "swap_char": "0/4",
            "sub_char": "0/4",
        },
        "report_content_sha256": "a531360940ec35c7c31046ec1e3d6100ee260c9731a0484b55e161462024949c",
    }
    assert payload["quality"]["math500"]["correct"] == 3
    assert payload["quality"]["math500"]["n"] == 8
    assert payload["quality"]["code_trace"]["correct"] == 4
    assert payload["quality"]["code_trace"]["n"] == 6
    assert payload["quality"]["wikitext"]["ppl"] == 6.2277
    assert payload["gates"]["speed_run"] is False
    assert payload["gates"]["mmlu_run"] is False


def test_q3_k_l_terminal_receipt_closes_stock_three_bit_sweep() -> None:
    import json

    payload = json.loads(Q3_K_L_TERMINAL.read_text())
    assert payload["terminal"] is True
    assert payload["infrastructure_valid"] is True
    assert payload["decision"] == "reject_before_speed_and_mmlu"
    assert payload["quality"]["cute"]["correct"] == 7
    assert payload["quality"]["cute"]["n"] == 20
    assert payload["quality"]["cute"]["passed"] is True
    assert payload["quality"]["math500"] == {
        "correct": 3,
        "n": 8,
        "required": 4,
        "passed": False,
    }
    assert payload["quality"]["code_trace"]["correct"] == 4
    assert payload["quality"]["code_trace"]["n"] == 6
    assert payload["quality"]["wikitext"]["ppl"] == 6.2307
    assert payload["quality"]["wikitext"]["passed"] is True
    assert payload["gates"]["speed_run"] is False
    assert payload["gates"]["mmlu_run"] is False
    assert payload["stock_three_bit_sweep_closed"] is True
