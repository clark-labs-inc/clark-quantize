from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_q3_k_m_baseline_v1.sh"
)


def test_q3_baseline_is_qwen38_only_and_pins_the_bf16_parent() -> None:
    source = SCRIPT.read_text()
    assert "Qwen3.8-27B-BF16.gguf" in source
    assert "Qwen3.8-27B-Q3_K_M.gguf" in source
    assert "5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e" in source
    assert "Q3_K_M 4" in source
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
