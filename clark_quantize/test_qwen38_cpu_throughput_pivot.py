from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PIVOT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_cpu_throughput_pivot_v1.json"
)


def _payload() -> dict:
    return json.loads(PIVOT.read_text())


def test_pivot_optimizes_qwen38_cpu_throughput_without_fixed_bit_ceiling() -> None:
    payload = _payload()
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["promotion_gate"]["memory_and_storage"][
        "fixed_bits_per_weight_ceiling"
    ] is None
    assert payload["research_candidate"]["base3_hot_decode_forbidden"] is True


def test_pivot_keeps_capability_and_material_speed_gates() -> None:
    payload = _payload()
    quality = payload["promotion_gate"]["quality"]
    performance = payload["promotion_gate"]["performance"]
    assert quality == {
        "minimum_cute_correct_of_20": 7,
        "minimum_math500_correct_of_8": 4,
        "minimum_code_correct_of_6": 4,
        "minimum_mmlu_correct_of_64": 49,
        "maximum_wikitext_ppl_ratio_vs_q4": 1.1,
    }
    assert performance["minimum_decode_speedup_vs_q4"] >= 1.15
    assert performance["minimum_four_physical_core_decode_tokens_per_second"] >= 2.25
    assert payload["research_candidate"]["post_training_required_for_two_bit_bulk"] is True


def test_breakthrough_gate_is_nearly_lossless_and_materially_faster() -> None:
    gate = _payload()["breakthrough_gate"]
    assert gate["quality"]["maximum_wikitext_ppl_ratio_vs_q4"] <= 1.05
    assert gate["quality"]["minimum_mmlu_correct_of_64"] >= 50
    assert gate["performance"]["minimum_four_physical_core_decode_tokens_per_second"] >= 2.75
    assert gate["performance"]["minimum_decode_speedup_vs_q4"] >= 1.40
    assert gate["performance"]["stretch_four_physical_core_decode_tokens_per_second"] >= 3.2
    assert len(gate["claim_requirements"]) == 6


def test_q3_baseline_is_pinned_and_not_presented_as_research() -> None:
    baseline = _payload()["first_practical_baseline"]
    assert baseline["format"] == "Q3_K_M"
    assert baseline["source"]["bf16_bytes"] == 53_808_281_952
    assert len(baseline["source"]["bf16_sha256"]) == 64
    assert baseline["not_a_breakthrough_claim"] is True
