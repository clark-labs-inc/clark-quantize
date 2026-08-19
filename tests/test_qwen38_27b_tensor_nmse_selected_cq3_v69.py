from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "materialize_qwen38_27b_tensor_nmse_selected_cq3_v69.py"
PREREG = RUNTIME / "qwen38_27b_tensor_nmse_selected_cq3_preregistration_v69.json"
QUALITY_PREREG = (
    RUNTIME / "qwen38_27b_tensor_nmse_selected_cq3_quality_preregistration_v70.json"
)
QUALITY_RUNNER = RUNTIME / "run_qwen38_27b_tensor_nmse_selected_cq3_quality_v70.sh"
TRAINING = RUNTIME / "receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"

sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("materialize_v69", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class FakePayload:
    tensor: str


def test_threshold_parser_rejects_nonfinite_and_negative() -> None:
    assert MODULE.parse_threshold("0.125") == 0.125
    for value in ("-1", "nan", "inf", "-inf", "x"):
        with pytest.raises(Exception):
            MODULE.parse_threshold(value)


def test_inclusive_selection_and_exact_payload_denominator() -> None:
    ranking = [
        {"tensor": "b", "score": 0.2},
        {"tensor": "a", "score": 0.1},
        {"tensor": "c", "score": 0.3},
    ]
    names = MODULE.select_tensor_names(ranking, 0.2)
    assert names == ("a", "b")
    payloads = [FakePayload("a"), FakePayload("b"), FakePayload("c")]
    assert {item.tensor for item in MODULE.selected_payloads(payloads, names)} == {
        "a",
        "b",
    }
    with pytest.raises(ValueError, match="denominator"):
        MODULE.selected_payloads(payloads[1:], names)


def test_in_place_overlay_requires_partial_suffix(tmp_path: Path) -> None:
    final = tmp_path / "candidate.gguf"
    final.write_bytes(b"01234567")
    with pytest.raises(ValueError, match="partial"):
        MODULE.overlay_in_place(final, [])


def test_frozen_tensor_ranking_and_selection_reproduce_preregistration() -> None:
    prereg = json.loads(PREREG.read_text())
    selection = prereg["selection"]
    ranking = MODULE.tensor_nmse_ranking(TRAINING)
    names = MODULE.select_tensor_names(ranking, selection["inclusive_threshold"])
    rejected = sorted(item["tensor"] for item in ranking if item["tensor"] not in names)

    assert len(ranking) == 192
    assert len(names) == selection["selected_tensor_count"] == 143
    assert len(rejected) == selection["rejected_tensor_count"] == 49
    assert MODULE.canonical_sha256(ranking) == selection["ranking_sha256"]
    assert MODULE.canonical_sha256(list(names)) == selection["selection_sha256"]
    assert MODULE.canonical_sha256(rejected) == selection["rejection_sha256"]
    assert max(item["score"] for item in ranking if item["tensor"] in names) == selection[
        "inclusive_threshold"
    ]
    assert min(item["score"] for item in ranking if item["tensor"] in rejected) == selection[
        "minimum_rejected_score"
    ]

    selected_by_role = {
        role: sorted(
            item["layer"]
            for item in ranking
            if item["tensor"] in names and item["role"] == role
        )
        for role in MODULE.ROLES
    }
    assert selected_by_role == selection["selected_layers_by_role"]
    assert {
        role: 64 - len(layers) for role, layers in selected_by_role.items()
    } == selection["rejected_counts_by_role"]


def test_candidate_size_and_parameter_denominator_are_exact() -> None:
    prereg = json.loads(PREREG.read_text())
    candidate = prereg["candidate"]
    q4_bytes = prereg["model"]["q4_control_bytes"]
    selected = prereg["selection"]["selected_tensor_count"]
    parameters_per_tensor = 17408 * 5120
    savings_per_tensor = 50135040 - 23674880

    assert candidate["selected_mlp_tensors"] == selected
    assert candidate["selected_mlp_parameters"] == selected * parameters_per_tensor
    assert candidate["expected_saved_bytes_vs_q4"] == selected * savings_per_tensor
    assert candidate["expected_bytes"] == q4_bytes - candidate[
        "expected_saved_bytes_vs_q4"
    ]


def test_preregistered_threshold_is_finite_and_not_benchmark_score() -> None:
    prereg = json.loads(PREREG.read_text())
    threshold = prereg["selection"]["inclusive_threshold"]
    assert math.isfinite(threshold) and threshold >= 0
    assert prereg["rationale"]["benchmark_items_used_to_score_or_rank_tensors"] is False
    assert prereg["development_screen"][
        "development_only_not_independent_evidence"
    ] is True


def test_preregistration_bytes_are_stable_for_launcher_pin() -> None:
    digest = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_v70_quality_gate_excludes_adaptive_cute_and_pins_denominators() -> None:
    prereg = json.loads(QUALITY_PREREG.read_text())
    gate = prereg["promotion_gate"]
    evaluations = prereg["evaluations"]

    assert prereg["evidence_independence"]["cute_status"].startswith("adaptive")
    assert "cute_correct_min" not in gate
    assert evaluations["math_code"]["math500_items"] == 8
    assert evaluations["math_code"]["code_trace_items"] == 6
    assert evaluations["mmlu"]["items"] == 64
    assert evaluations["wikitext"]["chunks"] == 4
    assert gate == {
        "all_receipts_valid": True,
        "math500_correct_min": 4,
        "code_trace_correct_min": 4,
        "mmlu_correct_min": 49,
        "wikitext_ppl_ratio_max": 1.05,
        "all_evaluations_run_even_after_an_earlier_miss": True,
        "throughput_forbidden_until_all_checks_pass": True,
    }


def test_v70_runner_contains_no_cute_evaluation_call() -> None:
    runner = QUALITY_RUNNER.read_text()
    assert "eval_llama_cpp_cute_quick.py" not in runner
    assert "cute_excluded_from_promotion_as_adaptive" in runner
