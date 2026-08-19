from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREGISTRATION = (
    RUNTIME / "qwen38_27b_trained_compact_whole_model_gate_preregistration_v60.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v60_freezes_qwen38_only_before_candidate_observation() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    assert preregistration["status"] == (
        "frozen_before_candidate_materialization_or_whole_model_observation"
    )
    assert preregistration["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert preregistration["scope"]["qwen_4b_compute_forbidden"]
    assert preregistration["candidate_admission"][
        "high_precision_weight_materialization_forbidden"
    ]


def test_v60_authenticates_frozen_manifests_evaluators_and_q4_controls() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    frozen = preregistration["frozen_inputs"]
    assert frozen["cute_manifest"]["sha256"] == sha256(
        RUNTIME / "cute_quick_v1.json"
    )
    assert frozen["math_code_manifest"]["sha256"] == sha256(
        RUNTIME / "math_code_canary_v1.json"
    )
    assert frozen["mmlu_manifest"]["sha256"] == sha256(
        RUNTIME / "qwen35_4b_cpu_layer30_mmlu_v1.json"
    )
    assert frozen["evaluators"] == {
        "cute_sha256": sha256(RUNTIME / "eval_llama_cpp_cute_quick.py"),
        "math_code_sha256": sha256(
            RUNTIME / "eval_llama_cpp_math_code_canary.py"
        ),
        "mmlu_sha256": sha256(RUNTIME / "eval_llama_cpp_mmlu.py"),
        "wikitext_sha256": sha256(RUNTIME / "eval_llama_cpp_wikitext_quick.py"),
    }
    control = RUNTIME / "receipts/qwen38_27b_cpu_q4_baseline_v1"
    assert frozen["q4_baseline"] == {
        "summary_sha256": sha256(control / "summary.json"),
        "cute_sha256": sha256(control / "cute.json"),
        "math_code_sha256": sha256(control / "math-code.json"),
        "mmlu_sha256": sha256(control / "mmlu.json"),
        "wikitext_sha256": sha256(control / "wikitext.json"),
    }


def test_v60_freezes_catastrophic_forgetting_and_conventional_speed_gates() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    protocol = preregistration["evaluation_protocol"]
    quality = preregistration["quality_gates"]
    performance = preregistration["performance_protocol"]

    assert protocol["cute"]["denominator"] == 20
    assert protocol["math500"]["denominator"] == 8
    assert protocol["code"]["denominator"] == 6
    assert protocol["mmlu"]["denominator"] == 64
    assert protocol["wikitext"]["chunks"] == 4
    assert quality["candidate_minimums"] == {
        "cute_correct": 7,
        "math500_correct": 4,
        "code_correct": 5,
        "mmlu_correct": 51,
    }
    assert quality["maximum_candidate_over_q4_wikitext_ppl_ratio"] == 1.05
    assert performance["mode"].startswith("conventional autoregressive")
    assert performance["minimum_candidate_weighted_tokens_per_second"] == 2.75
    assert performance["minimum_candidate_over_q4_weighted_decode_speedup"] == 1.4
    assert performance["stretch_candidate_weighted_tokens_per_second"] == 3.2
    assert preregistration["optional_speculation_after_conventional_pass"][
        "cannot_rescue_conventional_failure"
    ]
