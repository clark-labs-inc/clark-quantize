from __future__ import annotations
import hashlib, json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
RUNTIME=REPO/"experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG=RUNTIME/"qwen38_27b_trained_compact_factorial_diagnosis_preregistration_v66.json"
RUNNER=RUNTIME/"run_qwen38_27b_trained_compact_factorial_diagnosis_v66.sh"
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def test_v66_is_exact_two_arm_causal_diagnosis():
    p=json.loads(PREREG.read_text()); arms=p["factorial_arms"]
    assert [a["name"] for a in arms]==["cq3_mlp_q4_attention","q4_mlp_aff3_attention"]
    assert arms[0]["aff3_layers"]=="none" and arms[1]["aff3_layers"]=="0-63"
    assert p["terminal_parent"]["v65_receipt_sha256"]==sha256(RUNTIME/"receipts/qwen38_27b_trained_compact_http_canary_v65/receipt.json")
    assert p["terminal_parent"]["full_candidate_rejected"]

def test_v66_runs_frozen_cute_denominators_without_performance():
    p=json.loads(PREREG.read_text()); e=p["evaluation"]
    assert e["cute_manifest_sha256"]==sha256(RUNTIME/"cute_quick_v1.json")
    assert e["cute_evaluator_sha256"]==sha256(RUNTIME/"eval_llama_cpp_cute_quick.py")
    assert e["cute_denominator_per_arm"]==20 and e["functional_arm_cute_correct_minimum"]==7
    assert e["cute_runs_even_if_france_canary_fails"]
    assert p["decision_rules"]["all_40_cute_items_required"]
    assert p["decision_rules"]["do_not_run_mmlu_or_performance_in_v66"]

def test_v66_runner_is_restartable_and_resource_bounded():
    source=RUNNER.read_text()
    assert f"expected_prereg={sha256(PREREG)}" in source
    assert "QWEN38_AFF3_LAYERS=0-63" in source
    assert "QWEN38_RESIDUAL_DIR" in source
    assert "MemoryMax=30000000000" in source and "MemorySwapMax=0" in source
    assert "cq3_mlp_q4_attention" in source and "q4_mlp_aff3_attention" in source
    assert 'len(cute.get("records", [])) != 20' in source
    assert 'cute_correct_minimum": 7' in source
