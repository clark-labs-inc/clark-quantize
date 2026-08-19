from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_imatrix_cq3_hybrid_materialization_v80.py"
PREREG = RUNTIME / "qwen38_27b_imatrix_cq3_hybrid_materialization_preregistration_v80.json"
LAUNCHER = RUNTIME / "launch_qwen38_27b_imatrix_cq3_hybrid_materialization_v80.sh"
SPEC = importlib.util.spec_from_file_location("imatrix_cq3_v80", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v80_is_one_whole_model_composition_not_an_adaptive_layer_sweep() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["frozen_before_candidate_materialization"] is True
    assert prereg["authenticated_parents"]["trained_cq3_partition"]["selected_layer_count"] == 32
    assert prereg["authenticated_parents"]["trained_cq3_partition"]["selected_tensor_count"] == 96
    assert prereg["stop_rules"][0] == "do not tune the 32-layer partition after observing v80"
    assert len(MODULE.LAYERS) == 32
    assert len(MODULE.expected_tensors()) == 96


def test_v80_complete_artifact_math_and_quality_gates_are_explicit() -> None:
    prereg = json.loads(PREREG.read_text())
    artifact = prereg["artifact_contract"]
    assert artifact["expected_bytes"] == MODULE.EXPECTED_CANDIDATE_BYTES
    assert artifact["bytes_saved_vs_imatrix_q4"] == (
        MODULE.EXPECTED_SOURCE_BYTES - MODULE.EXPECTED_CANDIDATE_BYTES
    )
    assert artifact["honest_complete_file_bits_per_text_parameter"] > 3.88
    assert artifact["honest_complete_file_bits_per_text_parameter"] < 3.89
    gate = prereg["next_whole_model_gate_if_materialization_passes"]
    assert gate["minimum_cute_correct_of_20"] == 8
    assert gate["minimum_math500_correct_of_8"] == 4
    assert gate["minimum_code_correct_of_6"] == 4
    assert gate["minimum_mmlu_correct_of_64"] == 50
    assert gate["maximum_wikitext_ppl_ratio_vs_q4"] == 1.05


def test_v80_runner_restricts_changes_to_selected_mlp_triplets() -> None:
    expression = MODULE.target_regex()
    assert expression.startswith("^blk\\.")
    assert expression.endswith("\\.ffn_(down|gate|up)\\.weight$")
    assert "46" not in expression.split("(", 1)[1].split(")", 1)[0].split("|")
    source = MODULE_PATH.read_text()
    assert '"--baseline-target-type",\n                "12"' in source
    assert '"--candidate-target-type",\n                "43"' in source
    assert "all_nonselected_file_bytes_identical" in PREREG.read_text()


def test_v80_launcher_is_disk_bounded_and_restartable() -> None:
    source = LAUNCHER.read_text()
    assert "MemoryMax=30000000000" in source
    assert "MemorySwapMax=0" in source
    assert "Restart=on-failure" in source
    assert "AllowedCPUs=0,2,4,6" in source
    runner = MODULE_PATH.read_text()
    assert "partial.unlink()" in runner
    assert "candidate exists without its materialization receipt" in runner
    assert "need {required} free bytes" in runner
