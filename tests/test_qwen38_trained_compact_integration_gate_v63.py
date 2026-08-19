from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_trained_compact_integration_gate_preregistration_v63.json"
RUNNER = RUNTIME / "run_qwen38_27b_trained_compact_integration_gate_v63.sh"
LAUNCHER = RUNTIME / "launch_qwen38_27b_trained_compact_integration_gate_v63.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v63_preregistration_binds_materialization_and_runtime_source() -> None:
    prereg = json.loads(PREREG.read_text())
    parent = prereg["parent_materialization"]
    source = prereg["authenticated_runtime_source"]
    assert prereg["model"] == "Qwen/Qwen3.8-27B"
    assert parent["preregistration_sha256"] == sha256(
        RUNTIME / "qwen38_27b_trained_compact_x8_runtime_preregistration_v62.json"
    )
    assert parent["runner_sha256"] == sha256(
        RUNTIME / "run_qwen38_27b_trained_compact_materialization_v62.sh"
    )
    assert parent["candidate_target_tensor_count"] == 192
    assert parent["aff3_layer_count"] == 64
    assert parent["residual_payload_bytes"] == 0
    assert source["tracked_diff_sha256"] == (
        "f76b42067337f35d8f5721b92faffcff90f4e1cd15afe8e4aa22ec4fdf67eb4a"
    )
    assert source["v54_transform_receipt_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_cq3_aff3_llamacpp_v54-transform.json"
    )


def test_v63_freezes_exact_representative_graph_parity() -> None:
    prereg = json.loads(PREREG.read_text())
    gate = prereg["correctness_gate"]
    rules = prereg["decision_rules"]
    assert gate["trained_aff3_layers"] == [0, 32, 63]
    assert gate["token_counts"] == [1, 8]
    assert gate["ggml_graph_output_must_be_bit_exact_to_direct_v47"]
    assert gate["maximum_absolute_error"] == 0
    assert gate["observed_four_task_mask"] == 15
    assert gate["persistent_dense_weight_bytes"] == 0
    assert gate["residual_environment_must_be_unset"]
    assert rules["whole_model_generation_forbidden_inside_v63"]
    assert rules["performance_claim_forbidden_inside_v63"]
    assert rules["v59_lut_runtime_remains_forbidden"]
    assert rules["v45_residual_runtime_remains_forbidden"]


def test_v63_runner_is_fail_closed_and_resource_bounded() -> None:
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert f"expected_prereg={sha256(PREREG)}" in runner
    assert "sha256sum -c \"$v62_manifest\"" in runner
    assert "status --porcelain" in runner
    assert "CLARK_QWEN38_27B_CQ3_X8_BENCH" in runner
    assert "for layer in 0 32 63" in runner
    assert "for tokens in 1 8" in runner
    assert "output_bit_exact_vs_direct_v47" in runner
    assert "whole_model_generation_ran\": False" in runner
    assert "QWEN38_RESIDUAL_DIR" in runner
    assert "MemoryMax=14000000000" in launcher
    assert "MemorySwapMax=0" in launcher
    assert "AllowedCPUs=0,2,4,6" in launcher
    assert "RestartPreventExitStatus=2" in launcher
