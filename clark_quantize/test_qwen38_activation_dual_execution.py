from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
METHOD = RUNTIME / "qwen38_27b_activation_dual_binary_method_preregistration_v1.json"
EXECUTION = RUNTIME / "qwen38_27b_activation_dual_binary_execution_amendment_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_activation_dual_execution_is_bound_to_the_frozen_method() -> None:
    method = json.loads(METHOD.read_text())
    execution = json.loads(EXECUTION.read_text())
    assert method["execution_authorized"] is False
    assert execution["execution_authorized"] is True
    assert execution["launch_scheduled"] is False
    assert execution["model"] == method["model"] == "Qwen/Qwen3.8-27B"
    assert execution["revision"] == method["revision"]
    assert execution["method_preregistration_sha256"] == sha256(METHOD)
    assert execution["execution_scope"]["target_parameters"] == 267_386_880
    assert execution["execution_scope"]["candidate_sweep"] is False
    assert execution["execution_scope"]["qwen_4b_compute"] is False


def test_activation_dual_execution_sources_are_byte_frozen() -> None:
    execution = json.loads(EXECUTION.read_text())
    for descriptor in execution["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_activation_dual_execution_cannot_preempt_whole_model_controls() -> None:
    execution = json.loads(EXECUTION.read_text())
    service = (RUNTIME / "quant-qwen38-27b-activation-dual-binary-mlp-v1.service").read_text()
    assert "quant-qwen38-27b-row-scale-pipeline-v2.service" in service
    assert "Requires=quant-qwen38-27b-activation-binary-mlp-v1.service" in service
    assert execution["priority"]["queue_position"].startswith(
        "after authenticated whole-model"
    )
    assert execution["resource_contract"]["memory_max_bytes"] == 24 * 1024**3
    assert "MemoryMax=24G" in service
    assert "MemorySwapMax=0" in service


def test_activation_dual_runner_replays_both_same_rate_controls() -> None:
    runner = (RUNTIME / "run_qwen38_27b_activation_dual_binary_mlp_v1.py").read_text()
    assert 'RuntimeError(f"direct-ternary replay changed for {name}")' in runner
    assert 'RuntimeError(f"frozen-right replay changed for {name}")' in runner
    assert 'RuntimeError("complete direct-ternary replay changed")' in runner
    assert 'RuntimeError("complete frozen-right replay changed")' in runner
    assert '"scientific_novelty_or_breakthrough_proven": False' in runner
