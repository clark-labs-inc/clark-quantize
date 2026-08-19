from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_trained_compact_model_graph_preregistration_v64.json"
RUNNER = RUNTIME / "run_qwen38_27b_trained_compact_model_graph_v64.sh"
LAUNCHER = RUNTIME / "launch_qwen38_27b_trained_compact_model_graph_v64.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v64_binds_the_unobserved_integration_parent() -> None:
    prereg = json.loads(PREREG.read_text())
    parents = prereg["required_parents"]
    assert prereg["status"].startswith("frozen_after_v63_source_freeze")
    assert parents["v63_preregistration_sha256"] == sha256(
        RUNTIME / "qwen38_27b_trained_compact_integration_gate_preregistration_v63.json"
    )
    assert parents["v63_runner_sha256"] == sha256(
        RUNTIME / "run_qwen38_27b_trained_compact_integration_gate_v63.sh"
    )
    assert parents["v63_terminal_integration_pass_required"]


def test_v64_is_bounded_all_layer_free_generation_only() -> None:
    prereg = json.loads(PREREG.read_text())
    candidate = prereg["candidate_contract"]
    execution = prereg["execution"]
    gate = prereg["pass_gate"]
    assert candidate["trained_cq3_target_tensors"] == 192
    assert candidate["trained_aff3_layer_set"] == "0-63"
    assert candidate["trained_aff3_images"] == 64
    assert candidate["post_block_residual_payload_bytes"] == 0
    assert execution["requested_generated_tokens"] == 8
    assert execution["runtime_max_seconds"] == 420
    assert execution["file_size_max_bytes_per_stream"] == 1048576
    assert execution["performance_timing"] is False
    assert gate["generated_output_nonempty"]
    assert gate["interactive_mode_absent"]
    assert gate["performance_markers_absent"]


def test_v64_runner_enforces_resource_and_stream_bounds() -> None:
    prereg_sha = sha256(PREREG)
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert f"expected_prereg={prereg_sha}" in runner
    assert "QWEN38_AFF3_LAYERS=0-63" in runner
    assert "QWEN38_RESIDUAL_DIR" in runner
    assert "MemoryMax=30000000000" in runner
    assert "MemorySwapMax=0" in runner
    assert "RuntimeMaxSec=420" in runner
    assert "LimitFSIZE=1048576" in runner
    assert "--no-conversation" in runner
    assert "--single-turn" in runner
    assert "--no-perf" in runner
    assert "systemctl list-units" in launcher
