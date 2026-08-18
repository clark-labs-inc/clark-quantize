from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
AMENDMENT = RUNTIME / "qwen38_27b_llamacpp_ternary_controls_execution_amendment_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tq_execution_repair_preserves_all_scientific_inputs() -> None:
    payload = json.loads(AMENDMENT.read_text())
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["execution_authorized"] is True
    assert payload["initial_attempt"]["classification"] == "infrastructure_invalid"
    assert payload["initial_attempt"]["model_conversion_started"] is False
    assert payload["remediation"]["scientific_method_changed"] is False
    assert payload["remediation"]["conversion_runner_changed"] is False
    assert payload["remediation"]["thresholds_changed"] is False
    for descriptor in payload["frozen_sources"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_tq_watcher_invokes_the_nonexecutable_runner_through_bash() -> None:
    watcher = (RUNTIME / "watch_qwen38_27b_llamacpp_ternary_controls_v1.sh").read_text()
    assert 'bash "$runtime/run_qwen38_27b_llamacpp_ternary_controls_v1.sh"' in watcher
    assert '"$runtime/qwen38_27b_llamacpp_ternary_controls_execution_amendment_v1.json"' in watcher
    assert "chmod" not in watcher
