from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_activation_dual_binary_method_preregistration_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_activation_dual_method_is_frozen_before_real_execution() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["status"] == "method_frozen_before_any_real_activation_dual_run"
    assert payload["execution_authorized"] is False
    assert payload["scope"]["target_parameters"] == 267_386_880
    assert payload["scope"]["complete_gate_up_down_required"] is True
    assert payload["scope"]["sampled_weight_rows_forbidden"] is True
    assert payload["candidate"]["rank"] == 40
    assert payload["candidate"]["projected_complete_mlp_bits_per_weight"] < 1.55
    assert payload["gate"]["minimum_full_mlp_validation_error_reduction_vs_frozen_right"] == 0.05
    assert payload["execution_contract"]["runner_and_watcher_sha256_must_be_added_before_launch"] is True


def test_activation_dual_method_sources_are_byte_frozen() -> None:
    payload = json.loads(PREREG.read_text())
    for descriptor in payload["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]
