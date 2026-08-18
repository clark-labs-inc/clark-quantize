from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "qwen38_27b_llamacpp_ternary_controls_preregistration_v1.json"
)


def test_llamacpp_ternary_controls_are_frozen_and_qwen38_only() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert payload["status"] == "frozen_before_conversion_or_evaluation"
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert [row["name"] for row in payload["formats"]] == ["TQ1_0", "TQ2_0"]
    assert payload["resource_contract"]["serial_formats"] is True
    assert payload["resource_contract"]["final_artifacts_retained"] == 2
    assert payload["evaluation"]["early_rejection_is_valid"] is True


def test_llamacpp_ternary_control_sources_are_byte_frozen() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert "PENDING" not in PREREGISTRATION.read_text()
    for descriptor in payload["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
