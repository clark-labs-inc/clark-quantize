from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "qwen38_27b_activation_binary_mlp_preregistration_v1.json"
)


def test_activation_binary_mlp_preregistration_is_frozen_and_consistent() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert payload["status"] == "frozen_before_full_real_activation_run"
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["target_parameters"] == 267_386_880
    assert payload["scope"]["complete_gate_up_down_required"] is True
    assert payload["candidate"]["rank"] == 40
    assert payload["candidate"]["projected_complete_mlp_bits_per_weight"] == (
        1.5398763020833333
    )
    assert payload["gate"]["maximum_complete_mlp_bits_per_weight"] == 1.55
    for descriptor in payload["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]


def test_activation_binary_mlp_preregistration_forbids_narrow_claims() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert payload["scope"]["sampled_weight_rows_forbidden"] is True
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["interpretation"]["weight_or_activation_loss_is_not_capability"] is True
    assert payload["interpretation"]["breakthrough_claim_permitted"] is False
    assert "PENDING" not in PREREGISTRATION.read_text()
