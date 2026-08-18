from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "qwen38_27b_binary_low_rank_residual_preregistration_v1.json"
)


def test_binary_low_rank_preregistration_is_frozen_and_self_consistent() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert payload["status"] == "frozen_before_full_real_weight_run"
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["all_rows_and_columns_required"] is True
    assert payload["scope"]["sampled_row_spans_forbidden"] is True
    assert payload["candidate"]["rank"] == 40
    assert payload["gate"]["maximum_component_bits_per_weight"] == 1.55
    for descriptor in payload["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]


def test_binary_low_rank_preregistration_has_no_unfrozen_placeholders() -> None:
    assert "PENDING" not in PREREGISTRATION.read_text()
