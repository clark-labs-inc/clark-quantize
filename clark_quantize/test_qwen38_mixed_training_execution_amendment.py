from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
AMENDMENT = (
    RUNTIME
    / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_training_execution_amendment_v1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixed_training_execution_is_frozen_before_training() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["status"] == (
        "frozen_after_gate_0_before_any_mixed_training_or_confirmation_score"
    )
    assert amendment["state_at_freeze"]["mixed_training_had_started"] is False
    assert amendment["state_at_freeze"]["fresh_confirmation_had_been_scored"] is False
    assert amendment["fixed_block_order"] == [0, 32, 63]
    assert amendment["calibration_identity"]["selected_rows"] == 64
    assert amendment["calibration_identity"]["validation_rows_forbidden"] is True
    assert amendment["parent_preregistration_sha256"] == sha256(
        RUNTIME / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1.json"
    )
    assert amendment["parent_pre_execution_amendment_sha256"] == sha256(
        RUNTIME
        / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_pre_execution_amendment_v1.json"
    )
    assert amendment["gate_0_receipt_sha256"] == sha256(
        RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/direct-screen.json"
    )


def test_mixed_training_execution_has_one_calibration_only_path() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    cq3 = amendment["cq3_execution"]
    aff3 = amendment["aff3_execution"]
    assert cq3["row_shard"] == 128
    assert cq3["steps_per_shard"] == 6
    assert cq3["selection_candidates"] == list(range(7))
    assert cq3["gate_and_up_are_frozen_before_down_input_recomputation"] is True
    assert aff3["outer_steps"] == 8
    assert aff3["global_l2_radius_each_step"] == 0.02
    assert aff3["backtracking_factors_in_order"] == [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
    ]
    assert amendment["gate_1_scoring_order"][2].startswith(
        "open the already sealed fresh-confirmation cache"
    )
    assert amendment["complete"] is True
    assert amendment["terminal"] is True
