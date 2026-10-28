from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_row_scale_pipeline_v3"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_row_scale_terminal_receipts_are_authenticated() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    assert decision["service_terminal_observation"]["result"] == "success"
    assert decision["service_terminal_observation"]["exec_main_status"] == 0
    for descriptor in decision["receipts"].values():
        path = ROOT / descriptor["path"]
        assert path.stat().st_size == descriptor["bytes"]
        assert sha256(path) == descriptor["sha256"]
    pipeline = json.loads((ROOT / "pipeline/receipt.json").read_text())
    assert pipeline["terminal"] is True
    assert pipeline["valid_nonpromotion"] is True


def test_mapped_parent_completed_and_failed_reconstruction_not_storage() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    gate = decision["mapped_gate"]
    mapped = json.loads((ROOT / "results/mapped.json").read_text())
    assert mapped["complete"] is True
    assert len(mapped["records"]) == gate["complete_tensors"] == 402
    assert mapped["summary"]["scales"] == gate["complete_scales"] == 209_920_000
    assert gate["storage_subgate_passed"] is True
    assert gate["reconstruction_subgate_passed"] is False
    assert gate["tensors_over_normalized_rmse_limit"] == sum(
        row["metrics"]["normalized_rmse"] > gate["maximum_tensor_normalized_rmse"]
        for row in mapped["records"]
    )
    assert gate["tensors_over_p99_relative_error_limit"] == sum(
        row["metrics"]["p99_relative_error_nonzero_scales"]
        > gate["maximum_tensor_p99_relative_error"]
        for row in mapped["records"]
    )


def test_supplemental_fit_was_correctly_skipped_without_complete_artifact_claim() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    skipped = json.loads((ROOT / "results/full-skip.json").read_text())
    assert skipped["terminal"] is True
    assert skipped["full_fit_launched"] is False
    assert decision["supplemental_gate"]["launched"] is False
    assert decision["decision"]["complete_851_tensor_artifact_claimed"] is False
    assert decision["decision"]["retuning_authorized"] is False
