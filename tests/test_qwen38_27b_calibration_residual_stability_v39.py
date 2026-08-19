from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_calibration_residual_stability_v39.json"
RECEIPTS = RUNTIME / "receipts/qwen38_27b_calibration_residual_stability_v39"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v39_frozen_inputs_and_unchanged_kernel_are_exact() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["status"] == "frozen_after_terminal_v38_before_confirmation_run"
    assert prereg["scope"]["kernel_code_change_from_v38"] is False
    assert prereg["scope"]["payload_change_from_v38"] is False
    for descriptor in prereg["frozen_inputs"].values():
        path = REPO / descriptor["path"]
        assert sha256(path) == descriptor["sha256"]


def test_v39_parent_failed_only_the_raw_range_gate() -> None:
    prereg = json.loads(PREREG.read_text())
    decision = json.loads((REPO / prereg["parent"]["terminal_decision_path"]).read_text())
    receipt = json.loads((REPO / prereg["parent"]["receipt_path"]).read_text())
    assert decision["decision"] == "reject_v38_on_preregistered_full_range_stability_gate"
    assert receipt["gate"]["passed"] is False
    failed = [name for name, passed in receipt["gate"]["checks"].items() if not passed]
    assert failed == [prereg["parent"]["required_only_failed_check"]]
    assert prereg["post_v38_protocol_rationale"]["v38_remains_rejected"] is True


def test_v39_block_median_protocol_is_fixed_and_balanced() -> None:
    prereg = json.loads(PREREG.read_text())
    measurement = prereg["measurement"]
    assert measurement["blocks_per_arm_per_payload"] == 7
    assert measurement["warmups_per_arm_per_block"] == 3
    assert measurement["samples_per_arm_per_block"] == 9
    assert measurement["inner_native_calls_per_sample"] == 128
    assert measurement["total_timed_native_calls"] == 2 * 2 * 7 * 9 * 128
    gate = prereg["gate"]
    assert gate["minimum_avx2_speedup_vs_scalar"] == 1.25
    assert gate["maximum_projected_all_64_layers_avx2_ns"] == 7272727
    assert gate["maximum_avx2_block_median_range_fraction"] == 0.1


def test_v39_runner_retains_every_raw_sample_and_alternates_order() -> None:
    prereg = json.loads(PREREG.read_text())
    source = (REPO / prereg["frozen_inputs"]["benchmark"]["path"]).read_text()
    assert '"samples_ns": observed' in source
    assert "block_index % 2 == 0" in source
    assert "summarize_blocks" in source
    assert "median_of_block_medians_ns" in source
    assert "library_sha256" in source


def test_v39_terminal_pass_retains_all_raw_samples_and_v38_rejection() -> None:
    receipt = json.loads((RECEIPTS / "receipt.json").read_text())
    decision = json.loads((RECEIPTS / "terminal-decision.json").read_text())
    assert receipt["gate"]["passed"] is True
    assert decision["decision"] == (
        "admit_unchanged_exact_residual_kernel_to_llama_cpp_integration_gate"
    )
    assert decision["receipt"]["sha256"] == sha256(RECEIPTS / "receipt.json")
    assert decision["unchanged_kernel"]["v38_remains_rejected"] is True
    raw_samples = sum(
        len(block["samples_ns"])
        for row in receipt["records"]
        for arm in ("scalar_timing", "avx2_timing")
        for block in row[arm]["blocks"]
    )
    assert raw_samples == decision["denominators"]["retained_raw_timing_samples"] == 252
    assert all(receipt["gate"]["checks"].values())
