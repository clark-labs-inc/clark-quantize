from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREGISTRATION = RUNTIME / "qwen38_27b_trained_compact_x8_runtime_preregistration_v62.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v62_binds_negatives_and_forbids_failed_runtime_paths() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    outcomes = preregistration["frozen_outcomes"]
    assert preregistration["status"].startswith("frozen_after_v45_and_v58_negatives")
    assert outcomes["v45_residual"]["incident_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_all_layer_sensitivity_residual_v45_protocol_incident/incident.json"
    )
    assert outcomes["v45_residual"]["payload_materialization_forbidden"]
    assert outcomes["v58_lut32"]["terminal_decision_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_cq3_exact_lut32_avx2_v58/terminal-decision.json"
    )
    assert outcomes["v58_lut32"]["v59_implementation_forbidden"]


def test_v62_authenticates_existing_x8_runtime_and_materializers() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    parent = preregistration["accepted_runtime_parent"]
    materialization = preregistration["materialization"]
    assert parent["cq3_x8_v15_receipt_sha256"] == sha256(
        RUNTIME / "receipts/qwen38_27b_cq3_avx2_x8_repack_v15/terminal-decision.json"
    )
    assert parent["cq3_x8_native_speedup"] == 1.30177569
    assert parent["cq3_x8_storage_expansion_fraction"] == 0.0
    assert materialization["trained_cq3_overlay"]["sha256"] == sha256(
        RUNTIME / "materialize_qwen38_27b_all_layer_trained_cq3_v46.py"
    )
    assert materialization["trained_aff3_images"]["sha256"] == sha256(
        RUNTIME / "materialize_qwen38_27b_all_layer_trained_aff3_v53.py"
    )
    assert materialization["post_block_residual_payload_bytes"] == 0


def test_v62_preserves_whole_model_quality_and_speed_gates() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    gates = preregistration["whole_model_gate_inheritance"]
    assert gates["v60_preregistration_sha256"] == sha256(
        RUNTIME / "qwen38_27b_trained_compact_whole_model_gate_preregistration_v60.json"
    )
    assert gates["v61_amendment_sha256"] == sha256(
        RUNTIME / "qwen38_27b_trained_compact_base_only_gate_amendment_v61.json"
    )
    assert gates["candidate_over_q4_wikitext_ppl_ratio_maximum"] == 1.05
    assert gates["conventional_weighted_tokens_per_second_minimum"] == 2.75
    assert gates["candidate_over_q4_weighted_decode_speedup_minimum"] == 1.4
    assert gates["mtp_cannot_rescue_conventional_failure"]
