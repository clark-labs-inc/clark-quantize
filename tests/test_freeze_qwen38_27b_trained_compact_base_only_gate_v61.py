from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "freeze_qwen38_27b_trained_compact_base_only_gate_v61.py"
V60 = RUNTIME / "qwen38_27b_trained_compact_whole_model_gate_preregistration_v60.json"
INCIDENT = RUNTIME / (
    "receipts/qwen38_27b_all_layer_sensitivity_residual_v45_protocol_incident/incident.json"
)
AMENDMENT = RUNTIME / "qwen38_27b_trained_compact_base_only_gate_amendment_v61.json"


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_v61", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_negative(module):
    layers = [
        {"layer": layer, "complete": True, "terminal": True}
        for layer in range(64)
    ]
    return {
        "schema": module.V45_SCHEMA,
        "model": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "repository_commit": "v45-commit",
        "preregistration_sha256": "v45-prereg",
        "training_receipt_sha256": "v44-receipt",
        "layers": layers,
        "gate": {
            "required_layers": 64,
            "observed_layers": 64,
            "all_layers_terminal": True,
            "all_active_layers_calibration_not_regressed": True,
            "all_active_layers_validation_not_regressed": False,
            "all_bypass_layers_exact": True,
            "all_final_validation_nmse_within_maximum": True,
            "all_pilot_modes_exact": True,
            "all_active_pilot_payloads_exact": True,
            "pilot_bypass_layer_has_no_payload": True,
            "residual_payload_within_maximum": True,
            "mean_active_relative_validation_nmse_reduction": 0.01,
            "minimum_mean_active_relative_validation_nmse_reduction": 0.1,
            "passed": False,
        },
        "complete": True,
        "terminal": True,
    }


def test_v61_freezes_separate_residual_free_candidate_without_gate_changes() -> None:
    module = load_module()
    v60 = json.loads(V60.read_text())
    payload = module.build_amendment(
        v45_receipt=synthetic_negative(module),
        v45_receipt_sha256="v45-sha",
        v60_preregistration=v60,
        generator_sha256="generator-sha",
        repository_commit="generator-commit",
        frozen_at_utc="2026-08-19T00:00:00Z",
    )
    assert payload["status"].startswith("frozen_after_v45_negative")
    assert payload["terminal_v45_outcome"]["original_v60_residual_candidate_rejected"]
    assert payload["candidate_identity"]["post_block_residual_payload_bytes"] == 0
    assert payload["candidate_identity"]["QWEN38_RESIDUAL_DIR_must_be_unset"]
    assert payload["candidate_identity"]["v45_payload_materialization_forbidden"]
    assert payload["inherited_frozen_gates"] == {
        "cute_correct_minimum": 7,
        "math500_correct_minimum": 4,
        "code_correct_minimum": 5,
        "mmlu_correct_minimum": 51,
        "candidate_over_q4_wikitext_ppl_ratio_maximum": 1.05,
        "candidate_weighted_tokens_per_second_minimum": 2.75,
        "candidate_over_q4_weighted_decode_speedup_minimum": 1.4,
        "candidate_weighted_tokens_per_second_stretch": 3.2,
        "free_generation_required": True,
        "resident_memory_and_energy_receipts_required": True,
    }


@pytest.mark.parametrize("mutation", ["passed", "short", "not_terminal", "no_failure"])
def test_v61_rejects_nonterminal_or_nonnegative_v45(mutation) -> None:
    module = load_module()
    receipt = synthetic_negative(module)
    if mutation == "passed":
        receipt["gate"]["passed"] = True
    elif mutation == "short":
        receipt["layers"].pop()
    elif mutation == "not_terminal":
        receipt["layers"][9]["terminal"] = False
    else:
        receipt["gate"]["all_active_layers_validation_not_regressed"] = True
        receipt["gate"]["mean_active_relative_validation_nmse_reduction"] = 0.1
    with pytest.raises(ValueError):
        module.validate_v45_negative(receipt)


def test_v61_uses_atomic_no_overwrite_and_clean_source_contract() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite a frozen v61 amendment" in source
    assert '"--untracked-files=no"' in source
    assert "os.replace(temporary, path)" in source


def test_v61_accepts_authenticated_protocol_negative_incident() -> None:
    module = load_module()
    outcome = module.validate_v45_outcome(json.loads(INCIDENT.read_text()))
    assert outcome["classification"] == "terminal_protocol_negative_no_v45_receipt"
    assert outcome["preregistration_sha256"] == (
        "c4b64d7bdb6af2e6e09fc857cf0defd7bcd9a2b24903beb3b5023c3692a51664"
    )
    assert "pilot_sensitivity_mode_reproduction_protocol" in outcome[
        "failed_frozen_gates"
    ]


def test_v61_frozen_amendment_binds_actual_incident_and_preserves_v60_gates() -> None:
    payload = json.loads(AMENDMENT.read_text())
    assert payload["frozen_inputs"]["v45_outcome_sha256"] == (
        "c88dac6c4c12cfa7533fb032f2a4bb8fa147123c5c182788627a2bfb3ab6fbe2"
    )
    assert payload["parent_v60"][
        "all_manifests_prompts_quality_thresholds_and_performance_thresholds_unchanged"
    ]
    assert payload["candidate_identity"]["post_block_residual_payload_bytes"] == 0
    assert payload["decision_rules"]["failed_v45_residual_must_never_execute"]
