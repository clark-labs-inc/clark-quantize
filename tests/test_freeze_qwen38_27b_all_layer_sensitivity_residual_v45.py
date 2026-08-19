from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "freeze_qwen38_27b_all_layer_sensitivity_residual_v45.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_v45", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_receipt(module):
    layers = []
    for layer in range(64):
        layers.append(
            {
                "layer": layer,
                "passed": True,
                "pilot_payload_algebra_exact": (
                    True if layer in module.PILOT_LAYERS else None
                ),
                "pilot_score_exact_within_1e_8": (
                    True if layer in module.PILOT_LAYERS else None
                ),
            }
        )
    return {
        "schema": module.TRAINING_SCHEMA,
        "model": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "repository_commit": "training-commit",
        "preregistration_sha256": "training-prereg",
        "direct_receipt_sha256": "direct-receipt",
        "full_context_receipt_sha256": module.FULL_CONTEXT_RECEIPT_SHA256,
        "layers": layers,
        "gate": {
            "passed": True,
            "observed_layers": 64,
            "all_layers_calibration_not_regressed_vs_direct": True,
            "all_payload_roundtrips_exact": True,
            "all_values_finite": True,
            "all_pilot_payload_algebras_exact": True,
            "all_pilot_scores_exact_within_1e_8": True,
            "payload_storage_within_maximum": True,
            "mean_relative_calibration_nmse_reduction_vs_direct": 0.5,
            "total_selected_payload_bytes": 5_000_000_000,
        },
        "complete": True,
        "terminal": True,
    }


def test_build_v45_preregistration_binds_exact_v44_receipt() -> None:
    module = load_module()
    prereg = module.build_preregistration(
        training_receipt=synthetic_receipt(module),
        training_receipt_sha256="training-sha",
        runner_sha256="runner-sha",
        source_commit="source-commit",
        frozen_at_utc="2026-08-19T00:00:00Z",
    )
    assert prereg["status"] == "frozen_after_v44_pass_before_residual_fit"
    assert prereg["scope"]["target_layers"] == list(range(64))
    assert prereg["parent_training"]["receipt_sha256"] == "training-sha"
    assert prereg["frozen_inputs"]["runner_sha256"] == "runner-sha"
    assert prereg["sensitivity_rule"]["minimum_base_validation_nmse"] == 0.001
    assert prereg["adapter"]["rank"] == 16
    assert prereg["adapter"]["ridge_fraction"] == 0.01
    assert prereg["data_protocol"]["fresh_confirmation_examples_used"] == 0
    assert prereg["data_protocol"]["benchmark_examples_used"] == 0


@pytest.mark.parametrize("mutation", ["failed_layer", "wrong_order", "pilot_drift"])
def test_freeze_v45_rejects_changed_v44_terminal_evidence(mutation) -> None:
    module = load_module()
    receipt = synthetic_receipt(module)
    if mutation == "failed_layer":
        receipt["layers"][11]["passed"] = False
    elif mutation == "wrong_order":
        receipt["layers"][0]["layer"] = 1
    else:
        receipt["layers"][32]["pilot_payload_algebra_exact"] = False
    with pytest.raises(ValueError):
        module.validate_training_receipt(receipt)


def test_freeze_v45_rejects_failed_parent_gate() -> None:
    module = load_module()
    receipt = synthetic_receipt(module)
    receipt["gate"]["all_payload_roundtrips_exact"] = False
    with pytest.raises(ValueError, match="training gate"):
        module.validate_training_receipt(receipt)


def test_freeze_v45_uses_atomic_no_overwrite_contract() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite a frozen v45" in source
    assert '"--untracked-files=no"' in source
    assert "os.replace(temporary, path)" in source
