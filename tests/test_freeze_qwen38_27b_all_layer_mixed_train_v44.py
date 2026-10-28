from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "freeze_qwen38_27b_all_layer_mixed_train_v44.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_v44", SCRIPT)
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
                "direct_mixed_calibration_complete_block_nmse": 0.01
                + layer / 100_000,
                "passed": True,
                "aff3_pilot_algebra_exact": layer in module.PILOT_LAYERS or None,
                "cq3_pilot_algebra_exact": layer in module.PILOT_LAYERS or None,
                "pilot_score_exact_within_1e_8": (
                    layer in module.PILOT_LAYERS or None
                ),
            }
        )
    maximum = max(
        record["direct_mixed_calibration_complete_block_nmse"]
        for record in layers
    )
    return {
        "schema": module.DIRECT_SCHEMA,
        "model": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "repository_commit": "direct-commit",
        "preregistration_sha256": "direct-prereg",
        "full_context_receipt_sha256": module.FULL_CONTEXT_RECEIPT_SHA256,
        "layers": layers,
        "gate": {
            "passed": True,
            "maximum_observed_calibration_nmse": maximum,
            "all_layers_below_maximum": True,
            "total_selected_payload_bytes": 5_000_000_000,
            "payload_storage_within_maximum": True,
        },
        "complete": True,
        "terminal": True,
    }


def test_build_v44_preregistration_binds_exact_v43_receipt() -> None:
    module = load_module()
    prereg = module.build_preregistration(
        direct_receipt=synthetic_receipt(module),
        direct_receipt_sha256="receipt-sha",
        runner_sha256="runner-sha",
        source_commit="source-commit",
        frozen_at_utc="2026-08-19T00:00:00Z",
    )
    assert prereg["status"] == "frozen_after_v43_pass_before_training"
    assert prereg["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["target_layers"] == list(range(64))
    assert prereg["parent_direct"]["receipt_sha256"] == "receipt-sha"
    assert prereg["frozen_inputs"]["runner"] == {
        "path": "run_qwen38_27b_all_layer_mixed_train_v44.py",
        "sha256": "runner-sha",
        "repository_commit": "source-commit",
    }
    assert prereg["data_protocol"]["internal_validation_sequences_used"] == 0
    assert prereg["data_protocol"]["benchmark_examples_used"] == 0
    assert prereg["training"]["mlp_cq3"]["steps_per_row_shard"] == 6
    assert prereg["training"]["attention_output_aff3"]["outer_steps"] == 8


@pytest.mark.parametrize("mutation", ["failed_layer", "wrong_order", "pilot_drift"])
def test_freeze_rejects_changed_v43_terminal_evidence(mutation) -> None:
    module = load_module()
    receipt = synthetic_receipt(module)
    if mutation == "failed_layer":
        receipt["layers"][11]["passed"] = False
    elif mutation == "wrong_order":
        receipt["layers"][0]["layer"] = 1
    else:
        receipt["layers"][32]["aff3_pilot_algebra_exact"] = False
    with pytest.raises(ValueError):
        module.validate_direct_receipt(receipt)


def test_freeze_rejects_storage_or_score_beyond_frozen_gate() -> None:
    module = load_module()
    receipt = synthetic_receipt(module)
    receipt["gate"]["total_selected_payload_bytes"] = 5_100_000_001
    with pytest.raises(ValueError, match="numeric gate"):
        module.validate_direct_receipt(receipt)
    receipt = synthetic_receipt(module)
    receipt["layers"][63][
        "direct_mixed_calibration_complete_block_nmse"
    ] = 0.12001
    receipt["gate"]["maximum_observed_calibration_nmse"] = 0.12001
    with pytest.raises(ValueError, match="numeric gate"):
        module.validate_direct_receipt(receipt)


def test_freeze_uses_atomic_no_overwrite_cli_contract() -> None:
    source = SCRIPT.read_text()
    assert 'if args.output.exists()' in source
    assert "refusing to overwrite" in source
    assert '"--untracked-files=no"' in source
    assert "os.replace(temporary, path)" in source
