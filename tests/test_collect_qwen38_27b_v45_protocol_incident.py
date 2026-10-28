from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime/collect_qwen38_27b_v45_protocol_incident.py"


def load_module():
    spec = importlib.util.spec_from_file_location("incident_v45", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_layers(module):
    return [
        {
            "schema": module.LAYER_SCHEMA,
            "model": module.MODEL_ID,
            "revision": module.MODEL_REVISION,
            "layer": layer,
            "mode": "rank16_int8_residual",
            "base_validation_nmse": 0.01,
            "final_validation_nmse": 0.0101 if layer in (4, 7) else 0.0099,
            "relative_validation_nmse_reduction": -0.01 if layer in (4, 7) else 0.01,
            "persistent_payload_bytes": 174144,
            "complete": True,
            "terminal": True,
        }
        for layer in range(32)
    ]


def test_v45_incident_summary_retains_denominator_and_regressions() -> None:
    module = load_module()
    summary = module.summarize_partial_layers(synthetic_layers(module))
    assert summary["terminal_layers"] == 32
    assert summary["active_residual_layers"] == 32
    assert summary["exact_bypass_layers"] == 0
    assert summary["regressed_layers"] == [4, 7]
    assert summary["frozen_no_validation_regression_gate_already_missed"]
    assert summary["frozen_mean_reduction_gate_already_missed"]


def test_v45_incident_rejects_denominator_drift() -> None:
    module = load_module()
    layers = synthetic_layers(module)
    layers.pop()
    with pytest.raises(ValueError, match="denominator"):
        module.summarize_partial_layers(layers)


def test_v45_incident_requires_exact_protocol_failure_and_stopped_state() -> None:
    module = load_module()
    log = (
        '{"event": "v45_layer_started", "layer": 32}\n'
        + module.FAILURE_TEXT
        + "\nScheduled restart job, restart counter is at 1\n"
    )
    state = (
        "ActiveState=inactive\nSubState=dead\nreceipt.json=absent\n"
        "complete.json=absent\n32\n/x/layers/32/layer-32.csr\n"
        "/x/layers/32/layer-32.json\n"
    )
    module.validate_protocol_evidence(log, state)
    with pytest.raises(ValueError, match="transcript"):
        module.validate_protocol_evidence(log.replace(module.FAILURE_TEXT, "other"), state)
