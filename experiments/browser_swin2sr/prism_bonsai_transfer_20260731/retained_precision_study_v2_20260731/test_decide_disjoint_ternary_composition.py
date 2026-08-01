import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("decide_disjoint_ternary_composition.py")
SPEC = importlib.util.spec_from_file_location(
    "decide_disjoint_ternary_composition", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(gt_lower=0.01, edge_mean=0.0, teacher_mean=0.0):
    def metric(mean, lower):
        return {
            "samples": 3040,
            "candidate_minus_reference_mean": mean,
            "paired_bootstrap_95ci": [lower, mean + 0.1],
        }

    return {
        "metrics": {
            "psnr_vs_ground_truth": metric(0.05, gt_lower),
            "edge_psnr_vs_ground_truth": metric(edge_mean, edge_mean - 0.1),
            "psnr_vs_teacher": metric(teacher_mean, teacher_mean - 0.1),
        }
    }


def test_gate_requires_every_frozen_check():
    assert MODULE.gate(comparison(0.001, 0.0, 0.1))["passed"] is True
    assert MODULE.gate(comparison(0.0, 1.0, 1.0))["passed"] is False
    assert MODULE.gate(comparison(0.1, -1e-9, 1.0))["passed"] is False
    assert MODULE.gate(comparison(0.1, 1.0, -1e-9))["passed"] is False


def test_comparison_verification_pins_paths_hashes_and_sample_count(tmp_path):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text('{"state_sha256":"reference-state"}')
    candidate.write_text('{"state_sha256":"candidate-state"}')
    payload = comparison()
    payload.update({
        "reference": {
            "path": str(reference),
            "state_sha256": "reference-state",
        },
        "candidate": {
            "path": str(candidate),
            "state_sha256": "candidate-state",
        },
    })
    MODULE.verify_comparison(payload, reference, candidate)
    payload["candidate"]["state_sha256"] = "wrong"
    with pytest.raises(ValueError, match="candidate state hash"):
        MODULE.verify_comparison(payload, reference, candidate)


def test_comparison_verification_rejects_partial_development_set(tmp_path):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text('{"state_sha256":"reference-state"}')
    candidate.write_text('{"state_sha256":"candidate-state"}')
    payload = comparison()
    payload.update({
        "reference": {
            "path": str(reference),
            "state_sha256": "reference-state",
        },
        "candidate": {
            "path": str(candidate),
            "state_sha256": "candidate-state",
        },
    })
    payload["metrics"]["psnr_vs_teacher"]["samples"] = 3039
    with pytest.raises(ValueError, match="3,040"):
        MODULE.verify_comparison(payload, reference, candidate)
