import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("decide_retained_precision_discovery.py")
SPEC = importlib.util.spec_from_file_location(
    "decide_retained_precision_discovery", MODULE_PATH
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


def test_discovery_gate_passes_only_all_three_frozen_checks():
    result = MODULE.island_gate(comparison(0.001, 0.0, 0.02))
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_discovery_gate_rejects_zero_ground_truth_lower_bound():
    result = MODULE.island_gate(comparison(0.0, 1.0, 1.0))
    assert result["passed"] is False
    assert result["checks"]["ground_truth_lower_95_positive"] is False


def test_discovery_gate_rejects_negative_secondary_metric():
    edge = MODULE.island_gate(comparison(0.01, -1e-9, 1.0))
    teacher = MODULE.island_gate(comparison(0.01, 1.0, -1e-9))
    assert edge["passed"] is False
    assert teacher["passed"] is False


def test_comparison_output_path_is_taken_from_sealed_command():
    command = ["python", "compare.py", "--out", "result.json"]
    assert MODULE.command_output(command) == Path("result.json")


def test_comparison_verification_pins_reports_states_and_sample_count(tmp_path):
    ternary = tmp_path / "ternary.json"
    fp16 = tmp_path / "fp16.json"
    ternary.write_text(json.dumps({"state_sha256": "ternary-state"}))
    fp16.write_text(json.dumps({"state_sha256": "fp16-state"}))
    comparison_payload = {
        "reference": {
            "path": str(ternary),
            "state_sha256": "ternary-state",
        },
        "candidate": {"path": str(fp16), "state_sha256": "fp16-state"},
        "metrics": {
            name: {"samples": 3040}
            for name in (
                "psnr_vs_ground_truth",
                "edge_psnr_vs_ground_truth",
                "psnr_vs_teacher",
            )
        },
    }
    MODULE.verify_comparison(comparison_payload, ternary, fp16)

    comparison_payload["candidate"]["state_sha256"] = "other-state"
    with pytest.raises(ValueError, match="candidate state hash"):
        MODULE.verify_comparison(comparison_payload, ternary, fp16)

    comparison_payload["candidate"]["state_sha256"] = "fp16-state"
    comparison_payload["metrics"]["psnr_vs_teacher"]["samples"] = 3039
    with pytest.raises(ValueError, match="3,040"):
        MODULE.verify_comparison(comparison_payload, ternary, fp16)
