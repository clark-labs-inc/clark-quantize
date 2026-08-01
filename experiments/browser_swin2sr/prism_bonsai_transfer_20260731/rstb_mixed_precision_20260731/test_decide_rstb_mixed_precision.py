import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_rstb_mixed_precision.py")
SPEC = importlib.util.spec_from_file_location("decide_rstb_mixed", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(gt=1.0, gt_lower=0.9, edge=1.0, teacher=2.0):
    def metric(mean, lower=0.0):
        return {
            "samples": 3040,
            "candidate_minus_reference_mean": mean,
            "paired_bootstrap_95ci": [lower, mean + 0.01],
        }

    return {
        "metrics": {
            "psnr_vs_ground_truth": metric(gt, gt_lower),
            "edge_psnr_vs_ground_truth": metric(edge),
            "psnr_vs_teacher": metric(teacher),
        }
    }


def test_candidate_must_retain_ninety_percent_of_every_fp16_gain():
    fp16 = comparison()
    passing = MODULE.candidate_gate(comparison(0.91, 0.8, 0.91, 1.81), fp16, 0.9)
    failing = MODULE.candidate_gate(comparison(0.91, 0.8, 0.89, 1.81), fp16, 0.9)
    assert passing["passed"] is True
    assert failing["passed"] is False


def test_candidate_requires_positive_ground_truth_lower_bound():
    result = MODULE.candidate_gate(comparison(1.0, 0.0, 1.0, 2.0), comparison(), 0.9)
    assert result["passed"] is False


def test_selection_prefers_smallest_passing_representation():
    results = {
        "int4_per_output": {"passed": True},
        "int4_per_kernel": {"passed": True},
        "int8_per_output": {"passed": True},
    }
    storage = {
        "int4_per_output": 10_560,
        "int4_per_kernel": 38_880,
        "int8_per_output": 75_360,
        "fp16": 204_480,
    }
    assert MODULE.choose(results, storage) == "int4_per_output"


def test_selection_retains_fp16_when_no_low_bit_arm_passes():
    results = {
        "int4_per_output": {"passed": False},
        "int4_per_kernel": {"passed": False},
        "int8_per_output": {"passed": False},
    }
    storage = {
        "int4_per_output": 10_560,
        "int4_per_kernel": 38_880,
        "int8_per_output": 75_360,
        "fp16": 204_480,
    }
    assert MODULE.choose(results, storage) == "fp16"


def test_rejects_nonpositive_fp16_effect():
    try:
        MODULE.candidate_gate(comparison(), comparison(edge=0.0), 0.9)
    except ValueError as error:
        assert "FP16 RSTB effect" in str(error)
    else:
        raise AssertionError("nonpositive FP16 effect should be rejected")
