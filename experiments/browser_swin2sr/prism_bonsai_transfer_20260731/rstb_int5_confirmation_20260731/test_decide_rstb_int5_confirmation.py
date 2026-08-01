import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_rstb_int5_confirmation.py")
SPEC = importlib.util.spec_from_file_location("decide_rstb_int5_confirmation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(gt_mean=0.5, gt_lower=0.45, edge=0.4, teacher=1.0):
    return {
        "metrics": {
            "psnr_vs_ground_truth": {
                "candidate_minus_reference_mean": gt_mean,
                "paired_bootstrap_95ci": [gt_lower, gt_mean + 0.05],
            },
            "edge_psnr_vs_ground_truth": {
                "candidate_minus_reference_mean": edge,
            },
            "psnr_vs_teacher": {
                "candidate_minus_reference_mean": teacher,
            },
        }
    }


def test_fp16_gate_requires_all_three_axes():
    assert MODULE.fp16_gate(comparison())["passed"]
    assert not MODULE.fp16_gate(comparison(gt_lower=0.0))["passed"]
    assert not MODULE.fp16_gate(comparison(edge=-0.01))["passed"]
    assert not MODULE.fp16_gate(comparison(teacher=-0.01))["passed"]


def test_int5_candidate_is_judged_against_own_seed_fp16_effect():
    fp16 = comparison(gt_mean=0.5, edge=0.4, teacher=1.0)
    passing = comparison(gt_mean=0.46, edge=0.37, teacher=0.91)
    failing = comparison(gt_mean=0.46, edge=0.37, teacher=0.89)
    assert MODULE.candidate_gate(passing, fp16, 0.9)["passed"]
    assert not MODULE.candidate_gate(failing, fp16, 0.9)["passed"]
