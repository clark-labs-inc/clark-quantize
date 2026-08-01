import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_image_io_ablation.py")
SPEC = importlib.util.spec_from_file_location("decide_image_io_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(gt=0.1, gt_lower=0.09, edge=0.1, teacher=0.2):
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


def test_candidate_must_retain_ninety_percent_of_every_joint_gain():
    joint = comparison()
    passing = MODULE.candidate_gate(comparison(0.091, 0.08, 0.091, 0.181), joint, 0.9)
    failing = MODULE.candidate_gate(comparison(0.091, 0.08, 0.089, 0.181), joint, 0.9)
    assert passing["passed"] is True
    assert failing["passed"] is False


def test_candidate_requires_positive_ground_truth_lower_bound():
    result = MODULE.candidate_gate(comparison(0.1, 0.0, 0.1, 0.2), comparison(), 0.9)
    assert result["passed"] is False


def test_selection_prefers_smallest_passing_subset():
    results = {
        "input_only": {"passed": True},
        "reconstruction_only": {"passed": True},
    }
    storage = {"input_only": 1_296, "reconstruction_only": 5_184, "joint": 6_480}
    assert MODULE.choose(results, storage) == "input_only"


def test_selection_retains_joint_when_no_subset_passes():
    results = {
        "input_only": {"passed": False},
        "reconstruction_only": {"passed": False},
    }
    storage = {"input_only": 1_296, "reconstruction_only": 5_184, "joint": 6_480}
    assert MODULE.choose(results, storage) == "joint"
