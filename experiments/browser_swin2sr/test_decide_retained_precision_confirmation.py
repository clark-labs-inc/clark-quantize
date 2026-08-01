import importlib.util
import math
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_retained_precision_confirmation.py")
SPEC = importlib.util.spec_from_file_location(
    "decide_retained_precision_confirmation", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(gt=True, edge=True, teacher=True, gt_mean=0.1):
    return {
        "checks": {
            "ground_truth_lower_95_positive": gt,
            "edge_mean_nonnegative": edge,
            "teacher_mean_nonnegative": teacher,
        },
        "ground_truth_mean": gt_mean,
        "edge_mean": 0.02 if edge else -0.02,
        "teacher_mean": 0.03 if teacher else -0.03,
    }


def test_final_gate_requires_every_seed_to_pass_every_check():
    passing = {31: result(), 32: result(gt_mean=0.2), 33: result(gt_mean=0.3)}
    decision = MODULE.final_gate(passing)
    assert decision["passed"] is True
    assert math.isclose(
        decision["three_seed_means"]["ground_truth_mean"], 0.2
    )


def test_final_gate_rejects_one_seed_ground_truth_failure():
    seeds = {31: result(), 32: result(gt=False), 33: result()}
    decision = MODULE.final_gate(seeds)
    assert decision["passed"] is False
    assert decision["checks"]["all_seed_ground_truth_lower_95_positive"] is False


def test_final_gate_rejects_one_seed_secondary_failure():
    edge = {31: result(), 32: result(edge=False), 33: result()}
    teacher = {31: result(), 32: result(), 33: result(teacher=False)}
    assert MODULE.final_gate(edge)["passed"] is False
    assert MODULE.final_gate(teacher)["passed"] is False
