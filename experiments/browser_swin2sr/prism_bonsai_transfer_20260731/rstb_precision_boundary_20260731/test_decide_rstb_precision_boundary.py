import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_rstb_precision_boundary.py")
SPEC = importlib.util.spec_from_file_location("decide_boundary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_choose_smallest_passing_boundary_arm():
    results = {
        "int5_per_output": {"passed": False},
        "int6_per_output": {"passed": True},
        "int5_per_kernel": {"passed": True},
        "int7_per_output": {"passed": True},
        "int6_per_kernel": {"passed": True},
    }
    storage = {
        "int5_per_output": 26_760,
        "int6_per_output": 42_960,
        "int5_per_kernel": 55_080,
        "int7_per_output": 59_160,
        "int6_per_kernel": 71_280,
        "int8_per_output": 75_360,
    }
    assert MODULE.choose(results, storage) == "int6_per_output"


def test_choose_prior_int8_when_no_boundary_arm_passes():
    results = {
        "int5_per_output": {"passed": False},
        "int6_per_output": {"passed": False},
    }
    storage = {
        "int5_per_output": 26_760,
        "int6_per_output": 42_960,
        "int8_per_output": 75_360,
    }
    assert MODULE.choose(results, storage) == "int8_per_output"
