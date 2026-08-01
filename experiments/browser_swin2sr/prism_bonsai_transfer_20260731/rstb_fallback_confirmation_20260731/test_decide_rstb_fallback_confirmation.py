import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("decide_rstb_fallback_confirmation.py")
SPEC = importlib.util.spec_from_file_location("decide_rstb_fallback", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_choose_smallest_fallback_passing_both_seeds():
    results = {
        "int7_per_output": {"passed": True},
        "int6_per_kernel": {"passed": True},
    }
    storage = {
        "int7_per_output": 59_160,
        "int6_per_kernel": 71_280,
        "fp16": 204_480,
    }
    assert MODULE.choose(results, storage) == "int7_per_output"


def test_choose_fp16_when_no_fallback_passes():
    results = {
        "int7_per_output": {"passed": False},
        "int6_per_kernel": {"passed": False},
    }
    storage = {
        "int7_per_output": 59_160,
        "int6_per_kernel": 71_280,
        "fp16": 204_480,
    }
    assert MODULE.choose(results, storage) == "fp16"
