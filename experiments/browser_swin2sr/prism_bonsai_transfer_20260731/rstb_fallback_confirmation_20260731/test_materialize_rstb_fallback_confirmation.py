import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("materialize_rstb_fallback_confirmation.py")
SPEC = importlib.util.spec_from_file_location("rstb_fallback", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fallback_geometries_are_exact():
    weight = torch.randn(60, 60, 3, 3)
    expected = {
        "int7_per_output": (28_350, 120),
        "int6_per_kernel": (24_300, 7_200),
    }
    for arm, config in MODULE.ARM_SPECS.items():
        _, receipt = MODULE.quantize_weight_nbit(weight, **config)
        assert (receipt["code_bytes"], receipt["scale_bytes"]) == expected[arm]
        assert receipt["scale_fp16_roundtrip_exact"]


def test_source_pair_rejects_cross_seed():
    def receipt(seed):
        return {
            "parent_checkpoint_sha256": "p",
            "retained_precision_training": {"seed": seed},
        }
    try:
        MODULE.verify_source_pair(receipt(20260732), receipt(20260733), 20260732)
    except RuntimeError as error:
        assert "FP16 source seed" in str(error)
    else:
        raise AssertionError("cross-seed pair was accepted")
