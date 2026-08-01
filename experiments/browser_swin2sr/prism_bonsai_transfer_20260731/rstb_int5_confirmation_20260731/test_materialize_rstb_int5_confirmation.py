import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("materialize_rstb_int5_confirmation.py")
SPEC = importlib.util.spec_from_file_location("rstb_int5_confirmation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def receipt(seed, precision, parent="parent"):
    return {
        "parent_checkpoint_sha256": parent,
        "retained_precision_training": {
            "seed": seed,
            "island": "rstb_fusion_fp16",
            "precision": precision,
        },
    }


def test_source_pair_accepts_exact_seed_and_parent():
    MODULE.verify_source_pair(
        receipt(20260732, "ternary"),
        receipt(20260732, "fp16"),
        20260732,
    )


def test_source_pair_rejects_cross_seed_sources():
    try:
        MODULE.verify_source_pair(
            receipt(20260732, "ternary"),
            receipt(20260733, "fp16"),
            20260732,
        )
    except RuntimeError as error:
        assert "FP16 source seed" in str(error)
    else:
        raise AssertionError("cross-seed source pair was accepted")


def test_source_pair_rejects_different_parents():
    try:
        MODULE.verify_source_pair(
            receipt(20260732, "ternary", "a"),
            receipt(20260732, "fp16", "b"),
            20260732,
        )
    except RuntimeError as error:
        assert "different parents" in str(error)
    else:
        raise AssertionError("different-parent source pair was accepted")
