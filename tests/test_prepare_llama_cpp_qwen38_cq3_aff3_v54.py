from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/llama_cpp/"
    "prepare_llama_cpp_qwen38_cq3_aff3_v54.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_cq3_aff3_v54", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v54_freezes_exact_accepted_patch_chain() -> None:
    assert MODULE.LLAMA_REVISION == "4df29be4f4c3673f428170fda944a5b19f743bb8"
    assert [MODULE.sha256_path(path) for path, _ in MODULE.PATCHES] == [
        expected for _, expected in MODULE.PATCHES
    ]
    assert [path.name for path, _ in MODULE.PATCHES] == [
        "llama_cpp_4df29be_cq3.patch",
        "llama_cpp_4df29be_cq3_x86_pair_v1.patch",
        "llama_cpp_cq3_avx2_x8_repack_v15.patch",
        "llama_cpp_cq3_avx2_x8_repack_integration_v15.patch",
    ]


def test_v54_accepts_only_the_native_pass_not_a_model_speed_claim() -> None:
    receipt = json.loads(MODULE.ACCEPTED_RECEIPT.read_text())
    MODULE.validate_accepted_receipt(receipt)

    assert receipt["native_kernel_passed"] is True
    assert receipt["selective_model_promoted"] is False
    rejected = dict(receipt, native_kernel_passed=False)
    with pytest.raises(ValueError, match="native gates"):
        MODULE.validate_accepted_receipt(rejected)


def test_v54_composes_aff3_before_cq3_and_authenticates_output() -> None:
    source = SOURCE.read_text()

    aff3_position = source.index("aff3_output = subprocess.check_output")
    patch_position = source.index("applied = []", aff3_position)
    assert aff3_position < patch_position
    assert '"git", "apply", "--check"' in source
    assert '"git", "diff", "--check"' in source
    assert '"working_tree_files_sha256"' in source
    assert '"high_precision_weight_materialization": False' in source
