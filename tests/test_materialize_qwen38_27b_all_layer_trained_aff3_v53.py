import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "materialize_qwen38_27b_all_layer_trained_aff3_v53.py"
)
SPEC = importlib.util.spec_from_file_location("trained_aff3_v53", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parent_receipt(schema: str, layer_schema: str) -> dict:
    layers = [
        {
            "schema": layer_schema,
            "layer": layer,
            "passed": True,
            "complete": True,
            "terminal": True,
        }
        for layer in range(64)
    ]
    return {
        "schema": schema,
        "model": MODULE.MODEL,
        "revision": MODULE.REVISION,
        "repository_commit": "a" * 40,
        "preregistration_sha256": "b" * 64,
        "gate": {"passed": True},
        "layers": layers,
        "complete": True,
        "terminal": True,
    }


def test_v53_requires_exact_passed_64_layer_parents() -> None:
    direct_sha = "c" * 64
    direct = parent_receipt(MODULE.DIRECT_SCHEMA, MODULE.DIRECT_LAYER_SCHEMA)
    trained = parent_receipt(
        MODULE.TRAINING_SCHEMA, MODULE.TRAINING_LAYER_SCHEMA
    )
    trained["direct_receipt_sha256"] = direct_sha

    MODULE.validate_parent_receipts(
        trained, direct, direct_receipt_sha256=direct_sha
    )
    trained["layers"].pop()
    with pytest.raises(ValueError, match="denominator"):
        MODULE.validate_parent_receipts(
            trained, direct, direct_receipt_sha256=direct_sha
        )


def test_v53_source_derives_exact_runtime_algebra_without_dense_weights() -> None:
    source = SOURCE.read_text()

    assert "build_from_compressed_artifact" in source
    assert "deterministic_givens_pairings" in source
    assert 'f"layer-{layer:02d}.aff3x16"' in source
    assert '"persistent_dense_weight_bytes": 0' in source
    assert "all_images_authenticated" in source
    assert "list(range(64))" in source
    assert "for layer in range(64)" in source


def test_v53_safe_child_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        MODULE.safe_child(tmp_path, "../outside")
