from __future__ import annotations

from pathlib import Path

from seal_scale_codebook_qat_study import SOURCE_FILES, verify_source_closure


def test_scale_qat_seal_has_complete_local_import_closure() -> None:
    repository = Path(__file__).resolve().parents[2]
    resolved = verify_source_closure(repository)

    assert set(resolved) == set(SOURCE_FILES)
    assert "experiments/browser_swin2sr/onnx_packed_bridge.py" in resolved[
        "experiments/browser_swin2sr/rewrite_onnx_ternary.py"
    ]
