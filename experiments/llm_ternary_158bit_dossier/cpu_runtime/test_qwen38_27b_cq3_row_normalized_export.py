from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import pytest

from experiments.llm_ternary_158bit_dossier.cpu_runtime.run_qwen38_27b_cq3_export_v1 import (
    scale_sources,
)


def descriptor(root: Path, name: str, content: bytes) -> dict:
    path = root / name
    path.write_bytes(content)
    return {
        "path": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_row_normalized_scale_sources_preserve_canonical_part_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        record = {
            "rows": 2,
            "scale_count": 3,
            "source_scale_sha256": "1" * 64,
            "indices": descriptor(root, "indices.bin", b"\x21\x03"),
            "anchors": descriptor(root, "anchors.f16", b"\x00<\x00@"),
            "codebook": descriptor(root, "codebook.f16", bytes(32)),
        }
        scale, paths = scale_sources(
            "matrix.weight",
            {"matrix.weight": record},
            {},
            root,
            root,
            row_normalized=True,
        )
        assert [path.name for path in paths] == [
            "indices.bin",
            "anchors.f16",
            "codebook.f16",
        ]
        assert scale["anchors"] == record["anchors"]
        assert sum(path.stat().st_size for path in paths) == 38


def test_row_normalized_scale_sources_reject_wrong_anchor_coverage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        record = {
            "rows": 2,
            "scale_count": 3,
            "source_scale_sha256": "2" * 64,
            "indices": descriptor(root, "indices.bin", b"\x21\x03"),
            "anchors": descriptor(root, "anchors.f16", bytes(6)),
            "codebook": descriptor(root, "codebook.f16", bytes(32)),
        }
        with pytest.raises(RuntimeError, match="row anchors"):
            scale_sources(
                "matrix.weight",
                {"matrix.weight": record},
                {},
                root,
                root,
                row_normalized=True,
            )
