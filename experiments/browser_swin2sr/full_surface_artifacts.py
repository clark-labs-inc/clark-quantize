"""Artifact commands shared by the paired full-surface study."""

from __future__ import annotations

import sys
from pathlib import Path


def materialization_command(args, destination: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("materialize_surface_states.py")),
        str(destination / "checkpoint-final.pt"),
        "--out",
        str(destination / "canonical"),
        "--source-device",
        args.device,
    ]


def portable_export_command(args, destination: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("build_portable_onnx_artifact.py")),
        str(destination / "canonical" / "materialization.json"),
        "--base-onnx",
        str(args.base_onnx),
        "--out",
        str(destination / "portable" / "model.onnx"),
        "--scope",
        "all_matrix_and_conv",
    ]
