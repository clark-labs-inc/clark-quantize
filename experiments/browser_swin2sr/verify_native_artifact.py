#!/usr/bin/env python3
"""Verify the native Rust loader/kernel against independent NumPy decoding."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.packing import unpack_trits_20


DEFAULT_TENSOR = (
    "swin2sr.encoder.stages.0.layers.0.attention.self.query.weight"
)
MAX_ABSOLUTE_ERROR = 1e-6
MAX_MEAN_ABSOLUTE_ERROR = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def run_native(runtime: Path, artifact: Path, tensor: str) -> dict:
    process = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--bin",
            "artifact_conformance",
            "--",
            str(artifact / "model.json"),
            str(artifact / "model.bin"),
            tensor,
        ],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def numpy_reference(artifact: Path, tensor_name: str, batch_size: int) -> np.ndarray:
    manifest = json.loads((artifact / "model.json").read_text())
    record = manifest["tensors"][tensor_name]
    blob = (artifact / "model.bin").read_bytes()
    code_range = record["codes"]
    scale_range = record["scales"]
    packed = np.frombuffer(
        blob,
        dtype="<u4",
        count=code_range["bytes"] // 4,
        offset=code_range["offset"],
    )
    scales = np.frombuffer(
        blob,
        dtype="<f2",
        count=scale_range["bytes"] // 2,
        offset=scale_range["offset"],
    ).astype(np.float32)
    shape = tuple(record["shape"])
    codes = unpack_trits_20(packed, shape, record["pad"]).astype(np.float32)
    flat = codes.reshape(-1)
    group_size = int(manifest["group_size"])
    weight = (
        flat.reshape(-1, group_size) * scales.reshape(-1, 1)
    ).reshape(shape)
    inputs = np.asarray(
        [
            ((index * 11 % 29) - 14) / 13
            for index in range(batch_size * shape[1])
        ],
        dtype=np.float32,
    ).reshape(batch_size, shape[1])
    return inputs @ weight.T


def main() -> None:
    args = parse_args()
    runtime = Path(__file__).parent / "runtime"
    native = run_native(runtime, args.artifact.resolve(), args.tensor)
    reference = numpy_reference(
        args.artifact,
        args.tensor,
        int(native["batch_size"]),
    )
    actual = np.asarray(native["output"], dtype=np.float32).reshape(reference.shape)
    delta = np.abs(actual - reference)
    maximum = float(delta.max())
    mean = float(delta.mean())
    passed = maximum <= MAX_ABSOLUTE_ERROR and mean <= MAX_MEAN_ABSOLUTE_ERROR
    report = {
        "schema_version": 1,
        "artifact": str(args.artifact),
        "tensor": args.tensor,
        "shape": native["shape"],
        "batch_size": native["batch_size"],
        "group_size": native["group_size"],
        "max_absolute_error": maximum,
        "mean_absolute_error": mean,
        "native_checksum": native["checksum"],
        "numpy_checksum": float(
            sum(
                (index + 1) * value
                for index, value in enumerate(reference.reshape(-1))
            )
        ),
        "tolerances": {
            "max_absolute_error": MAX_ABSOLUTE_ERROR,
            "mean_absolute_error": MAX_MEAN_ABSOLUTE_ERROR,
        },
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
