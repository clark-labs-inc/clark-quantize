#!/usr/bin/env python3
"""Create a deterministic full-graph Swin2SR ONNX conformance fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

from build_portable_onnx_artifact import MODEL_ID, MODEL_REVISION


SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/build_portable_onnx_artifact.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/full_surface_optimizer.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/island_policy.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/mixed_precision_artifact.py",
    "experiments/browser_swin2sr/onnx_packed_bridge.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/prepare_full_onnx_smoke.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/recovery_controls.py",
    "experiments/browser_swin2sr/retained_precision_islands.py",
    "experiments/browser_swin2sr/rewrite_onnx_ternary.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/surface_artifact_state.py",
    "experiments/browser_swin2sr/surface_curriculum.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
    "experiments/browser_swin2sr/surface_run_state.py",
    "experiments/browser_swin2sr/surface_schedule.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/training_data_contract.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_input(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    denominator = float(size - 1)
    checker = ((x // 4 + y // 4) % 2).astype(np.float32)
    return np.stack(
        [
            x / denominator,
            y / denominator,
            checker * 0.75 + (x + y) / (8.0 * denominator),
        ],
        axis=0,
    )[None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    if args.size < 8 or args.size % 8 != 0:
        raise ValueError("--size must be at least 8 and divisible by 8")
    values = fixture_input(args.size)
    with torch.inference_mode():
        output = model(torch.from_numpy(values)).reconstruction.cpu().numpy()
    payload = {
        "format": "backdropkit-swin2sr-portable-onnx-smoke-v1",
        "hard_state_sha256": sha256_file(args.state),
        "input_name": "pixel_values",
        "output_name": "reconstruction",
        "input_shape": list(values.shape),
        "output_shape": list(output.shape),
        "input": values.reshape(-1).tolist(),
        "expected": output.reshape(-1).tolist(),
        "source_files": {
            name: sha256_file(Path(__file__).resolve().parents[2] / name)
            for name in SOURCE_FILES
        },
    }
    if args.model is not None:
        payload["model"] = str(args.model.resolve())
        payload["model_sha256"] = sha256_file(args.model)
    if args.package is not None:
        if args.model is None:
            raise ValueError("--package requires --model")
        payload["model_package"] = str(args.package.resolve())
        payload["model_package_compression"] = "gzip"
        payload["model_package_bytes"] = args.package.stat().st_size
        payload["model_package_sha256"] = sha256_file(args.package)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(args.out),
                "fixture_bytes": args.out.stat().st_size,
                "input_shape": payload["input_shape"],
                "output_shape": payload["output_shape"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
