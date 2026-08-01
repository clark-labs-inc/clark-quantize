#!/usr/bin/env python3
"""Create a canonical full-graph fixture for a packed q1 browser artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Swin2SRForImageSuperResolution

from evaluate_scale_final_candidate import (
    selected_candidate,
    validate_candidate_artifacts,
)
from full_surface_provenance import file_sha256
from prepare_full_onnx_smoke import MODEL_ID, MODEL_REVISION, fixture_input


FORMAT = "backdropkit-swin2sr-packed-v3-smoke-v1"
SOURCE_FILES = (
    "experiments/browser_swin2sr/evaluate_scale_final_candidate.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/pack_browser_v3_bundle.py",
    "experiments/browser_swin2sr/prepare_full_onnx_smoke.py",
    "experiments/browser_swin2sr/prepare_scale_final_candidate.py",
    "experiments/browser_swin2sr/prepare_scale_runtime_fixture.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.size < 8 or args.size % 8:
        raise ValueError("--size must be at least 8 and divisible by 8")
    candidate = selected_candidate(args.candidate_manifest)
    validate_candidate_artifacts(candidate)
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    state = torch.load(
        candidate["hard_state"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    values = fixture_input(args.size)
    with torch.inference_mode():
        output = model(torch.from_numpy(values)).reconstruction.cpu().numpy()
    package = candidate["runtime_package"]
    repository = Path(__file__).resolve().parents[2]
    payload = {
        "format": FORMAT,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "model": package["runtime_package"],
        "model_sha256": package["runtime_package_sha256"],
        "model_bytes": package["runtime_package_bytes"],
        "model_compression": package["runtime_package_compression"],
        "expanded_model_sha256": package["expanded_bundle_sha256"],
        "expanded_model_bytes": package["expanded_bundle_bytes"],
        "artifact_manifest_sha256": candidate["artifact_manifest"]["sha256"],
        "artifact_binary_sha256": candidate["artifact_binary"]["sha256"],
        "input_shape": list(values.shape),
        "output_shape": list(output.shape),
        "input": values.reshape(-1).tolist(),
        "expected": output.reshape(-1).tolist(),
        "source_files": {
            relative: file_sha256(repository / relative)
            for relative in SOURCE_FILES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "input_shape": payload["input_shape"],
        "output_shape": payload["output_shape"],
        "model_sha256": payload["model_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
