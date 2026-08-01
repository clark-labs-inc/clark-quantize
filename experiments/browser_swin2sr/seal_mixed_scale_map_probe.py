#!/usr/bin/env python3
"""Freeze category-specific scale maps before their image results are observed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_scale_codebook_study import FORMAT as PRIOR_FORMAT


FORMAT = "backdropkit-swin2sr-mixed-scale-map-protocol-v1"
SOURCE_FILES = (
    "experiments/browser_swin2sr/mixed_scale_codebook.py",
    "experiments/browser_swin2sr/evaluate_mixed_scale_maps.py",
    "experiments/browser_swin2sr/evaluate_scale_codebooks.py",
    "experiments/browser_swin2sr/scale_codebook.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/rewrite_onnx_ternary.py",
    "experiments/browser_swin2sr/rotation_control.py",
)
MAPS = {
    "uniform_q1": {"core_linear": 1, "image_io_conv": 1, "position_bias_linear": 1, "residual_body_conv": 1},
    "image_io_q4": {"core_linear": 1, "image_io_conv": 4, "position_bias_linear": 1, "residual_body_conv": 1},
    "position_bias_q4": {"core_linear": 1, "image_io_conv": 1, "position_bias_linear": 4, "residual_body_conv": 1},
    "residual_body_q4": {"core_linear": 1, "image_io_conv": 1, "position_bias_linear": 1, "residual_body_conv": 4},
    "core_q4": {"core_linear": 4, "image_io_conv": 1, "position_bias_linear": 1, "residual_body_conv": 1},
    "sensitive_io_position": {"core_linear": 1, "image_io_conv": 4, "position_bias_linear": 2, "residual_body_conv": 1},
    "sensitive_io_position_residual": {"core_linear": 1, "image_io_conv": 4, "position_bias_linear": 2, "residual_body_conv": 2},
    "uniform_q2": {"core_linear": 2, "image_io_conv": 2, "position_bias_linear": 2, "residual_body_conv": 2},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"mixed scale protocol exists: {args.out}")
    prior = json.loads(args.prior_protocol.read_text())
    if prior.get("format") != PRIOR_FORMAT:
        raise RuntimeError("mixed probe requires the sealed uniform-scale protocol")
    if prior.get("model_promotion_authorized") or not prior.get("final_heldout_locked"):
        raise RuntimeError("prior scale protocol violates promotion isolation")
    repository = Path(__file__).resolve().parents[2]
    protocol = {
        "format": FORMAT,
        "status": "preregistered_before_category_scale_image_results",
        "selection_source": "development_mechanism_diagnostic",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "prior_protocol": str(args.prior_protocol),
        "prior_protocol_sha256": file_sha256(args.prior_protocol),
        "materialization": prior["materialization"],
        "materialization_sha256": prior["materialization_sha256"],
        "bundle": prior["bundle"],
        "bundle_sha256": prior["bundle_sha256"],
        "bundle_manifest_sha256": prior["bundle_manifest_sha256"],
        "q4_reference": prior["q4_reference"],
        "q4_reference_sha256": prior["q4_reference_sha256"],
        "screen": prior["screen"],
        "screen_source_counts": prior["screen_source_counts"],
        "candidate_maps": MAPS,
        "interpretation": (
            "category isolation only; results may choose scale-QAT arms but cannot "
            "authorize promotion or access final heldout"
        ),
        "source_sha256": {
            name: file_sha256(repository / name) for name in SOURCE_FILES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "candidate_maps": len(MAPS),
        "screen_source_counts": protocol["screen_source_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
