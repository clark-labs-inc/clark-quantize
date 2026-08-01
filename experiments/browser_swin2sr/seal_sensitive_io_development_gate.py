#!/usr/bin/env python3
"""Seal the full-bundle sensitive-I/O development gate before results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from full_surface_provenance import file_sha256


SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/decide_sensitive_io_development.py",
    "experiments/browser_swin2sr/decide_sensitive_io_rescue.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/evaluate_hard_started_stratified.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/full_surface_optimizer.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/island_policy.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/recovery_controls.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/seal_sensitive_io_development_gate.py",
    "experiments/browser_swin2sr/surface_curriculum.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
    "experiments/browser_swin2sr/surface_run_state.py",
    "experiments/browser_swin2sr/surface_schedule.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/training_data_contract.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--source-margin-db", type=float, default=0.10)
    parser.add_argument("--mean-margin-db", type=float, default=0.10)
    return parser.parse_args()


def shared_study_inputs(paths: list[Path]) -> dict:
    studies = [json.loads(path.read_text()) for path in paths]
    if len(studies) < 2:
        raise RuntimeError("development gate requires multiple matched studies")
    for study in studies:
        if study.get("selection_source") != "development":
            raise RuntimeError("study is not development-only")
    shared_keys = (
        "bundle",
        "bundle_sha256",
        "q4_reference",
        "q4_reference_sha256",
        "parent_checkpoint_sha256",
    )
    shared = {key: studies[0][key] for key in shared_keys}
    for study in studies[1:]:
        for key, expected in shared.items():
            if study.get(key) != expected:
                raise RuntimeError(f"matched studies differ on {key}")
    return shared


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"development gate already exists: {args.out}")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    repository = Path(__file__).resolve().parents[2]
    shared = shared_study_inputs(args.study_manifest)
    bundle = Path(shared["bundle"])
    q4_reference = Path(shared["q4_reference"])
    if file_sha256(bundle) != shared["bundle_sha256"]:
        raise RuntimeError("development bundle changed")
    if file_sha256(q4_reference) != shared["q4_reference_sha256"]:
        raise RuntimeError("q4 development reference changed")
    manifest = {
        "format": "backdropkit-swin2sr-sensitive-io-development-gate-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "study_manifests": {
            str(path): file_sha256(path)
            for path in args.study_manifest
        },
        **shared,
        "source_files": {
            name: file_sha256(repository / name)
            for name in SOURCE_FILES
        },
        "expected_checkpoint_candidates": 2,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "source_margin_db": args.source_margin_db,
        "mean_margin_db": args.mean_margin_db,
        "selection_rule": {
            "overall_ground_truth_within_best_db": args.mean_margin_db,
            "every_source_mean_psnr_within_best_db": args.source_margin_db,
            "select": "smallest convolution LR multiplier among eligible arms",
            "rank": [
                "convolution_learning_rate_multiplier_ascending",
                "development_ground_truth_psnr_descending",
            ],
        },
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "study_manifests": len(args.study_manifest),
        "source_files": len(SOURCE_FILES),
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
