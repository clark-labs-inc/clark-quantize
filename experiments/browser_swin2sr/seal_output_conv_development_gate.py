#!/usr/bin/env python3
"""Seal the matched output-convolution development gate before training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-output-conv-development-gate-v1"
TARGET_SURFACE = "upsample.conv"
EXPECTED_GROUPS = {2, 3, 9}
SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/decide_output_conv_development.py",
    "experiments/browser_swin2sr/decide_sensitive_io_development.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/evaluate_hard_started_stratified.py",
    "experiments/browser_swin2sr/evaluate_output_conv_stratified.py",
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
    "experiments/browser_swin2sr/seal_output_conv_development_gate.py",
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
    parser.add_argument("--selection-step", type=int, default=24_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--source-margin-db", type=float, default=0.10)
    parser.add_argument("--mean-margin-db", type=float, default=0.10)
    return parser.parse_args()


def command_value(command: list[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise RuntimeError(f"training command does not seal one {flag}")
    return command[indices[0] + 1]


def study_arm(study: dict) -> tuple[str, list[str]]:
    arms = study.get("arms") or {}
    if len(arms) != 1:
        raise RuntimeError("output study must contain exactly one arm")
    name, payload = next(iter(arms.items()))
    command = payload.get("training_command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise RuntimeError("output study has no canonical training command")
    return name, command


def matched_output_studies(
    paths: list[Path],
    *,
    selection_step: int,
) -> dict:
    if len(paths) != len(EXPECTED_GROUPS):
        raise RuntimeError("output gate requires exactly g9, g3, and g2 studies")
    studies = [(path, json.loads(path.read_text())) for path in paths]
    shared_keys = (
        "bundle",
        "bundle_sha256",
        "q4_reference",
        "q4_reference_sha256",
        "parent_checkpoint_sha256",
        "training_data",
        "training_data_receipts",
        "training_state_sha256",
    )
    first = studies[0][1]
    shared = {key: first[key] for key in shared_keys}
    non_output_map = None
    by_group = {}
    for path, study in studies:
        if study.get("selection_source") != "development":
            raise RuntimeError("output study is not development-only")
        if not study.get("final_heldout_locked"):
            raise RuntimeError("output study does not lock final heldout")
        for key, expected in shared.items():
            if study.get(key) != expected:
                raise RuntimeError(f"matched output studies differ on {key}")
        group_sizes = {
            name: int(size)
            for name, size in (study.get("surface_group_sizes") or {}).items()
        }
        if TARGET_SURFACE not in group_sizes:
            raise RuntimeError("output study does not name its target group")
        group = group_sizes.pop(TARGET_SURFACE)
        if group not in EXPECTED_GROUPS or group in by_group:
            raise RuntimeError(f"unexpected or duplicate output group: {group}")
        if non_output_map is None:
            non_output_map = group_sizes
        elif group_sizes != non_output_map:
            raise RuntimeError("output studies differ outside target surface")
        _, command = study_arm(study)
        if int(command_value(command, "--steps")) != selection_step:
            raise RuntimeError("output study ends at a different selection step")
        if int(command_value(command, "--schedule-steps")) != selection_step:
            raise RuntimeError("output study uses a different hard schedule")
        by_group[group] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    if set(by_group) != EXPECTED_GROUPS:
        raise RuntimeError("output studies do not cover g9, g3, and g2")
    return {
        **shared,
        "non_output_surface_group_sizes": non_output_map,
        "study_manifests_by_output_group": {
            str(group): by_group[group]
            for group in sorted(by_group)
        },
    }


def build_manifest(
    paths: list[Path],
    *,
    repository: Path,
    selection_step: int,
    bootstrap_samples: int,
    seed: int,
    source_margin_db: float,
    mean_margin_db: float,
) -> dict:
    if selection_step < 1:
        raise ValueError("selection step must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if source_margin_db < 0 or mean_margin_db < 0:
        raise ValueError("selection margins must be nonnegative")
    matched = matched_output_studies(
        paths,
        selection_step=selection_step,
    )
    bundle = Path(matched["bundle"])
    q4_reference = Path(matched["q4_reference"])
    if file_sha256(bundle) != matched["bundle_sha256"]:
        raise RuntimeError("output development bundle changed")
    if file_sha256(q4_reference) != matched["q4_reference_sha256"]:
        raise RuntimeError("output q4 development reference changed")
    return {
        "format": FORMAT,
        "created_at_unix": time.time(),
        "selection_source": "development",
        "study_manifests": {
            str(path): file_sha256(path)
            for path in paths
        },
        **matched,
        "source_files": {
            name: file_sha256(repository / name)
            for name in SOURCE_FILES
        },
        "control_group_size": 9,
        "candidate_group_sizes": [2, 3],
        "selection_step": selection_step,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "source_margin_db": source_margin_db,
        "mean_margin_db": mean_margin_db,
        "selection_rule": {
            "must_beat_control": [
                "overall_ground_truth_psnr",
                "overall_teacher_psnr",
            ],
            "overall_ground_truth_within_best_db": mean_margin_db,
            "every_source_mean_psnr_within_best_db": source_margin_db,
            "select": "largest eligible output group size",
        },
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"output gate already exists: {args.out}")
    repository = Path(__file__).resolve().parents[2]
    manifest = build_manifest(
        args.study_manifest,
        repository=repository,
        selection_step=args.selection_step,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        source_margin_db=args.source_margin_db,
        mean_margin_db=args.mean_margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "study_manifests": len(args.study_manifest),
        "source_files": len(SOURCE_FILES),
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
