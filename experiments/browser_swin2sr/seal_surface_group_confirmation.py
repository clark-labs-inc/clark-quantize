#!/usr/bin/env python3
"""Seal a larger paired confirmation of ternary scale-group candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-surface-group-confirmation-v1"
SCREEN_FORMAT = "backdropkit-swin2sr-surface-group-sensitivity-v1"
GROUP_SIZES = (2, 3, 6)
SOURCE_FILES = (
    "experiments/browser_swin2sr/analyze_surface_group_sensitivity.py",
    "experiments/browser_swin2sr/analyze_surface_restore_sensitivity.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/probe_sensitive_io_group_sizes.py",
    "experiments/browser_swin2sr/surface_artifact_state.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale-byte-budget", type=int, default=65536)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def select_candidates(
    surfaces: list[dict],
    scale_byte_budget: int,
) -> tuple[list[dict], int]:
    eligible = []
    for surface in surfaces:
        delta = surface["delta_vs_hard"]
        scale_bytes = int(surface["storage"]["scale_bytes_delta"])
        if (
            delta["psnr_vs_ground_truth"] > 0
            and delta["edge_psnr_vs_ground_truth"] > 0
            and delta["psnr_vs_teacher"] >= -0.1
            and scale_bytes > 0
        ):
            score = delta["psnr_vs_ground_truth"] / (scale_bytes / 1024)
            eligible.append({
                "name": surface["name"],
                "category": surface["category"],
                "screen_delta_vs_hard": delta,
                "group2_scale_bytes_delta": scale_bytes,
                "screen_ground_truth_db_per_added_kib": score,
            })
    eligible.sort(key=lambda item: (
        -item["screen_ground_truth_db_per_added_kib"],
        item["name"],
    ))
    selected = []
    consumed = 0
    for candidate in eligible:
        scale_bytes = candidate["group2_scale_bytes_delta"]
        if consumed + scale_bytes <= scale_byte_budget:
            selected.append(candidate)
            consumed += scale_bytes
    return selected, consumed


def confirmation_command(
    repository: Path,
    screen: dict,
    out: Path,
    candidates: list[dict],
    group_size: int,
    samples: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(repository / SOURCE_FILES[0]),
        screen["checkpoint"],
        "--bundle", screen["bundle"],
        "--out", str(out / f"group{group_size}-confirmation-{samples}.json"),
        "--target-group-size", str(group_size),
        "--method", "current",
        "--max-samples", str(samples),
        "--bootstrap-resamples", str(bootstrap_resamples),
        "--bootstrap-seed", str(bootstrap_seed),
        "--device", "cpu",
        "--torch-threads", "1",
    ]
    for candidate in candidates:
        command.extend(("--surface", candidate["name"]))
    return command


def validate_screen(screen: dict, screen_path: Path) -> None:
    if screen.get("format") != SCREEN_FORMAT:
        raise RuntimeError("selection screen has an unsupported format")
    if screen.get("sample_count") != 8:
        raise RuntimeError("selection screen is not the frozen 8-sample screen")
    if screen.get("target_group_size") != 2:
        raise RuntimeError("selection screen did not test group size 2")
    if screen.get("candidate_method") != "current":
        raise RuntimeError("selection screen changed quantizer method")
    if screen.get("surface_step") != 24000:
        raise RuntimeError("selection screen is not the selected step 24000")
    if screen.get("model_promotion_authorized"):
        raise RuntimeError("diagnostic screen improperly authorizes promotion")
    if not screen.get("final_heldout_locked"):
        raise RuntimeError("diagnostic screen did not lock final heldout")
    checkpoint = Path(screen["checkpoint"])
    bundle = Path(screen["bundle"])
    if file_sha256(checkpoint) != screen["checkpoint_sha256"]:
        raise RuntimeError("selection checkpoint identity changed")
    if file_sha256(bundle) != screen["bundle_sha256"]:
        raise RuntimeError("development bundle identity changed")
    if not screen_path.is_file():
        raise RuntimeError("selection screen is missing")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"confirmation study exists: {args.out}")
    if min(
        args.scale_byte_budget,
        args.samples,
        args.bootstrap_resamples,
    ) < 1:
        raise ValueError("budget, samples, and bootstrap resamples must be positive")
    repository = Path(__file__).resolve().parents[2]
    screen = json.loads(args.screen.read_text())
    validate_screen(screen, args.screen)
    candidates, consumed = select_candidates(
        screen["surfaces"],
        args.scale_byte_budget,
    )
    if not candidates:
        raise RuntimeError("selection screen produced no confirmation candidates")
    source_hashes = {
        relative: file_sha256(repository / relative)
        for relative in SOURCE_FILES
    }
    commands = {
        str(group_size): confirmation_command(
            repository,
            screen,
            args.out,
            candidates,
            group_size,
            args.samples,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        for group_size in GROUP_SIZES
    }
    manifest = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "development_screen",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "selection_screen": str(args.screen),
        "selection_screen_sha256": file_sha256(args.screen),
        "checkpoint": screen["checkpoint"],
        "checkpoint_sha256": screen["checkpoint_sha256"],
        "bundle": screen["bundle"],
        "bundle_sha256": screen["bundle_sha256"],
        "bundle_manifest_sha256": screen["bundle_manifest_sha256"],
        "selection_rule": {
            "screen_samples": 8,
            "required_ground_truth_delta_db": "> 0",
            "required_edge_delta_db": "> 0",
            "required_teacher_delta_db": ">= -0.1",
            "ranking": "ground_truth_delta_db_per_added_kib_descending",
            "scale_byte_budget_at_group2": args.scale_byte_budget,
            "greedy_tie_break": "surface_name_ascending",
        },
        "candidate_count": len(candidates),
        "selected_group2_scale_bytes_delta": consumed,
        "candidates": candidates,
        "confirmation": {
            "samples": args.samples,
            "group_sizes": list(GROUP_SIZES),
            "bootstrap": {
                "method": "paired_nonparametric_mean",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence": 0.95,
            },
            "individual_pass_gate": {
                "ground_truth_lower_95_db": "> 0",
                "edge_mean_delta_db": ">= -0.02",
                "teacher_mean_delta_db": ">= -0.1",
            },
            "group_choice": (
                "highest confirmed ground-truth dB per added KiB per surface"
            ),
            "joint_candidate_gate": (
                "re-evaluate the selected surface map jointly under the same "
                "64 KiB budget before any recovery training"
            ),
        },
        "commands": commands,
        "source_sha256": source_hashes,
    }
    args.out.mkdir(parents=True)
    manifest_path = args.out / "study-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate_count": len(candidates),
        "selected_group2_scale_bytes_delta": consumed,
        "commands": commands,
    }, indent=2))


if __name__ == "__main__":
    main()
