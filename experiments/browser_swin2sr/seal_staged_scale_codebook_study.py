#!/usr/bin/env python3
"""Seal matched q1 collapse arms from the best completed scale-QAT bridge."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from decide_scale_codebook_qat import FORMAT as SCALE_DECISION_FORMAT
from full_surface_provenance import file_sha256
from seal_scale_codebook_qat_study import (
    FORMAT as SCALE_STUDY_FORMAT,
    SOURCE_FILES as SCALE_SOURCE_FILES,
)
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-staged-scale-codebook-study-v1"
FINAL_BITS = {
    "core_linear": 1,
    "image_io_conv": 1,
    "position_bias_linear": 1,
    "residual_body_conv": 1,
}
ARMS = {
    "late_io": {
        "schedule_profile": "late_io_v1",
        "hard_shadow_weight_start": 0.05,
        "hard_shadow_weight_end": 0.25,
    },
    "gentle_io": {
        "schedule_profile": "gentle_io_v1",
        "hard_shadow_weight_start": 0.05,
        "hard_shadow_weight_end": 0.25,
    },
    "gentle_io_strong_shadow": {
        "schedule_profile": "gentle_io_v1",
        "hard_shadow_weight_start": 0.15,
        "hard_shadow_weight_end": 0.40,
    },
}
SOURCE_FILES = tuple(dict.fromkeys((
    *SCALE_SOURCE_FILES,
    "experiments/browser_swin2sr/staged_scale_codebook.py",
    "experiments/browser_swin2sr/staged_scale_codebook_state.py",
    "experiments/browser_swin2sr/train_staged_scale_codebook_recovery.py",
    "experiments/browser_swin2sr/materialize_staged_scale_codebook.py",
    "experiments/browser_swin2sr/seal_staged_scale_codebook_study.py",
    "experiments/browser_swin2sr/decide_staged_scale_codebook.py",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-decision", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--master-learning-rate", type=float, default=1e-6)
    parser.add_argument("--center-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def sealed_command(sealed_root: Path, script: str, *arguments: str) -> list[str]:
    browser = sealed_root / "experiments/browser_swin2sr"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={browser}:{sealed_root}",
        sys.executable,
        str(sealed_root / script),
        *arguments,
    ]


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"staged scale-QAT study exists: {args.out}")
    if args.steps != 4000 or args.device != "mps":
        raise ValueError("initial staged scale study is sealed at 4,000 MPS steps")
    if args.master_learning_rate != 1e-6 or args.center_learning_rate != 1e-5:
        raise ValueError("initial staged scale learning rates are frozen")
    decision = json.loads(args.scale_decision.read_text())
    if decision.get("format") != SCALE_DECISION_FORMAT:
        raise RuntimeError("unsupported scale-QAT decision")
    if decision.get("approximately_1_58_endpoint_eligible"):
        raise RuntimeError("scale-QAT already produced an eligible q1 endpoint")
    if decision.get("model_promotion_authorized") or not decision.get(
        "final_heldout_locked"
    ):
        raise RuntimeError("scale-QAT decision violates promotion isolation")
    bridge_arm = decision.get("next_stage_parent_arm")
    if bridge_arm not in decision["candidates"]:
        raise RuntimeError("scale-QAT decision has no valid next-stage bridge")
    scale_study_path = Path(decision["study_manifest"])
    if file_sha256(scale_study_path) != decision["study_manifest_sha256"]:
        raise RuntimeError("scale-QAT study identity changed")
    scale_study = json.loads(scale_study_path.read_text())
    if scale_study.get("format") != SCALE_STUDY_FORMAT:
        raise RuntimeError("unsupported parent scale-QAT study")
    candidate = decision["candidates"][bridge_arm]
    bridge = Path(candidate["checkpoint"])
    if file_sha256(bridge) != candidate["checkpoint_sha256"]:
        raise RuntimeError("selected scale bridge identity changed")
    for key in ("bundle", "q4_reference", "promotion_policy"):
        if file_sha256(Path(scale_study[key])) != scale_study[f"{key}_sha256"]:
            raise RuntimeError(f"parent scale-QAT input changed: {key}")
    repository = Path(__file__).resolve().parents[2]
    fingerprints = dataset_fingerprints(args.data)
    sealed_root = args.out / "sealed_sources"
    for relative in SOURCE_FILES:
        source = repository / relative
        destination = sealed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_sha256 = {
        relative: file_sha256(sealed_root / relative) for relative in SOURCE_FILES
    }
    arms = {}
    for arm, controls in ARMS.items():
        arm_root = args.out / arm
        run = arm_root / "run"
        checkpoint = run / f"checkpoint-collapse-{args.steps}.pt"
        materialization = arm_root / "materialized" / "materialization.json"
        training = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/train_staged_scale_codebook_recovery.py",
            "--bridge", str(bridge),
            "--out", str(run),
            "--validation-data", str(args.validation_data),
            "--steps", str(args.steps),
            "--master-learning-rate", str(args.master_learning_rate),
            "--center-learning-rate", str(args.center_learning_rate),
            "--validate-every", "500",
            "--checkpoint-every", "500",
            "--hard-shadow-every", "2",
            "--hard-shadow-weight-start", str(controls["hard_shadow_weight_start"]),
            "--hard-shadow-weight-end", str(controls["hard_shadow_weight_end"]),
            "--schedule-profile", controls["schedule_profile"],
            "--device", args.device,
        )
        for category, bits in sorted(FINAL_BITS.items()):
            training.extend(("--target-bits", f"{category}={bits}"))
        for root in args.data:
            training.extend(("--data", str(root)))
        arms[arm] = {
            **controls,
            "bits_by_category": FINAL_BITS,
            "training_command": training,
            "final_checkpoint": str(checkpoint),
            "materialization_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/materialize_staged_scale_codebook.py",
                str(checkpoint),
                "--out", str(arm_root / "materialized"),
                "--source-device", "cpu",
            ),
            "materialization": str(materialization),
            "evaluation_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/evaluate_stratified.py",
                str(arm_root / "materialized" / "dense-hard.pt"),
                "--bundle", scale_study["bundle"],
                "--out", str(arm_root / "development"),
                "--device", args.device,
            ),
            "development_report": str(arm_root / "development" / "metrics.json"),
            "artifact_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/export_browser_artifact.py",
                str(materialization),
                "--out", str(arm_root / "artifact"),
            ),
            "artifact": str(arm_root / "artifact"),
        }
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_staged_scale_results",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "scale_decision": str(args.scale_decision),
        "scale_decision_sha256": file_sha256(args.scale_decision),
        "parent_scale_study": str(scale_study_path),
        "parent_scale_study_sha256": file_sha256(scale_study_path),
        "bridge_arm": bridge_arm,
        "bridge_checkpoint": str(bridge),
        "bridge_checkpoint_sha256": file_sha256(bridge),
        "bundle": scale_study["bundle"],
        "bundle_sha256": scale_study["bundle_sha256"],
        "q4_reference": scale_study["q4_reference"],
        "q4_reference_sha256": scale_study["q4_reference_sha256"],
        "promotion_policy": scale_study["promotion_policy"],
        "promotion_policy_sha256": scale_study["promotion_policy_sha256"],
        "training_data": fingerprints,
        "validation_data": str(args.validation_data),
        "steps": args.steps,
        "master_learning_rate": args.master_learning_rate,
        "center_learning_rate": args.center_learning_rate,
        "device": args.device,
        "arms": arms,
        "size_rule": scale_study["size_rule"],
        "source_sha256": source_sha256,
    }
    decision_path = args.out / "staged-scale-codebook-decision.json"
    manifest["decision_command"] = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/decide_staged_scale_codebook.py",
        "--study-manifest", str(args.out / "study-manifest.json"),
        "--out", str(decision_path),
    )
    args.out.mkdir(parents=True)
    path = args.out / "study-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(path),
        "out_sha256": file_sha256(path),
        "bridge_arm": bridge_arm,
        "bridge_checkpoint_sha256": manifest["bridge_checkpoint_sha256"],
        "arms": sorted(arms),
    }, indent=2))


if __name__ == "__main__":
    main()
