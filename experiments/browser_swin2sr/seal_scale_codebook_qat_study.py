#!/usr/bin/env python3
"""Seal matched gradual scale-QAT arms from the selected recovery parent."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from seal_optimizer_transfer_study import SOURCE_FILES as TRAINING_SOURCE_FILES
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-scale-codebook-qat-study-v1"
RECOVERY_DECISION_FORMAT = "backdropkit-swin2sr-surface-group-recovery-decision-v1"
ARMS = {
    "uniform_q1": {
        "core_linear": 1,
        "image_io_conv": 1,
        "position_bias_linear": 1,
        "residual_body_conv": 1,
    },
    "image_io_bridge": {
        "core_linear": 1,
        "image_io_conv": 4,
        "position_bias_linear": 1,
        "residual_body_conv": 1,
    },
    "sensitive_bridge": {
        "core_linear": 1,
        "image_io_conv": 4,
        "position_bias_linear": 2,
        "residual_body_conv": 2,
    },
}
SOURCE_FILES = tuple(dict.fromkeys((
    *TRAINING_SOURCE_FILES,
    "clark_air/quant/packing.py",
    "experiments/browser_swin2sr/artifact_ledger.py",
    "experiments/browser_swin2sr/scale_codebook.py",
    "experiments/browser_swin2sr/scale_codebook_parent.py",
    "experiments/browser_swin2sr/scale_codebook_qat.py",
    "experiments/browser_swin2sr/scale_codebook_state.py",
    "experiments/browser_swin2sr/retained_precision_islands.py",
    "experiments/browser_swin2sr/retained_precision_state.py",
    "experiments/browser_swin2sr/train_scale_codebook_recovery.py",
    "experiments/browser_swin2sr/materialize_scale_codebook.py",
    "experiments/browser_swin2sr/export_browser_artifact.py",
    "experiments/browser_swin2sr/mixed_precision_artifact.py",
    "experiments/browser_swin2sr/onnx_packed_bridge.py",
    "experiments/browser_swin2sr/rewrite_onnx_ternary.py",
    "experiments/browser_swin2sr/decide_scale_codebook_qat.py",
    "experiments/browser_swin2sr/seal_scale_codebook_qat_study.py",
)))


def verify_source_closure(repository: Path) -> dict[str, list[str]]:
    """Prove every repository-local Python import is included in the seal."""
    source_paths = {relative: (repository / relative).resolve() for relative in SOURCE_FILES}
    included = set(source_paths.values())
    search_roots = (repository, repository / "experiments/browser_swin2sr")
    resolved: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for relative, source in source_paths.items():
        tree = ast.parse(source.read_text(), filename=str(source))
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                candidates = []
                for root in search_roots:
                    candidates.extend((
                        root.joinpath(*parts).with_suffix(".py"),
                        root.joinpath(*parts, "__init__.py"),
                    ))
                dependency = next((path.resolve() for path in candidates if path.exists()), None)
                if dependency is None:
                    continue
                dependency_relative = str(dependency.relative_to(repository))
                dependencies.add(dependency_relative)
                if dependency not in included:
                    missing.setdefault(relative, []).append(dependency_relative)
        resolved[relative] = sorted(dependencies)
    if missing:
        details = "; ".join(
            f"{source}: {sorted(set(dependencies))}"
            for source, dependencies in sorted(missing.items())
        )
        raise RuntimeError(f"sealed source dependency closure is incomplete: {details}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-decision", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
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


def parent_contract(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if len(payload.get("quantized_modules", [])) != 204:
        raise RuntimeError("scale-QAT parent does not cover 204 surfaces")
    if int(payload.get("surface_step", -1)) < int(
        payload["surface_training"]["schedule_steps"]
    ):
        raise RuntimeError("scale-QAT parent is not all-hard")
    return {
        "surface_step": int(payload["surface_step"]),
        "batch_size": int(payload["surface_training"]["batch_size"]),
        "crop_size": int(payload["surface_training"]["crop_size"]),
        "seed": int(payload["surface_training"]["seed"]),
        "optimizer": payload["surface_training"]["optimizer"],
        "pixel_objective": payload["surface_training"].get(
            "pixel_objective", "charbonnier"
        ),
        "surface_group_sizes": payload["quantization"].get(
            "surface_group_sizes", {}
        ),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"scale-QAT study exists: {args.out}")
    if args.steps != 4000 or args.device != "mps":
        raise ValueError("initial scale-QAT study is sealed at 4,000 MPS steps")
    if args.master_learning_rate != 1e-6 or args.center_learning_rate != 1e-5:
        raise ValueError("initial scale-QAT learning rates are frozen")
    decision = json.loads(args.recovery_decision.read_text())
    if decision.get("format") != RECOVERY_DECISION_FORMAT:
        raise RuntimeError("unsupported recovery decision")
    if decision.get("model_promotion_authorized") or not decision.get(
        "final_heldout_locked"
    ):
        raise RuntimeError("recovery decision violates promotion isolation")
    parent = Path(decision["selected_checkpoint"])
    if file_sha256(parent) != decision["selected_checkpoint_sha256"]:
        raise RuntimeError("selected scale-QAT parent identity changed")
    recovery_study = Path(decision["study_manifest"])
    if file_sha256(recovery_study) != decision["study_manifest_sha256"]:
        raise RuntimeError("recovery study identity changed")
    recovery = json.loads(recovery_study.read_text())
    bundle = Path(recovery["bundle"])
    q4_reference = Path(recovery["q4_reference"])
    if file_sha256(bundle) != recovery["bundle_sha256"]:
        raise RuntimeError("scale-QAT development bundle changed")
    if file_sha256(q4_reference) != recovery["q4_reference_sha256"]:
        raise RuntimeError("scale-QAT q4 reference changed")
    if not args.promotion_policy.exists():
        raise FileNotFoundError(args.promotion_policy)
    config = parent_contract(parent)
    repository = Path(__file__).resolve().parents[2]
    source_closure = verify_source_closure(repository)
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
    for arm, mapping in ARMS.items():
        arm_root = args.out / arm
        run = arm_root / "run"
        checkpoint = run / f"checkpoint-scale-{args.steps}.pt"
        materialization = arm_root / "materialized" / "materialization.json"
        development = arm_root / "development"
        artifact = arm_root / "artifact"
        training = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/train_scale_codebook_recovery.py",
            "--parent", str(parent),
            "--out", str(run),
            "--validation-data", str(args.validation_data),
            "--steps", str(args.steps),
            "--master-learning-rate", str(args.master_learning_rate),
            "--center-learning-rate", str(args.center_learning_rate),
            "--validate-every", "500",
            "--checkpoint-every", "500",
            "--hard-shadow-every", "2",
            "--hard-shadow-weight-start", "0.05",
            "--hard-shadow-weight-end", "0.25",
            "--device", args.device,
        )
        for category, bits in sorted(mapping.items()):
            training.extend(("--target-bits", f"{category}={bits}"))
        for root in args.data:
            training.extend(("--data", str(root)))
        arms[arm] = {
            "bits_by_category": mapping,
            "training_command": training,
            "final_checkpoint": str(checkpoint),
            "materialization_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/materialize_scale_codebook.py",
                str(checkpoint),
                "--out", str(arm_root / "materialized"),
                "--source-device", "cpu",
            ),
            "materialization": str(materialization),
            "evaluation_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/evaluate_stratified.py",
                str(arm_root / "materialized" / "dense-hard.pt"),
                "--bundle", str(bundle),
                "--out", str(development),
                "--device", args.device,
            ),
            "development_report": str(development / "metrics.json"),
            "artifact_command": sealed_command(
                sealed_root,
                "experiments/browser_swin2sr/export_browser_artifact.py",
                str(materialization),
                "--out", str(artifact),
            ),
            "artifact": str(artifact),
        }
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_scale_qat_results",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "recovery_decision": str(args.recovery_decision),
        "recovery_decision_sha256": file_sha256(args.recovery_decision),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "parent_contract": config,
        "bundle": str(bundle),
        "bundle_sha256": file_sha256(bundle),
        "q4_reference": str(q4_reference),
        "q4_reference_sha256": file_sha256(q4_reference),
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "training_data": fingerprints,
        "validation_data": str(args.validation_data),
        "steps": args.steps,
        "master_learning_rate": args.master_learning_rate,
        "center_learning_rate": args.center_learning_rate,
        "device": args.device,
        "arms": arms,
        "size_rule": {
            "quality_model_declared_bytes": 5_680_000,
            "minimum_reduction_fraction": 0.85,
            "uniform_q1_max_packed_trunk_bits": 1.66,
        },
        "source_sha256": source_sha256,
        "source_closure": {
            "verified": True,
            "resolved_local_imports": source_closure,
        },
    }
    decision_path = args.out / "scale-codebook-qat-decision.json"
    manifest["decision_command"] = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/decide_scale_codebook_qat.py",
        "--study-manifest", str(args.out / "study-manifest.json"),
        "--out", str(decision_path),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "study-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(path),
        "out_sha256": file_sha256(path),
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "arms": sorted(arms),
    }, indent=2))


if __name__ == "__main__":
    main()
