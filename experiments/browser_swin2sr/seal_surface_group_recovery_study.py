#!/usr/bin/env python3
"""Seal a matched recovery study for a confirmed ternary surface map."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from fork_surface_group_map_checkpoint import write_fork
from full_surface_provenance import file_sha256
from seal_pixel_objective_study import SOURCE_FILES as BASE_SOURCE_FILES
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-surface-group-recovery-study-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-surface-group-recovery-protocol-v1"
OPTIMIZER_DECISION_FORMAT = (
    "backdropkit-swin2sr-optimizer-transfer-decision-v1"
)
OBJECTIVE_DECISION_FORMAT = (
    "backdropkit-swin2sr-pixel-objective-decision-v1"
)
SURFACE_DECISION_FORMAT = (
    "backdropkit-swin2sr-surface-group-decision-v1"
)
JOINT_FORMAT = "backdropkit-swin2sr-surface-group-joint-evaluation-v1"
SOURCE_FILES = tuple(dict.fromkeys((
    *BASE_SOURCE_FILES,
    "experiments/browser_swin2sr/decide_surface_group_recovery.py",
    "experiments/browser_swin2sr/fork_surface_group_map_checkpoint.py",
    "experiments/browser_swin2sr/seal_surface_group_recovery_study.py",
)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-decision", type=Path, required=True)
    parser.add_argument("--objective-decision", type=Path, required=True)
    parser.add_argument("--objective-study", type=Path, required=True)
    parser.add_argument("--surface-decision", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps-after-parent", type=int, default=2000)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def selected_parent(
    optimizer_decision: dict,
    objective_decision: dict,
) -> dict:
    if optimizer_decision.get("optimizer_recipe_selected"):
        return {
            "selection": "optimizer_transfer",
            "checkpoint": optimizer_decision["selected_checkpoint"],
            "checkpoint_sha256": optimizer_decision[
                "selected_checkpoint_sha256"
            ],
            "selected_optimizer": optimizer_decision["selected_optimizer"],
            "selected_pixel_objective": None,
        }
    objective = objective_decision["selected_pixel_objective"]
    receipt = objective_decision["receipts"][objective]
    return {
        "selection": "pixel_objective",
        "checkpoint": receipt["checkpoint"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "selected_optimizer": None,
        "selected_pixel_objective": objective,
    }


def parent_contract(checkpoint: Path) -> tuple[dict, dict[str, int], int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if len(payload.get("quantized_modules", [])) != 204:
        raise RuntimeError("parent checkpoint does not cover 204 surfaces")
    quantization = payload["quantization"]
    training = payload["surface_training"]
    surface_step = int(payload["surface_step"])
    if surface_step < int(training["schedule_steps"]):
        raise RuntimeError("parent checkpoint is not past its hard schedule")
    surface_map = {
        name: int(size)
        for name, size in (
            quantization.get("surface_group_sizes") or {}
        ).items()
    }
    if surface_map != {
        name: int(size)
        for name, size in (
            training.get("surface_group_sizes") or {}
        ).items()
    }:
        raise RuntimeError("parent surface group-size metadata disagrees")
    optimizer = training["optimizer"]
    config = {
        "group_size": int(quantization["group_size"]),
        "quantizer_method": quantization["method"],
        "conv_quantizer_method": quantization["conv_method"],
        "conv_grouping": quantization["conv_grouping"],
        "schedule_profile": quantization["schedule_profile"],
        "core_rotation": quantization["rotation"],
        "schedule_steps": int(training["schedule_steps"]),
        "batch_size": int(training["batch_size"]),
        "crop_size": int(training["crop_size"]),
        "optimizer": optimizer["name"],
        "learning_rate": float(optimizer["learning_rate"]),
        "convolution_learning_rate_multiplier": float(
            training["convolution_learning_rate_multiplier"]
        ),
        "correction_learning_rate_multiplier": float(
            training.get("correction_learning_rate_multiplier", 0.1)
        ),
        "loss_recipe": training["loss_recipe"],
        "pixel_objective": training.get("pixel_objective", "charbonnier"),
        "ground_truth_mse_floor": float(
            training.get("ground_truth_mse_floor", 3.2e-4)
        ),
        "teacher_mse_floor": float(
            training.get("teacher_mse_floor", 2.0e-5)
        ),
        "seed": int(training["seed"]),
        "hard_shadow_weight_start": float(
            training["hard_shadow_weight_start"]
        ),
        "hard_shadow_weight_end": float(training["hard_shadow_weight_end"]),
        "hard_shadow_every": int(training["hard_shadow_every"]),
    }
    return config, surface_map, surface_step


def sealed_command(
    sealed_root: Path,
    script: str,
    *arguments: str,
) -> list[str]:
    browser = sealed_root / "experiments/browser_swin2sr"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={browser}:{sealed_root}",
        sys.executable,
        str(sealed_root / script),
        *arguments,
    ]


def training_command(
    args: argparse.Namespace,
    sealed_root: Path,
    config: dict,
    surface_map: dict[str, int],
    fork: Path,
    out: Path,
    selection_step: int,
    training_roots: list[Path],
) -> list[str]:
    command = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/train_swin2sr_full_surface.py",
        "--base-checkpoint", str(fork),
        "--validation-data", str(args.validation_data),
        "--out", str(out),
        "--steps", str(selection_step),
        "--schedule-steps", str(config["schedule_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--crop-size", str(config["crop_size"]),
        "--group-size", str(config["group_size"]),
        "--quantizer-method", config["quantizer_method"],
        "--conv-quantizer-method", config["conv_quantizer_method"],
        "--conv-grouping", config["conv_grouping"],
        "--schedule-profile", config["schedule_profile"],
        "--core-rotation", config["core_rotation"],
        "--optimizer", config["optimizer"],
        "--learning-rate", str(config["learning_rate"]),
        "--convolution-learning-rate-multiplier",
        str(config["convolution_learning_rate_multiplier"]),
        "--correction-learning-rate-multiplier",
        str(config["correction_learning_rate_multiplier"]),
        "--loss-recipe", config["loss_recipe"],
        "--pixel-objective", config["pixel_objective"],
        "--ground-truth-mse-floor", str(config["ground_truth_mse_floor"]),
        "--teacher-mse-floor", str(config["teacher_mse_floor"]),
        "--validate-every", "500",
        "--checkpoint-every", "1000",
        "--hard-shadow-weight-start",
        str(config["hard_shadow_weight_start"]),
        "--hard-shadow-weight-end", str(config["hard_shadow_weight_end"]),
        "--hard-shadow-every", str(config["hard_shadow_every"]),
        "--device", args.device,
        "--seed", str(config["seed"]),
    )
    for name, size in sorted(surface_map.items()):
        command.extend(("--surface-group-size", f"{name}={size}"))
    for root in training_roots:
        command.extend(("--data", str(root)))
    return command


def validate_inputs(
    optimizer: dict,
    objective: dict,
    surface: dict,
    joint: dict,
    surface_path: Path,
) -> None:
    if optimizer.get("format") != OPTIMIZER_DECISION_FORMAT:
        raise RuntimeError("unsupported optimizer decision")
    if objective.get("format") != OBJECTIVE_DECISION_FORMAT:
        raise RuntimeError("unsupported objective decision")
    if surface.get("format") != SURFACE_DECISION_FORMAT:
        raise RuntimeError("unsupported surface decision")
    if joint.get("format") != JOINT_FORMAT:
        raise RuntimeError("unsupported joint surface report")
    for record in (optimizer, objective, surface, joint):
        if record.get("model_promotion_authorized"):
            raise RuntimeError("input receipt improperly authorizes promotion")
        if not record.get("final_heldout_locked"):
            raise RuntimeError("input receipt did not lock final heldout")
    if not joint.get("recovery_study_eligible"):
        raise RuntimeError("joint surface map did not pass its recovery gate")
    if joint.get("decision_sha256") != file_sha256(surface_path):
        raise RuntimeError("joint report uses another surface decision")
    if (
        joint["selected_surface_group_sizes"]
        != surface["selected_surface_group_sizes"]
    ):
        raise RuntimeError("joint report changed the selected surface map")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"surface-group recovery exists: {args.out}")
    if args.steps_after_parent != 2000 or args.device != "mps":
        raise ValueError("surface-group recovery is sealed at +2000 steps on MPS")
    optimizer = json.loads(args.optimizer_decision.read_text())
    objective = json.loads(args.objective_decision.read_text())
    surface = json.loads(args.surface_decision.read_text())
    joint = json.loads(args.joint_report.read_text())
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported surface-group recovery protocol")
    if protocol.get("model_promotion_authorized"):
        raise RuntimeError("recovery protocol improperly authorizes promotion")
    if not protocol.get("final_heldout_locked"):
        raise RuntimeError("recovery protocol did not lock final heldout")
    if protocol["surface_decision_sha256"] != file_sha256(
        args.surface_decision
    ):
        raise RuntimeError("recovery protocol uses another surface decision")
    if protocol["joint_report_sha256"] != file_sha256(args.joint_report):
        raise RuntimeError("recovery protocol uses another joint report")
    if protocol["steps_after_parent"] != args.steps_after_parent:
        raise RuntimeError("recovery protocol changed continuation length")
    if protocol["device"] != args.device:
        raise RuntimeError("recovery protocol changed device")
    validate_inputs(
        optimizer,
        objective,
        surface,
        joint,
        args.surface_decision,
    )
    parent = selected_parent(optimizer, objective)
    parent_checkpoint = Path(parent["checkpoint"])
    if file_sha256(parent_checkpoint) != parent["checkpoint_sha256"]:
        raise RuntimeError("selected parent checkpoint identity changed")
    config, parent_map, parent_step = parent_contract(parent_checkpoint)
    selected_map = {
        name: int(size)
        for name, size in surface["selected_surface_group_sizes"].items()
    }
    if selected_map != protocol["selected_surface_group_sizes"]:
        raise RuntimeError("recovery protocol changed selected surface map")
    if set(selected_map) & set(parent_map):
        raise RuntimeError("selected surface map overlaps parent overrides")
    selection_step = parent_step + args.steps_after_parent
    objective_study = json.loads(args.objective_study.read_text())
    training_roots = [
        Path(record["root"])
        for record in objective_study["training_data"]
    ]
    if dataset_fingerprints(training_roots) != objective_study["training_data"]:
        raise RuntimeError("training corpus identities changed")
    bundle = Path(objective_study["bundle"])
    q4_reference = Path(objective_study["q4_reference"])
    if file_sha256(bundle) != objective_study["bundle_sha256"]:
        raise RuntimeError("development bundle changed")
    if file_sha256(q4_reference) != objective_study["q4_reference_sha256"]:
        raise RuntimeError("q4 reference changed")
    repository = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True)
    sealed_root = args.out / "sealed_sources"
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = repository / relative
        destination = sealed_root / relative
        source_hashes[relative] = file_sha256(source)
        if source_hashes[relative] != protocol["source_sha256"][relative]:
            raise RuntimeError(f"recovery source changed: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    arm_records = {}
    for arm, added_map in (("control", {}), ("refined", selected_map)):
        root = args.out / arm
        fork = root / f"fork-step{parent_step}.pt"
        receipt_path = root / "fork-receipt.json"
        receipt = write_fork(
            parent_checkpoint,
            fork,
            receipt_path,
            surface_map=added_map,
            expect_surface_step=parent_step,
        )
        full_map = dict(sorted({**parent_map, **added_map}.items()))
        training = training_command(
            args,
            sealed_root,
            config,
            full_map,
            fork,
            root / "run",
            selection_step,
            training_roots,
        )
        checkpoint = root / "run/checkpoint-final.pt"
        materialization = root / "canonical/materialization.json"
        materialize = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/materialize_surface_states.py",
            str(checkpoint),
            "--out", str(root / "canonical"),
            "--source-device", args.device,
        )
        development = root / "development/metrics.json"
        evaluate = sealed_command(
            sealed_root,
            "experiments/browser_swin2sr/evaluate_stratified.py",
            str(root / "canonical/dense-hard.pt"),
            "--bundle", str(bundle),
            "--out", str(root / "development"),
            "--crop-size", "128",
            "--device", args.device,
        )
        arm_records[arm] = {
            "added_surface_group_sizes": added_map,
            "surface_group_sizes": full_map,
            "fork_checkpoint": str(fork),
            "fork_checkpoint_sha256": receipt["fork_checkpoint_sha256"],
            "fork_receipt": str(receipt_path),
            "fork_receipt_sha256": file_sha256(receipt_path),
            "training_command": training,
            "final_checkpoint": str(checkpoint),
            "materialization_command": materialize,
            "materialization": str(materialization),
            "evaluation_command": evaluate,
            "development_report": str(development),
        }
    decision_path = args.out / "surface-group-recovery-decision.json"
    decision_command = sealed_command(
        sealed_root,
        "experiments/browser_swin2sr/decide_surface_group_recovery.py",
        "--study-manifest", str(args.out / "study-manifest.json"),
        "--control", arm_records["control"]["development_report"],
        "--candidate", arm_records["refined"]["development_report"],
        "--out", str(decision_path),
    )
    manifest = {
        "format": FORMAT,
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "optimizer_decision": str(args.optimizer_decision),
        "optimizer_decision_sha256": file_sha256(args.optimizer_decision),
        "objective_decision": str(args.objective_decision),
        "objective_decision_sha256": file_sha256(args.objective_decision),
        "surface_decision": str(args.surface_decision),
        "surface_decision_sha256": file_sha256(args.surface_decision),
        "joint_report": str(args.joint_report),
        "joint_report_sha256": file_sha256(args.joint_report),
        "selected_parent": parent,
        "parent_surface_step": parent_step,
        "selection_step": selection_step,
        "selected_surface_group_sizes": selected_map,
        "selected_scale_bytes_delta": joint["storage"][
            "selected_scale_bytes_delta"
        ],
        "config": config,
        "training_data": objective_study["training_data"],
        "bundle": str(bundle),
        "bundle_sha256": objective_study["bundle_sha256"],
        "bundle_manifest_sha256": objective_study["bundle_manifest_sha256"],
        "q4_reference": str(q4_reference),
        "q4_reference_sha256": objective_study["q4_reference_sha256"],
        "arms": arm_records,
        "decision_rule": {
            "bootstrap_seed": 20260731,
            "bootstrap_samples": 10000,
            "ground_truth_lower_95_min_db": 0.0,
            "edge_mean_min_db": -0.1,
            "teacher_mean_min_db": -0.1,
            "every_source_mean_min_db": -0.1,
            "scale_byte_budget": 65536,
        },
        "decision_command": decision_command,
        "source_sha256": source_hashes,
    }
    manifest_path = args.out / "study-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selected_parent": parent,
        "parent_surface_step": parent_step,
        "selection_step": selection_step,
        "selected_surface_group_sizes": selected_map,
    }, indent=2))


if __name__ == "__main__":
    main()
