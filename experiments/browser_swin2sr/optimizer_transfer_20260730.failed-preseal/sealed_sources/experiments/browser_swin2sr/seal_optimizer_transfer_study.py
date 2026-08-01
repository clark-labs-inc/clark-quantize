#!/usr/bin/env python3
"""Seal QAT8-continuation versus fresh-teacher AdamW before training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from training_data_contract import dataset_fingerprints


INPUT_SURFACE = "swin2sr.first_convolution"
OUTPUT_SURFACE = "upsample.conv"
SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/decide_optimizer_transfer.py",
    "experiments/browser_swin2sr/decide_sensitive_io_development.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/evaluate_hard_started_stratified.py",
    "experiments/browser_swin2sr/evaluate_optimizer_transfer_stratified.py",
    "experiments/browser_swin2sr/evaluate_promotion.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/full_surface_cli.py",
    "experiments/browser_swin2sr/full_surface_optimizer.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/island_policy.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/recovery_controls.py",
    "experiments/browser_swin2sr/recovery_corpus_identity.py",
    "experiments/browser_swin2sr/recovery_data.py",
    "experiments/browser_swin2sr/recovery_resume_policy.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/seal_optimizer_transfer_study.py",
    "experiments/browser_swin2sr/surface_artifact_state.py",
    "experiments/browser_swin2sr/surface_curriculum.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
    "experiments/browser_swin2sr/surface_run_state.py",
    "experiments/browser_swin2sr/surface_schedule.py",
    "experiments/browser_swin2sr/train_swin2sr_full_surface.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/training_data_contract.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-decision", type=Path, required=True)
    parser.add_argument("--output-decision", type=Path, required=True)
    parser.add_argument("--output-study-manifest", type=Path, required=True)
    parser.add_argument("--qat8-checkpoint", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def selected_output_candidate(decision: dict) -> dict:
    selected_group = decision.get("selected_output_group_size")
    matching = [
        candidate
        for candidate in decision.get("candidates", [])
        if candidate.get("output_group_size") == selected_group
    ]
    if len(matching) != 1:
        raise RuntimeError("output decision has no unique selected candidate")
    return matching[0]


def verify_selected_recipe(
    input_decision: dict,
    output_decision: dict,
    checkpoint: Path,
) -> dict[str, int]:
    if (
        input_decision.get("format")
        != "backdropkit-swin2sr-sensitive-io-development-decision-v1"
        or not input_decision.get("sensitive_recipe_selected")
    ):
        raise RuntimeError("optimizer study has no selected input recipe")
    if (
        output_decision.get("format")
        != "backdropkit-swin2sr-output-conv-decision-v1"
        or not output_decision.get("output_recipe_selected")
    ):
        raise RuntimeError("optimizer study has no selected output recipe")
    for name, decision in (
        ("input", input_decision),
        ("output", output_decision),
    ):
        if decision.get("model_promotion_authorized"):
            raise RuntimeError(f"{name} decision improperly authorizes promotion")
        if not decision.get("final_heldout_locked"):
            raise RuntimeError(f"{name} decision does not lock final heldout")
    if (
        output_decision.get("parent_checkpoint_sha256")
        != input_decision.get("selected_checkpoint_sha256")
    ):
        raise RuntimeError("output study does not descend from selected input")
    if file_sha256(checkpoint) != output_decision["selected_checkpoint_sha256"]:
        raise RuntimeError("QAT8 checkpoint is not the selected output state")
    shared = {
        name: int(size)
        for name, size in (
            output_decision.get("shared_non_output_surface_map") or {}
        ).items()
    }
    if INPUT_SURFACE not in shared:
        raise RuntimeError("selected recipe has no input surface override")
    output_group = int(output_decision["selected_output_group_size"])
    if output_group not in (2, 3):
        raise RuntimeError("selected output group is not preregistered")
    return {
        **shared,
        OUTPUT_SURFACE: output_group,
    }


def checkpoint_contract(
    checkpoint: Path,
    surface_map: dict[str, int],
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if int(payload.get("surface_step", -1)) != 24000:
        raise RuntimeError("optimizer transfer parent is not at step 24000")
    if len(payload.get("quantized_modules", [])) != 204:
        raise RuntimeError("optimizer transfer parent does not cover 204 surfaces")
    quantization = payload["quantization"]
    training = payload["surface_training"]
    actual_map = {
        name: int(size)
        for name, size in (
            quantization.get("surface_group_sizes") or {}
        ).items()
    }
    if actual_map != surface_map:
        raise RuntimeError("optimizer parent surface map differs from decisions")
    if int(training["schedule_steps"]) != 24000:
        raise RuntimeError("optimizer parent uses a different hard schedule")
    optimizer = training.get("optimizer") or {}
    if optimizer.get("name") != "bonsai_qat8_muon":
        raise RuntimeError("continuation parent is not QAT8 plus periodic Muon")
    return {
        "group_size": int(quantization["group_size"]),
        "quantizer_method": quantization["method"],
        "conv_quantizer_method": quantization["conv_method"],
        "conv_grouping": quantization["conv_grouping"],
        "schedule_profile": quantization["schedule_profile"],
        "core_rotation": quantization["rotation"],
        "learning_rate": float(optimizer["learning_rate"]),
        "convolution_learning_rate_multiplier": float(
            training["convolution_learning_rate_multiplier"]
        ),
        "correction_learning_rate_multiplier": float(
            training.get("correction_learning_rate_multiplier", 0.10)
        ),
        "loss_recipe": training["loss_recipe"],
        "batch_size": int(training["batch_size"]),
        "crop_size": int(training["crop_size"]),
        "seed": int(training["seed"]),
        "hard_shadow_weight_start": float(
            training["hard_shadow_weight_start"]
        ),
        "hard_shadow_weight_end": float(training["hard_shadow_weight_end"]),
        "hard_shadow_every": int(training["hard_shadow_every"]),
    }


def development_quality_screen(
    policy: dict,
    q4: dict,
) -> dict:
    if (
        policy.get("format")
        != "backdropkit-swin2sr-promotion-policy-v1"
    ):
        raise RuntimeError("unsupported promotion policy")
    reference = policy.get("reference") or {}
    checks = {
        "model": q4.get("model"),
        "revision": q4.get("model_revision"),
        "model_sha256": q4.get("state_sha256"),
        "artifact_bytes": q4.get("artifact_bytes"),
        "samples": len(q4.get("samples", [])),
    }
    for key, actual in checks.items():
        if reference.get(key) != actual:
            raise RuntimeError(
                f"promotion policy q4 {key} differs: "
                f"{reference.get(key)!r} != {actual!r}"
            )
    paired = policy.get("paired_quality_gates")
    stratified = policy.get("stratified_gates")
    if not isinstance(paired, dict) or not paired:
        raise RuntimeError("promotion policy has no paired quality gates")
    if not isinstance(stratified, list) or not stratified:
        raise RuntimeError("promotion policy has no stratified quality gates")
    return {
        "seed": int(policy["seed"]),
        "bootstrap_samples": int(policy["bootstrap_samples"]),
        "paired_quality_gates": paired,
        "stratified_gates": stratified,
    }


def common_command(
    args: argparse.Namespace,
    destination: Path,
    config: dict,
    surface_map: dict[str, int],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_swin2sr_full_surface.py")),
        "--validation-data", str(args.validation_data),
        "--out", str(destination),
        "--steps", str(args.steps),
        "--schedule-steps", "24000",
        "--batch-size", str(config["batch_size"]),
        "--crop-size", str(config["crop_size"]),
        "--group-size", str(config["group_size"]),
        "--quantizer-method", config["quantizer_method"],
        "--conv-quantizer-method", config["conv_quantizer_method"],
        "--conv-grouping", config["conv_grouping"],
        "--schedule-profile", config["schedule_profile"],
        "--core-rotation", config["core_rotation"],
        "--learning-rate", str(config["learning_rate"]),
        "--convolution-learning-rate-multiplier",
        str(config["convolution_learning_rate_multiplier"]),
        "--correction-learning-rate-multiplier",
        str(config["correction_learning_rate_multiplier"]),
        "--loss-recipe", config["loss_recipe"],
        "--validate-every", "500",
        "--checkpoint-every", "1000",
        "--hard-shadow-weight-start",
        str(config["hard_shadow_weight_start"]),
        "--hard-shadow-weight-end", str(config["hard_shadow_weight_end"]),
        "--hard-shadow-every", str(config["hard_shadow_every"]),
        "--device", args.device,
        "--seed", str(config["seed"]),
    ]
    for name, size in sorted(surface_map.items()):
        command.extend(("--surface-group-size", f"{name}={size}"))
    for root in args.data:
        command.extend(("--data", str(root)))
    return command


def training_commands(
    args: argparse.Namespace,
    config: dict,
    surface_map: dict[str, int],
) -> dict[str, list[str]]:
    qat8 = common_command(
        args,
        args.out / "qat8" / "run",
        config,
        surface_map,
    )
    qat8.extend((
        "--optimizer",
        "bonsai_qat8_muon",
        "--base-checkpoint",
        str(args.qat8_checkpoint),
    ))
    adamw = common_command(
        args,
        args.out / "adamw" / "run",
        config,
        surface_map,
    )
    adamw.extend(("--optimizer", "adamw", "--initialization", "fp32"))
    return {"qat8": qat8, "adamw": adamw}


def materialization_command(
    args: argparse.Namespace,
    arm: str,
) -> list[str]:
    destination = args.out / arm
    return [
        sys.executable,
        str(Path(__file__).with_name("materialize_surface_states.py")),
        str(destination / "run" / "checkpoint-final.pt"),
        "--out",
        str(destination / "canonical"),
        "--source-device",
        args.device,
    ]


def evaluation_command(
    args: argparse.Namespace,
    arm: str,
    study_manifest: Path,
    bundle: Path,
    q4_reference: Path,
) -> list[str]:
    destination = args.out / arm
    return [
        sys.executable,
        str(
            Path(__file__).with_name(
                "evaluate_optimizer_transfer_stratified.py"
            )
        ),
        str(destination / "run" / "checkpoint-final.pt"),
        "--materialization",
        str(destination / "canonical" / "materialization.json"),
        "--study-manifest",
        str(study_manifest),
        "--bundle",
        str(bundle),
        "--q4-reference",
        str(q4_reference),
        "--out",
        str(destination / "development"),
        "--crop-size",
        "128",
        "--device",
        args.device,
    ]


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"optimizer study already exists: {args.out}")
    if args.steps != 30000 or args.device != "mps":
        raise ValueError("optimizer transfer is sealed at 30000 steps on MPS")
    repository = Path(__file__).resolve().parents[2]
    input_decision = json.loads(args.input_decision.read_text())
    output_decision = json.loads(args.output_decision.read_text())
    output_study = json.loads(args.output_study_manifest.read_text())
    surface_map = verify_selected_recipe(
        input_decision,
        output_decision,
        args.qat8_checkpoint,
    )
    selected_candidate = selected_output_candidate(output_decision)
    selected_report = json.loads(Path(selected_candidate["report"]).read_text())
    if (
        selected_report["study_manifest_sha256"]
        != file_sha256(args.output_study_manifest)
    ):
        raise RuntimeError("selected output report uses another study manifest")
    if output_study.get("selection_source") != "development":
        raise RuntimeError("output parent study is not development-only")
    datasets = dataset_fingerprints(args.data)
    if datasets != output_study["training_data"]:
        raise RuntimeError("optimizer-transfer training data drifted")
    config = checkpoint_contract(args.qat8_checkpoint, surface_map)
    commands = training_commands(args, config, surface_map)
    study_manifest = args.out / "study-manifest.json"
    bundle = Path(output_study["bundle"])
    q4_reference = Path(output_study["q4_reference"])
    q4 = json.loads(q4_reference.read_text())
    promotion_policy = json.loads(args.promotion_policy.read_text())
    quality_screen = development_quality_screen(promotion_policy, q4)
    materialization_commands = {
        arm: materialization_command(args, arm)
        for arm in commands
    }
    evaluation_commands = {
        arm: evaluation_command(
            args,
            arm,
            study_manifest,
            bundle,
            q4_reference,
        )
        for arm in commands
    }
    sealed_root = args.out / "sealed_sources"
    source_hashes = {}
    for name in SOURCE_FILES:
        source = repository / name
        source_hashes[name] = file_sha256(source)
        destination = sealed_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = {
        "format": "backdropkit-swin2sr-optimizer-transfer-v1",
        "created_at_unix": time.time(),
        "selection_source": "development",
        "input_decision": str(args.input_decision),
        "input_decision_sha256": file_sha256(args.input_decision),
        "output_decision": str(args.output_decision),
        "output_decision_sha256": file_sha256(args.output_decision),
        "output_study_manifest": str(args.output_study_manifest),
        "output_study_manifest_sha256": file_sha256(
            args.output_study_manifest
        ),
        "qat8_checkpoint": str(args.qat8_checkpoint),
        "qat8_checkpoint_sha256": file_sha256(args.qat8_checkpoint),
        "surface_group_sizes": surface_map,
        "config": config,
        "training_data": datasets,
        "training_data_receipts": output_study["training_data_receipts"],
        "bundle": output_study["bundle"],
        "bundle_sha256": output_study["bundle_sha256"],
        "q4_reference": output_study["q4_reference"],
        "q4_reference_sha256": output_study["q4_reference_sha256"],
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "development_quality_screen": quality_screen,
        "source_files": source_hashes,
        "sealed_source_root": str(sealed_root),
        "arms": {
            "qat8": {
                "initialization": "selected QAT8 continuation",
                "training_command": commands["qat8"],
                "materialization_command": materialization_commands["qat8"],
                "evaluation_command": evaluation_commands["qat8"],
            },
            "adamw": {
                "initialization": "fresh fp32 teacher",
                "training_command": commands["adamw"],
                "materialization_command": materialization_commands["adamw"],
                "evaluation_command": evaluation_commands["adamw"],
            },
        },
        "materialization_identity_policy": (
            "Project each continuous-master checkpoint exactly once on MPS. "
            "Rank its hashed dense-hard state and reuse the paired runtime "
            "state for every later backend artifact."
        ),
        "selection_step": 30000,
        "ranking_policy": [
            "worst_source_psnr_delta",
            "overall_psnr_delta",
            "edge_psnr_delta",
            "teacher_psnr_delta",
            "optimizer_arm_stable_tiebreak",
        ],
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    args.out.mkdir(parents=True)
    (args.out / "study-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for name, command in commands.items():
        arm = args.out / name
        arm.mkdir()
        (arm / "commands.json").write_text(
            json.dumps({
                "training": command,
                "materialization": materialization_commands[name],
                "evaluation": evaluation_commands[name],
            }, indent=2) + "\n"
        )
    print(json.dumps({
        "out": str(args.out / "study-manifest.json"),
        "surface_group_sizes": surface_map,
        "arms": sorted(commands),
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
