#!/usr/bin/env python3
"""Run matched fresh-FP32 and recovered-core full-surface Swin2SR lanes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from full_surface_artifacts import (
    materialization_command,
    portable_export_command,
)
from full_surface_provenance import (
    file_sha256 as sha256,
    selection_provenance,
    study_manifest,
    verify_eval_contract,
)
from run_recovery_factorial import paired_scores


SOURCE_FILES = (
    "artifact_state.py",
    "build_portable_onnx_artifact.py",
    "data_pipeline.py",
    "evaluate_stratified.py",
    "export_browser_artifact.py",
    "full_surface_artifacts.py",
    "full_surface_provenance.py",
    "materialize_surface_states.py",
    "optimizer_controls.py",
    "qat_objective.py",
    "quantizer_controls.py",
    "recovery_controls.py",
    "recovery_data.py",
    "recovery_run_state.py",
    "rewrite_onnx_ternary.py",
    "rotation_control.py",
    "run_full_surface_comparison.py",
    "run_recovery_factorial.py",
    "run_state.py",
    "surface_artifact_state.py",
    "surface_model_state.py",
    "surface_quantization.py",
    "surface_run_state.py",
    "surface_schedule.py",
    "train_swin2sr_full_surface.py",
    "training_data_contract.py",
    "verify_browser_artifact.py",
)
LANES = ("fp32", "recovered_core")


@dataclass(frozen=True)
class BaseQuantization:
    method: str
    group_size: int
    rotation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovered-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base-onnx", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--data-receipt", type=Path, action="append", default=[])
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--schedule-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--loss-recipe",
        choices=("balanced", "teacher_heavy"),
        required=True,
    )
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--eval-device",
        default="mps",
        help="Device for paired held-out scoring; must match the q4 reference.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--only", choices=LANES, action="append")
    return parser.parse_args()


def load_base_quantization(path: Path) -> BaseQuantization:
    import torch

    payload = torch.load(path, map_location="cpu")
    quantization = payload.get("quantization", {})
    if quantization.get("mode") != "ternary":
        raise ValueError("full-surface comparison requires a ternary recovery")
    method = quantization.get("method")
    if method not in {"optimal_mse", "absmean"}:
        raise ValueError(f"unsupported recovery quantizer: {method}")
    return BaseQuantization(
        method=method,
        group_size=int(quantization["group_size"]),
        rotation=str(quantization.get("rotation", "none")),
    )


def common_training_args(
    args: argparse.Namespace,
    destination: Path,
    base: BaseQuantization,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_swin2sr_full_surface.py")),
        "--validation-data",
        str(args.validation_data),
        "--out",
        str(destination),
        "--steps",
        str(args.steps),
        "--schedule-steps",
        str(args.schedule_steps),
        "--batch-size",
        str(args.batch_size),
        "--crop-size",
        str(args.crop_size),
        "--group-size",
        str(base.group_size),
        "--quantizer-method",
        base.method,
        "--core-rotation",
        base.rotation,
        "--learning-rate",
        str(args.learning_rate),
        "--loss-recipe",
        args.loss_recipe,
        "--validate-every",
        str(args.validate_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    for root in args.data:
        command.extend(("--data", str(root)))
    return command


def training_command(
    args: argparse.Namespace,
    lane: str,
    destination: Path,
    base: BaseQuantization,
) -> list[str]:
    command = common_training_args(args, destination, base)
    command.extend(("--initialization", lane))
    if lane == "recovered_core":
        command.extend((
            "--base-checkpoint",
            str(args.recovered_checkpoint),
        ))
    return command


def evaluation_command(
    args: argparse.Namespace,
    destination: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        str(destination / "canonical" / "dense-hard.pt"),
        "--bundle",
        str(args.bundle),
        "--out",
        str(destination / "heldout"),
        "--crop-size",
        str(args.crop_size),
        "--device",
        args.eval_device,
    ]


def export_command(destination: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("export_browser_artifact.py")),
        str(destination / "canonical" / "materialization.json"),
        "--out",
        str(destination / "artifact"),
    ]


def verification_command(
    args: argparse.Namespace,
    destination: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("verify_browser_artifact.py")),
        str(destination / "artifact"),
        "--reference-state",
        str(destination / "canonical" / "dense-hard.pt"),
        "--out",
        str(destination / "artifact" / "verification.json"),
        "--device",
        args.device,
    ]


async def run_process(command: list[str], log: Path) -> int:
    with log.open("ab") as output:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return await process.wait()


async def run_lane(
    args: argparse.Namespace,
    lane: str,
    base: BaseQuantization,
    q4_reference: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    destination = args.out / lane
    result_path = destination / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "complete":
            return result
    destination.mkdir(parents=True, exist_ok=True)
    training = training_command(args, lane, destination, base)
    materialize = materialization_command(args, destination)
    export = export_command(destination)
    portable_export = portable_export_command(args, destination)
    verification = verification_command(args, destination)
    evaluation = evaluation_command(args, destination)
    (destination / "commands.json").write_text(json.dumps({
        "training": training,
        "materialization": materialize,
        "export": export,
        "portable_export": portable_export,
        "verification": verification,
        "evaluation": evaluation,
    }, indent=2) + "\n")
    started = time.time()
    async with semaphore:
        training_code = await run_process(
            training,
            destination / "training.log",
        )
        materialization_code = (
            await run_process(
                materialize,
                destination / "materialization.log",
            )
            if training_code == 0
            else None
        )
        export_code = (
            await run_process(export, destination / "export.log")
            if materialization_code == 0
            else None
        )
        portable_export_code = (
            await run_process(
                portable_export,
                destination / "portable-export.log",
            )
            if export_code == 0
            else None
        )
        verification_code = (
            await run_process(
                verification,
                destination / "verification.log",
            )
            if portable_export_code == 0
            else None
        )
        evaluation_code = (
            await run_process(evaluation, destination / "heldout.log")
            if verification_code == 0
            else None
        )
    metrics_path = destination / "heldout" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    artifact = destination / "artifact"
    portable = destination / "portable" / "model.onnx"
    portable_ledger = portable.with_suffix(".ledger.json")
    portable_package = portable.with_suffix(portable.suffix + ".gz")
    artifact_bytes = (
        (artifact / "model.bin").stat().st_size
        + (artifact / "model.json").stat().st_size
        if metrics
        else None
    )
    if metrics:
        metrics["artifact_bytes"] = artifact_bytes
        metrics["runtime_artifact_bytes"] = portable_package.stat().st_size
        metrics["runtime_artifact_sha256"] = sha256(portable_package)
        metrics["runtime_artifact_compression"] = "gzip"
        metrics["runtime_expanded_artifact_bytes"] = portable.stat().st_size
        metrics["runtime_expanded_artifact_sha256"] = sha256(portable)
    result = {
        "lane": lane,
        "status": (
            "complete"
            if (
                training_code == 0
                and materialization_code == 0
                and export_code == 0
                and portable_export_code == 0
                and verification_code == 0
                and evaluation_code == 0
                and metrics
            )
            else "failed"
        ),
        "training_return_code": training_code,
        "materialization_return_code": materialization_code,
        "export_return_code": export_code,
        "portable_export_return_code": portable_export_code,
        "verification_return_code": verification_code,
        "evaluation_return_code": evaluation_code,
        "elapsed_seconds": time.time() - started,
        "materialization": str(
            destination / "canonical" / "materialization.json"
        ),
        "materialization_sha256": (
            sha256(destination / "canonical" / "materialization.json")
            if metrics
            else None
        ),
        "hard_state": str(destination / "canonical" / "dense-hard.pt"),
        "hard_state_sha256": (
            sha256(destination / "canonical" / "dense-hard.pt")
            if metrics
            else None
        ),
        "runtime_state": str(destination / "canonical" / "runtime-hard.pt"),
        "runtime_state_sha256": (
            sha256(destination / "canonical" / "runtime-hard.pt")
            if metrics
            else None
        ),
        "artifact": str(artifact),
        "artifact_bytes": artifact_bytes,
        "artifact_smaller_than_q4": (
            artifact_bytes < int(q4_reference["artifact_bytes"])
            if artifact_bytes is not None
            else False
        ),
        "artifact_manifest_sha256": (
            sha256(artifact / "model.json") if metrics else None
        ),
        "artifact_binary_sha256": (
            sha256(artifact / "model.bin") if metrics else None
        ),
        "portable_onnx": str(portable),
        "portable_onnx_bytes": portable.stat().st_size if metrics else None,
        "portable_onnx_sha256": sha256(portable) if metrics else None,
        "portable_ledger": str(portable_ledger),
        "portable_ledger_sha256": sha256(portable_ledger) if metrics else None,
        "runtime_package": str(portable_package),
        "runtime_package_bytes": (
            portable_package.stat().st_size if metrics else None
        ),
        "runtime_package_sha256": (
            sha256(portable_package) if metrics else None
        ),
        "runtime_package_compression": "gzip" if metrics else None,
        "runtime_package_smaller_than_q4": (
            portable_package.stat().st_size < int(q4_reference["artifact_bytes"])
            if metrics
            else False
        ),
        "metrics": metrics,
        "paired_q4": paired_scores(metrics, q4_reference) if metrics else None,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def rank_key(result: dict) -> tuple[float, float, float, float]:
    paired = result["paired_q4"]
    means = paired["mean_deltas"]
    return (
        paired["worst_source_psnr_delta"],
        means["psnr_vs_ground_truth"],
        means["edge_psnr_vs_ground_truth"],
        means["psnr_vs_teacher"],
    )


async def main_async(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.steps <= args.schedule_steps:
        raise ValueError("steps must extend beyond the hardening schedule")
    base = load_base_quantization(args.recovered_checkpoint)
    q4_reference = json.loads(args.q4_reference.read_text())
    verify_eval_contract(args, q4_reference)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = study_manifest(args, base, SOURCE_FILES)
    manifest_path = args.out / "study-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(
        run_lane(args, lane, base, q4_reference, semaphore)
        for lane in (args.only or LANES)
    ))
    completed = [
        result for result in results
        if result["status"] == "complete"
    ]
    ranked = sorted(completed, key=rank_key, reverse=True)
    by_lane = {result["lane"]: result for result in completed}
    direct = (
        paired_scores(
            by_lane["recovered_core"]["metrics"],
            by_lane["fp32"]["metrics"],
        )
        if set(by_lane) == set(LANES)
        else None
    )
    summary = {
        "format": "backdropkit-swin2sr-full-surface-summary-v1",
        **selection_provenance(args, q4_reference, manifest_path),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "ranking": [
            {
                "rank": index + 1,
                "lane": result["lane"],
                "materialization": result["materialization"],
                "materialization_sha256": result["materialization_sha256"],
                "hard_state": result["hard_state"],
                "hard_state_sha256": result["hard_state_sha256"],
                "runtime_state": result["runtime_state"],
                "runtime_state_sha256": result["runtime_state_sha256"],
                "artifact": result["artifact"],
                "artifact_bytes": result["artifact_bytes"],
                "artifact_smaller_than_q4": result["artifact_smaller_than_q4"],
                "artifact_manifest_sha256": result["artifact_manifest_sha256"],
                "artifact_binary_sha256": result["artifact_binary_sha256"],
                "portable_onnx": result["portable_onnx"],
                "portable_onnx_bytes": result["portable_onnx_bytes"],
                "portable_onnx_sha256": result["portable_onnx_sha256"],
                "portable_ledger": result["portable_ledger"],
                "portable_ledger_sha256": result["portable_ledger_sha256"],
                "runtime_package": result["runtime_package"],
                "runtime_package_bytes": result["runtime_package_bytes"],
                "runtime_package_sha256": result["runtime_package_sha256"],
                "runtime_package_compression": (
                    result["runtime_package_compression"]
                ),
                "runtime_package_smaller_than_q4": (
                    result["runtime_package_smaller_than_q4"]
                ),
                "paired_q4": result["paired_q4"],
            }
            for index, result in enumerate(ranked)
        ],
        "recovered_minus_fp32": direct,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} full-surface lanes failed")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
