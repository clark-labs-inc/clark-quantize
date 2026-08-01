#!/usr/bin/env python3
"""Run the preregistered winner-only Swin2SR recovery ablation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from training_data_contract import dataset_fingerprints, validation_receipts


GROUP_SIZES = (60, 30)
SCALAR_DECOMPOSITIONS = (False, True)
ISLAND_POLICIES = ("frozen", "norms", "norms_and_residual")
LOSS_RECIPES = ("balanced", "teacher_heavy")
SOURCE_FILES = (
    "artifact_state.py",
    "data_pipeline.py",
    "island_policy.py",
    "optimizer_controls.py",
    "qat_objective.py",
    "quantizer_controls.py",
    "recovery_controls.py",
    "recovery_corpus_identity.py",
    "recovery_data.py",
    "recovery_resume_policy.py",
    "recovery_run_state.py",
    "rotation_control.py",
    "run_recovery_factorial.py",
    "run_state.py",
    "train_swin2sr_recovery.py",
    "training_data_contract.py",
)


@dataclass(frozen=True)
class Arm:
    group_size: int
    scalar_decomposition: bool
    island_policy: str
    loss_recipe: str

    @property
    def name(self) -> str:
        scalar = "scalar" if self.scalar_decomposition else "plain"
        return (
            f"g{self.group_size}__{scalar}__"
            f"{self.island_policy}__{self.loss_recipe}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--data-receipt", type=Path, action="append", default=[])
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    parser.add_argument("--eval-max-samples", type=int, required=True)
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--scalar-lr-multiplier", type=float, default=100.0)
    parser.add_argument("--localized-stage-steps", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--promote-from", type=Path)
    parser.add_argument("--promote-count", type=int, default=6)
    parser.add_argument("--resume-stage")
    parser.add_argument(
        "--reset-data-stream",
        action="store_true",
        help="Reset sampler/RNG state at an explicit corpus transition.",
    )
    parser.add_argument("--only", action="append")
    return parser.parse_args()


def arms() -> list[Arm]:
    return [
        Arm(group_size, scalar, island, loss)
        for group_size, scalar, island, loss in product(
            GROUP_SIZES,
            SCALAR_DECOMPOSITIONS,
            ISLAND_POLICIES,
            LOSS_RECIPES,
        )
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_optimizer(path: Path) -> str:
    import torch

    payload = torch.load(path, map_location="cpu")
    name = payload.get("optimizer_config", {}).get("name", "adamw")
    if name == "adamw_prox":
        return "adamw"
    if name not in {"adamw", "bonsai_qat8", "bonsai_qat8_muon"}:
        raise ValueError(f"unsupported base optimizer: {name}")
    return name


def selected_arms(args) -> list[Arm]:
    selected = arms()
    if args.promote_from:
        summary = json.loads(args.promote_from.read_text())
        names = {
            row["name"]
            for row in summary["ranking"][: args.promote_count]
        }
        selected = [arm for arm in selected if arm.name in names]
        if not args.resume_stage:
            raise ValueError("--promote-from requires --resume-stage")
    if args.only:
        requested = set(args.only)
        selected = [arm for arm in selected if arm.name in requested]
        missing = requested - {arm.name for arm in selected}
        if missing:
            raise ValueError(f"unknown recovery arms: {sorted(missing)}")
    return selected


def train_command(args, arm: Arm, destination: Path, optimizer: str) -> list[str]:
    checkpoint = args.base_checkpoint
    resume_optimizer = False
    if args.resume_stage:
        checkpoint = (
            args.out / args.resume_stage / arm.name / "checkpoint-final.pt"
        )
        resume_optimizer = True
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_swin2sr_recovery.py")),
        "--base-checkpoint",
        str(checkpoint),
        "--validation-data",
        str(args.validation_data),
        "--out",
        str(destination),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--crop-size",
        str(args.crop_size),
        "--group-size",
        str(arm.group_size),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer",
        optimizer,
        "--scalar-lr-multiplier",
        str(args.scalar_lr_multiplier),
        "--island-policy",
        arm.island_policy,
        "--loss-recipe",
        arm.loss_recipe,
        "--localized-stage-steps",
        str(args.localized_stage_steps),
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
    if arm.scalar_decomposition:
        command.append("--scalar-decomposition")
    if resume_optimizer:
        command.append("--resume-optimizer")
    if args.reset_data_stream:
        command.append("--reset-data-stream")
    return command


def evaluation_command(args, destination: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        str(destination / "swin2sr-ternary-hard.pt"),
        "--out",
        str(destination / "heldout"),
        "--max-samples",
        str(args.eval_max_samples),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    for manifest in args.eval_manifest:
        command.extend(("--manifest", str(manifest)))
    return command


async def run_process(command: list[str], log: Path) -> int:
    with log.open("ab") as output:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return await process.wait()


def paired_scores(candidate: dict, reference: dict) -> dict:
    reference_samples = {
        sample["sha256"]: sample
        for sample in reference["samples"]
    }
    source_deltas = defaultdict(list)
    deltas = defaultdict(list)
    for sample in candidate["samples"]:
        baseline = reference_samples.get(sample["sha256"])
        if baseline is None:
            raise ValueError(f"q4 reference is missing {sample['sha256']}")
        source = sample.get("source", "unknown")
        for metric in (
            "psnr_vs_ground_truth",
            "edge_psnr_vs_ground_truth",
            "psnr_vs_teacher",
        ):
            delta = float(sample[metric]) - float(baseline[metric])
            deltas[metric].append(delta)
            if metric == "psnr_vs_ground_truth":
                source_deltas[source].append(delta)
    means = {
        metric: sum(values) / len(values)
        for metric, values in deltas.items()
    }
    sources = {
        source: {
            "samples": len(values),
            "mean_psnr_delta": sum(values) / len(values),
        }
        for source, values in sorted(source_deltas.items())
    }
    return {
        "samples": len(candidate["samples"]),
        "mean_deltas": means,
        "sources": sources,
        "worst_source_psnr_delta": min(
            row["mean_psnr_delta"]
            for row in sources.values()
        ),
    }


def promotion_slacks(result: dict) -> dict[str, float]:
    paired = result["paired_q4"]
    means = paired["mean_deltas"]
    return {
        "worst_source_psnr": paired["worst_source_psnr_delta"] - (-0.15),
        "overall_psnr": means["psnr_vs_ground_truth"] - (-0.10),
        "edge_psnr": means["edge_psnr_vs_ground_truth"] - (-0.15),
        "teacher_psnr": means["psnr_vs_teacher"] - (-0.25),
    }


async def run_arm(args, arm, semaphore, optimizer, q4_reference):
    destination = args.out / args.stage / arm.name
    result_path = destination / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "complete":
            return result
    destination.mkdir(parents=True, exist_ok=True)
    training = train_command(args, arm, destination, optimizer)
    evaluation = evaluation_command(args, destination)
    (destination / "commands.json").write_text(json.dumps({
        "training": training,
        "evaluation": evaluation,
    }, indent=2) + "\n")
    started = time.time()
    async with semaphore:
        training_code = await run_process(training, destination / "stdout.log")
        evaluation_code = (
            await run_process(evaluation, destination / "heldout.log")
            if training_code == 0
            else None
        )
    metrics_path = destination / "heldout" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    result = {
        "arm": asdict(arm),
        "name": arm.name,
        "status": (
            "complete"
            if training_code == 0 and evaluation_code == 0 and metrics
            else "failed"
        ),
        "training_return_code": training_code,
        "evaluation_return_code": evaluation_code,
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "paired_q4": (
            paired_scores(metrics, q4_reference)
            if metrics
            else None
        ),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def rank_key(result: dict) -> tuple[float, float, float, float, float]:
    paired = result["paired_q4"]
    means = paired["mean_deltas"]
    slacks = promotion_slacks(result)
    return (
        min(slacks.values()),
        paired["worst_source_psnr_delta"],
        means["psnr_vs_teacher"],
        means["psnr_vs_ground_truth"],
        means["edge_psnr_vs_ground_truth"],
    )


async def main_async(args) -> None:
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    selected = selected_arms(args)
    optimizer = base_optimizer(args.base_checkpoint)
    q4_reference = json.loads(args.q4_reference.read_text())
    datasets = dataset_fingerprints(args.data)
    data_receipts = validation_receipts(args.data_receipt, datasets)
    args.out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    manifest = {
        "format": "backdropkit-swin2sr-recovery-factorial-v1",
        "created_at_unix": time.time(),
        "stage": args.stage,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": sha256(args.q4_reference),
        "training_data": datasets,
        "training_data_receipts": data_receipts,
        "source_files": {
            name: sha256(source_root / name)
            for name in SOURCE_FILES
        },
        "optimizer": optimizer,
        "matrix": [asdict(arm) | {"name": arm.name} for arm in selected],
        "config": {
            key: (
                [str(item) for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "ranking_policy": [
            "minimum_promotion_gate_slack",
            "worst_source_psnr_delta",
            "teacher_psnr_delta",
            "overall_psnr_delta",
            "edge_psnr_delta",
        ],
    }
    (args.out / f"{args.stage}-study-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(
        run_arm(args, arm, semaphore, optimizer, q4_reference)
        for arm in selected
    ))
    successful = [
        result for result in results
        if result["status"] == "complete"
    ]
    ranked = sorted(successful, key=rank_key, reverse=True)
    summary = {
        "completed": len(ranked),
        "failed": len(results) - len(ranked),
        "ranking": [
            {
                "rank": index + 1,
                "name": result["name"],
                **result["paired_q4"],
                "promotion_slacks": promotion_slacks(result),
                "elapsed_seconds": result["elapsed_seconds"],
            }
            for index, result in enumerate(ranked)
        ],
        "promote_names": [
            result["name"]
            for result in ranked[: args.promote_count]
        ],
    }
    (args.out / f"{args.stage}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} recovery arms failed")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
