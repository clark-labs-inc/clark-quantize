#!/usr/bin/env python3
"""Run the preregistered 24-arm Swin2SR quantization/optimizer race."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


QUANTIZERS = ("optimal_mse", "absmean", "paretoq_seq")
ROTATIONS = ("none", "hadamard")
OPTIMIZERS = ("adamw", "adamw_prox", "bonsai_qat8", "bonsai_qat8_muon")
SCHEMA_VERSION = 1
SOURCE_FILES = (
    "artifact_state.py",
    "data_pipeline.py",
    "optimizer_controls.py",
    "qat_objective.py",
    "quantizer_controls.py",
    "rotation_control.py",
    "run_state.py",
    "train_swin2sr_qat.py",
)


@dataclass(frozen=True)
class Arm:
    quantizer: str
    rotation: str
    optimizer: str

    @property
    def name(self) -> str:
        return f"{self.quantizer}__{self.rotation}__{self.optimizer}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--schedule-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--promote-from", type=Path)
    parser.add_argument("--promote-count", type=int, default=8)
    parser.add_argument("--resume-stage")
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--eval-manifest", type=Path, action="append")
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--only", action="append", help="Run only named arms.")
    return parser.parse_args()


def all_arms() -> list[Arm]:
    return [
        Arm(quantizer, rotation, optimizer)
        for quantizer, rotation, optimizer in product(
            QUANTIZERS,
            ROTATIONS,
            OPTIMIZERS,
        )
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_validation(path: Path) -> dict | None:
    if not path.exists():
        return None
    result = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "validation":
                result = record
    return result


def arm_command(
    args: argparse.Namespace,
    arm: Arm,
    destination: Path,
    resume: Path | None,
) -> list[str]:
    script = Path(__file__).with_name("train_swin2sr_qat.py")
    command = [
        sys.executable,
        str(script),
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
        "--learning-rate",
        str(args.learning_rate),
        "--validate-every",
        str(args.validate_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--quantizer",
        arm.quantizer,
        "--rotation",
        arm.rotation,
        "--optimizer",
        arm.optimizer,
    ]
    for root in args.data:
        command.extend(("--data", str(root)))
    if resume:
        command.extend(("--resume", str(resume)))
    return command


async def evaluate_arm(
    args: argparse.Namespace,
    arm: Arm,
    destination: Path,
) -> tuple[int, dict | None]:
    if not args.eval_manifest:
        return 0, None
    output = destination / "heldout"
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stratified.py")),
        str(destination / "checkpoint-final.pt"),
        "--out",
        str(output),
        "--crop-size",
        str(args.crop_size),
        "--quantizer",
        arm.quantizer,
        "--rotation",
        arm.rotation,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    for manifest in args.eval_manifest:
        command.extend(("--manifest", str(manifest)))
    if args.eval_max_samples:
        command.extend(("--max-samples", str(args.eval_max_samples)))
    with (destination / "heldout.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return_code = await process.wait()
    metrics = output / "metrics.json"
    return return_code, json.loads(metrics.read_text()) if metrics.exists() else None


async def run_arm(
    args: argparse.Namespace,
    arm: Arm,
    semaphore: asyncio.Semaphore,
) -> dict:
    destination = args.out / args.stage / arm.name
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    prior = final_validation(destination / "train.jsonl")
    heldout_path = destination / "heldout" / "metrics.json"
    evaluation_ready = not args.eval_manifest or heldout_path.exists()
    training_ready = bool(
        prior and int(prior.get("step", 0)) >= args.steps
    )
    if training_ready and evaluation_ready:
        return json.loads(result_path.read_text()) if result_path.exists() else {
            "arm": asdict(arm),
            "name": arm.name,
            "status": "already_complete",
            "final": prior,
            "heldout": (
                json.loads(heldout_path.read_text())
                if heldout_path.exists()
                else None
            ),
        }
    if training_ready:
        started = time.time()
        async with semaphore:
            evaluation_code, heldout = await evaluate_arm(args, arm, destination)
        result = {
            "arm": asdict(arm),
            "name": arm.name,
            "status": "complete" if evaluation_code == 0 and heldout else "failed",
            "return_code": 0,
            "evaluation_return_code": evaluation_code,
            "elapsed_seconds": time.time() - started,
            "final": prior,
            "heldout": heldout,
        }
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result
    resume = None
    if args.resume_stage:
        resume = (
            args.out
            / args.resume_stage
            / arm.name
            / f"checkpoint-{args.resume_step}.pt"
        )
        if not resume.exists():
            raise FileNotFoundError(f"missing promotion checkpoint: {resume}")
    command = arm_command(args, arm, destination, resume)
    (destination / "command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    started = time.time()
    async with semaphore:
        with (destination / "stdout.log").open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            return_code = await process.wait()
        evaluation_code, heldout = await evaluate_arm(args, arm, destination)
    final = final_validation(destination / "train.jsonl")
    result = {
        "arm": asdict(arm),
        "name": arm.name,
        "status": (
            "complete"
            if return_code == 0 and evaluation_code == 0 and final
            else "failed"
        ),
        "return_code": return_code,
        "evaluation_return_code": evaluation_code,
        "elapsed_seconds": time.time() - started,
        "final": final,
        "heldout": heldout,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def rank_results(results: list[dict]) -> list[dict]:
    successful = [
        result
        for result in results
        if result.get("status") in ("complete", "already_complete")
        and result.get("final")
    ]
    use_heldout = bool(successful) and all(result.get("heldout") for result in successful)

    def rank_key(result: dict) -> tuple[float, float, float]:
        if use_heldout:
            overall = result["heldout"]["overall"]
            return (
                overall["psnr_vs_ground_truth"]["mean"],
                overall["psnr_vs_ground_truth"]["p10"],
                overall["edge_psnr_vs_ground_truth"]["mean"],
            )
        return (
            result["final"]["psnr_vs_ground_truth"],
            result["final"]["psnr_vs_teacher"],
            -result["final"].get("edge", float("inf")),
        )

    return sorted(
        successful,
        key=rank_key,
        reverse=True,
    )

def jsonable_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [jsonable_config_value(item) for item in value]
    return value


async def main_async(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    selected = all_arms()
    if args.promote_from:
        prior = json.loads(args.promote_from.read_text())
        selected_names = [
            row["name"]
            for row in prior["ranking"][: args.promote_count]
        ]
        selected = [arm for arm in selected if arm.name in selected_names]
        if not args.resume_stage or args.resume_step is None:
            raise ValueError(
                "--promote-from requires --resume-stage and --resume-step"
            )
    if args.only:
        requested = set(args.only)
        selected = [arm for arm in selected if arm.name in requested]
        missing = requested - {arm.name for arm in selected}
        if missing:
            raise ValueError(f"unknown arms: {sorted(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    source_root = source.parent
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_files": {
            name: sha256(source_root / name)
            for name in SOURCE_FILES
        },
        "python": sys.version,
        "pid": os.getpid(),
        "stage": args.stage,
        "matrix": [asdict(arm) | {"name": arm.name} for arm in selected],
        "config": {
            key: jsonable_config_value(item)
            for key, item in vars(args).items()
        },
    }
    (args.out / f"{args.stage}-study-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(run_arm(args, arm, semaphore) for arm in selected)
    )
    ranked = rank_results(results)
    summary = {
        "completed": len(ranked),
        "failed": len(results) - len(ranked),
        "ranking": [
            {
                "rank": index + 1,
                "name": result.get("name"),
                "ground_truth_psnr": result["final"]["psnr_vs_ground_truth"],
                "teacher_psnr": result["final"]["psnr_vs_teacher"],
                "heldout_ground_truth_psnr": (
                    result["heldout"]["overall"]["psnr_vs_ground_truth"]["mean"]
                    if result.get("heldout")
                    else None
                ),
                "heldout_ground_truth_p10": (
                    result["heldout"]["overall"]["psnr_vs_ground_truth"]["p10"]
                    if result.get("heldout")
                    else None
                ),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
            for index, result in enumerate(ranked)
        ],
        "promote_to_stage2": [
            result.get("name")
            for result in ranked[:8]
        ],
    }
    summary["promote_names"] = [
        result.get("name")
        for result in ranked[: args.promote_count]
    ]
    (args.out / f"{args.stage}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise RuntimeError(
            f"{summary['failed']} of {len(results)} study arms failed"
        )


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
