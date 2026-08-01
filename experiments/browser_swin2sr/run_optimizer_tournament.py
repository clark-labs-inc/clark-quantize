#!/usr/bin/env python3
"""Run a matched optimizer-by-learning-rate Swin2SR QAT tournament."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from optimizer_controls import OPTIMIZERS


SCHEMA_VERSION = 1
SOURCE_FILES = (
    "optimizer_controls.py",
    "qat_objective.py",
    "quantizer_controls.py",
    "run_optimizer_tournament.py",
    "train_swin2sr_qat.py",
    "../../clark_quantize/qat_optimizer_variants.py",
    "../../clark_quantize/bonsai_muon.py",
    "../../clark_quantize/bonsai_qat8.py",
    "../../clark_quantize/ternary_kd.py",
)


@dataclass(frozen=True)
class Arm:
    optimizer: str
    learning_rate_multiplier: float

    @property
    def name(self) -> str:
        multiplier = f"{self.learning_rate_multiplier:g}".replace(".", "p")
        return f"{self.optimizer}__lr_x{multiplier}"


def csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def csv_floats(value: str) -> list[float]:
    result = [float(part) for part in csv_strings(value)]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("LR multipliers must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--schedule-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--base-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--lr-multipliers",
        type=csv_floats,
        default=csv_floats("0.25,1,4"),
    )
    parser.add_argument(
        "--optimizers",
        type=csv_strings,
        default=list(OPTIMIZERS),
    )
    parser.add_argument("--quantizer", default="absmean")
    parser.add_argument("--rotation", default="hadamard")
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--stage", default="optimizer-smoke")
    parser.add_argument("--only", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.optimizers) - set(OPTIMIZERS))
    if unknown:
        parser.error(f"unknown optimizers: {', '.join(unknown)}")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arms(args: argparse.Namespace) -> list[Arm]:
    result = [
        Arm(optimizer, multiplier)
        for optimizer in args.optimizers
        for multiplier in args.lr_multipliers
    ]
    if args.only:
        selected = set(args.only)
        result = [arm for arm in result if arm.name in selected]
    return result


def command_for(args: argparse.Namespace, arm: Arm, destination: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_swin2sr_qat.py")),
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
        str(args.group_size),
        "--learning-rate",
        str(args.base_learning_rate * arm.learning_rate_multiplier),
        "--validate-every",
        str(args.validate_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--quantizer",
        args.quantizer,
        "--rotation",
        args.rotation,
        "--optimizer",
        arm.optimizer,
    ]
    for root in args.data:
        command.extend(("--data", str(root)))
    return command


def last_validation(path: Path) -> dict | None:
    if not path.exists():
        return None
    validation = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "validation":
                validation = record
    return validation


async def run_arm(
    args: argparse.Namespace,
    arm: Arm,
    semaphore: asyncio.Semaphore,
) -> dict:
    destination = args.out / arm.name
    destination.mkdir(parents=True, exist_ok=True)
    command = command_for(args, arm, destination)
    queued = time.time()
    async with semaphore:
        started = time.time()
        with (destination / "runner.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            return_code = await process.wait()
    result = {
        **asdict(arm),
        "name": arm.name,
        "learning_rate": args.base_learning_rate * arm.learning_rate_multiplier,
        "return_code": return_code,
        "elapsed_seconds": time.time() - started,
        "queue_seconds": started - queued,
        "validation": last_validation(destination / "train.jsonl"),
    }
    (destination / "runner-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


async def main_async(args: argparse.Namespace) -> int:
    selected_arms = arms(args)
    if not selected_arms:
        raise SystemExit("no optimizer arms selected")
    root = Path(__file__).resolve().parent
    source_hashes = {
        source: sha256((root / source).resolve())
        for source in SOURCE_FILES
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "created_unix": time.time(),
        "protocol": {
            "teacher_and_initialization": "identical model revision and seed",
            "data_order": "identical roots and seed",
            "quantizer": args.quantizer,
            "rotation": args.rotation,
            "group_size": args.group_size,
            "steps": args.steps,
            "schedule_steps": args.schedule_steps,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "device": args.device,
            "seed": args.seed,
        },
        "source_sha256": source_hashes,
        "arms": [
            {
                **asdict(arm),
                "name": arm.name,
                "learning_rate": (
                    args.base_learning_rate * arm.learning_rate_multiplier
                ),
                "command": command_for(args, arm, args.out / arm.name),
            }
            for arm in selected_arms
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"arms": len(selected_arms), "out": str(args.out)}))
        return 0
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(run_arm(args, arm, semaphore) for arm in selected_arms)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "completed_unix": time.time(),
        "failed": sum(result["return_code"] != 0 for result in results),
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arms": len(results), "failed": summary["failed"]}))
    return 1 if summary["failed"] else 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
