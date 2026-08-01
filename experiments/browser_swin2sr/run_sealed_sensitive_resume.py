#!/usr/bin/env python3
"""Resume a sensitive-I/O arm through its exact sealed training source."""

from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

import torch
import transformers

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-sealed-resume-v1"
TRAINER = "experiments/browser_swin2sr/train_swin2sr_full_surface.py"
PRELOAD_MODULES = {
    "clark_quantize.bonsai_qat8": "clark_quantize/bonsai_qat8.py",
    "clark_air.quant.optimal_ternary": (
        "clark_air/quant/optimal_ternary.py"
    ),
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface-step", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args, trainer_args = parser.parse_known_args()
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    return args, trainer_args


def option_value(arguments: list[str], flag: str) -> str:
    indices = [
        index for index, value in enumerate(arguments)
        if value == flag
    ]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise RuntimeError(f"resume command does not contain one {flag}")
    return arguments[indices[0] + 1]


def verify_checkpoint(
    checkpoint: Path,
    receipt_path: Path,
    *,
    surface_step: int,
) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("format")
        != "backdropkit-swin2sr-hard-started-validation-v2"
    ):
        raise RuntimeError("unsupported checkpoint receipt")
    matches = [
        item
        for item in receipt.get("checkpoints", [])
        if int(item["surface_step"]) == surface_step
    ]
    if len(matches) != 1:
        raise RuntimeError("checkpoint receipt does not identify exact step")
    record = matches[0]
    if file_sha256(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("resume checkpoint hash differs from receipt")
    return record


def verify_sealed_sources(
    study: dict,
    manifest_path: Path,
    repository: Path,
) -> Path:
    configured = study.get("sealed_source_root")
    if not configured:
        raise RuntimeError("study manifest has no sealed source root")
    sealed_root = Path(configured)
    if not sealed_root.is_absolute():
        sealed_root = repository / sealed_root
    for relative, expected in study.get("source_files", {}).items():
        source = sealed_root / relative
        if not source.exists() or file_sha256(source) != expected:
            raise RuntimeError(f"sealed source changed: {relative}")
    if study.get("selection_source") != "development":
        raise RuntimeError("resume study is not development-only")
    if not study.get("final_heldout_locked"):
        raise RuntimeError("resume study does not lock final heldout")
    return sealed_root


def preload_sealed_modules(sealed_root: Path) -> None:
    for module_name, relative in PRELOAD_MODULES.items():
        path = sealed_root / relative
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load sealed module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def local_import_closure(
    repository: Path,
    sealed_root: Path,
    *,
    sealed_at: float,
    module_names: set[str] | None = None,
) -> dict[str, dict]:
    closure = {}
    for name, module in sorted(sys.modules.items()):
        if module_names is not None and name not in module_names:
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(repository)
        except ValueError:
            continue
        if path.suffix not in (".py", ".pyc"):
            continue
        source = path.with_suffix(".py") if path.suffix == ".pyc" else path
        if not source.exists():
            continue
        try:
            sealed_relative = source.relative_to(sealed_root)
            source_kind = "sealed"
            recorded_path = str(sealed_relative)
        except ValueError:
            source_kind = "fallback"
            recorded_path = str(relative)
            if source.stat().st_mtime > sealed_at:
                raise RuntimeError(
                    f"fallback source postdates study seal: {relative}"
                )
        closure[name] = {
            "path": str(source),
            "recorded_path": recorded_path,
            "source_kind": source_kind,
            "sha256": file_sha256(source),
            "mtime_unix": source.stat().st_mtime,
        }
    return closure


def validate_trainer_arguments(
    arguments: list[str],
    checkpoint: Path,
    *,
    surface_step: int,
) -> int:
    if Path(option_value(arguments, "--base-checkpoint")).resolve() != (
        checkpoint.resolve()
    ):
        raise RuntimeError("trainer command uses a different checkpoint")
    target_step = int(option_value(arguments, "--steps"))
    if target_step <= surface_step:
        raise RuntimeError("trainer target does not advance checkpoint")
    if int(option_value(arguments, "--schedule-steps")) != 24000:
        raise RuntimeError("trainer command changed the hard schedule")
    if option_value(arguments, "--device") != "mps":
        raise RuntimeError("matched continuation must use MPS")
    if "--initialization" in arguments:
        raise RuntimeError("resume command must not reset initialization")
    return target_step


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def lineage_preserving_save(
    original_save,
    *,
    original_fork: str,
    checkpoint: Path,
    checkpoint_sha256: str,
):
    def save(path, *positional, **keywords):
        config = dict(keywords["config"])
        chain = list(config.get("resume_chain", []))
        chain.append({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        })
        config["resume_chain"] = chain
        config["resume_checkpoint"] = original_fork
        keywords["config"] = config
        return original_save(path, *positional, **keywords)

    return save


def main() -> None:
    args, trainer_args = parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"resume receipt already exists: {args.receipt}")
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    checkpoint_record = verify_checkpoint(
        args.checkpoint,
        args.checkpoint_receipt,
        surface_step=args.surface_step,
    )
    sealed_root = verify_sealed_sources(
        study,
        args.study_manifest,
        repository,
    )
    target_step = validate_trainer_arguments(
        trainer_args,
        args.checkpoint,
        surface_step=args.surface_step,
    )
    sealed_browser = sealed_root / "experiments/browser_swin2sr"
    current_browser = repository / "experiments/browser_swin2sr"
    for path in (current_browser, sealed_browser, sealed_root):
        sys.path.insert(0, str(path))
    baseline_modules = set(sys.modules)
    preload_sealed_modules(sealed_root)
    trainer_path = sealed_root / TRAINER
    namespace = runpy.run_path(
        str(trainer_path),
        run_name="_backdropkit_sealed_sensitive_trainer",
    )
    trainer_globals = namespace["train"].__globals__
    trainer_globals["save_checkpoint"] = lineage_preserving_save(
        trainer_globals["save_checkpoint"],
        original_fork=study["fork_checkpoint"],
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_record["checkpoint_sha256"],
    )
    imported_modules = set(sys.modules) - baseline_modules
    imported_modules.update(PRELOAD_MODULES)
    closure = local_import_closure(
        repository,
        sealed_root,
        sealed_at=float(study["created_at_unix"]),
        module_names=imported_modules,
    )
    command = [sys.executable, str(trainer_path), *trainer_args]
    base_receipt = {
        "format": FORMAT,
        "created_at_unix": time.time(),
        "status": "precommitted",
        "selection_source": "development",
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "checkpoint_receipt": str(args.checkpoint_receipt),
        "checkpoint_receipt_sha256": file_sha256(args.checkpoint_receipt),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "surface_step": args.surface_step,
        "target_step": target_step,
        "sealed_source_root": str(sealed_root),
        "sealed_trainer": str(trainer_path),
        "sealed_trainer_sha256": file_sha256(trainer_path),
        "runner": str(Path(__file__)),
        "runner_sha256": file_sha256(Path(__file__)),
        "local_import_closure": closure,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": command,
        "lineage_policy": {
            "resume_chain_checkpoint": str(args.checkpoint),
            "resume_chain_checkpoint_sha256": (
                checkpoint_record["checkpoint_sha256"]
            ),
            "evaluator_resume_checkpoint": study["fork_checkpoint"],
            "training_state_modified": False,
        },
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    write_receipt(args.receipt, base_receipt)
    if args.preflight_only:
        write_receipt(args.receipt, {
            **base_receipt,
            "status": "preflight-complete",
            "finished_at_unix": time.time(),
        })
        return
    previous_argv = sys.argv
    try:
        sys.argv = [str(trainer_path), *trainer_args]
        namespace["main"]()
    except BaseException as error:
        failed = {
            **base_receipt,
            "status": "failed",
            "finished_at_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_receipt(args.receipt, failed)
        raise
    finally:
        sys.argv = previous_argv
    output = Path(option_value(trainer_args, "--out"))
    final_checkpoint = output / f"checkpoint-surface-{target_step}.pt"
    if not final_checkpoint.exists():
        raise RuntimeError("sealed resume produced no target checkpoint")
    completed = {
        **base_receipt,
        "status": "complete",
        "finished_at_unix": time.time(),
        "target_checkpoint": str(final_checkpoint),
        "target_checkpoint_sha256": file_sha256(final_checkpoint),
    }
    write_receipt(args.receipt, completed)


if __name__ == "__main__":
    main()
