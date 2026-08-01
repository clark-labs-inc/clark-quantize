#!/usr/bin/env python3
"""Seal matched retained-precision versus ternary-control Swin2SR arms."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from full_surface_provenance import file_sha256
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-retained-precision-study-v2"
SEEDS = (20260731, 20260732, 20260733)
ISLANDS = ("image_io_fp16", "rstb_fusion_fp16")
PRECISIONS = ("ternary", "fp16")
OVERLAY_SOURCES = (
    "experiments/browser_swin2sr/retained_precision_islands.py",
    "experiments/browser_swin2sr/train_retained_fp16_island.py",
    "experiments/browser_swin2sr/materialize_retained_precision_checkpoint.py",
    "experiments/browser_swin2sr/compare_evaluations.py",
    "experiments/browser_swin2sr/seal_retained_precision_study.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-study", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--q4-receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supersedes-failed-preseal", type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def command(
    overlay: Path,
    base_sources: Path,
    script: str,
    *arguments: str,
) -> list[str]:
    overlay_browser = overlay / "experiments/browser_swin2sr"
    base_browser = base_sources / "experiments/browser_swin2sr"
    pythonpath = f"{overlay_browser}:{base_browser}:{base_sources}"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={pythonpath}",
        sys.executable,
        str(overlay / script if (overlay / script).exists() else base_sources / script),
        *arguments,
    ]


def verify_base_study(path: Path) -> tuple[dict, Path]:
    study = json.loads(path.read_text())
    if study.get("format") != "backdropkit-swin2sr-scale-codebook-qat-study-v1":
        raise ValueError("unsupported immutable base study")
    root = path.parent / "sealed_sources"
    for relative, expected in study["source_sha256"].items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise RuntimeError(f"base sealed source changed: {relative}")
    parent = Path(study["parent_checkpoint"])
    if file_sha256(parent) != study["parent_checkpoint_sha256"]:
        raise RuntimeError("base study parent checkpoint changed")
    return study, root


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"retained-precision study exists: {args.out}")
    if args.steps != 4000 or args.learning_rate != 4e-6 or args.device != "mps":
        raise ValueError("initial retained-precision study is frozen at 4000 MPS steps and 4e-6")
    base, base_sources = verify_base_study(args.base_study)
    repository = Path(__file__).resolve().parents[2]
    overlay = args.out / "sealed_overlay"
    for relative in OVERLAY_SOURCES:
        source = repository / relative
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    source_sha256 = {
        relative: file_sha256(overlay / relative)
        for relative in OVERLAY_SOURCES
    }
    parent = Path(base["parent_checkpoint"])
    bundle = Path(base["bundle"])
    if file_sha256(bundle) != base["bundle_sha256"]:
        raise RuntimeError("development bundle changed")
    policy = json.loads(args.promotion_policy.read_text())
    q4 = json.loads(args.q4_receipt.read_text())
    if policy["reference"]["model_sha256"] != q4["model_sha256"]:
        raise RuntimeError("promotion policy and q4 receipt disagree")
    arms = {}
    for seed in SEEDS:
        for island in ISLANDS:
            for precision in PRECISIONS:
                name = f"{island}__{precision}__seed{seed}"
                root = args.out / name
                run = root / "run"
                checkpoint = run / f"checkpoint-island-{args.steps}.pt"
                training = command(
                    overlay,
                    base_sources,
                    "experiments/browser_swin2sr/train_retained_fp16_island.py",
                    "--parent", str(parent),
                    "--out", str(run),
                    "--validation-data", str(args.validation_data),
                    "--island", island,
                    "--precision", precision,
                    "--seed", str(seed),
                    "--steps", str(args.steps),
                    "--learning-rate", str(args.learning_rate),
                    "--validate-every", "500",
                    "--checkpoint-every", "500",
                    "--device", args.device,
                )
                for data in args.data:
                    training.extend(("--data", str(data)))
                materialized = root / "materialized"
                development = root / "development"
                arms[name] = {
                    "stage": "discovery" if seed == SEEDS[0] else "confirmation_if_discovery_passes",
                    "island": island,
                    "precision": precision,
                    "seed": seed,
                    "training_command": training,
                    "final_checkpoint": str(checkpoint),
                    "materialization_command": command(
                        overlay,
                        base_sources,
                        "experiments/browser_swin2sr/materialize_retained_precision_checkpoint.py",
                        str(checkpoint),
                        "--out", str(materialized),
                        "--source-device", "cpu",
                    ),
                    "materialization": str(materialized / "materialization.json"),
                    "evaluation_command": command(
                        overlay,
                        base_sources,
                        "experiments/browser_swin2sr/evaluate_stratified.py",
                        str(materialized / "dense-hard.pt"),
                        "--bundle", str(bundle),
                        "--out", str(development),
                        "--device", args.device,
                    ),
                    "development_report": str(development / "metrics.json"),
                }
    for seed in SEEDS:
        for island in ISLANDS:
            prefix = f"{island}__"
            control = arms[f"{prefix}ternary__seed{seed}"]["development_report"]
            candidate = arms[f"{prefix}fp16__seed{seed}"]["development_report"]
            arms[f"{prefix}fp16__seed{seed}"]["control_comparison_command"] = command(
                overlay,
                base_sources,
                "experiments/browser_swin2sr/compare_evaluations.py",
                control,
                candidate,
                "--out", str(args.out / f"{island}__seed{seed}__fp16-vs-ternary.json"),
                "--bootstrap-samples", "10000",
                "--seed", str(seed),
            )
    manifest = {
        "format": FORMAT,
        "status": "sealed_before_training_results",
        "selection_source": "development_only",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "supersedes_failed_preseal": (
            str(args.supersedes_failed_preseal)
            if args.supersedes_failed_preseal else None
        ),
        "supersedes_failed_preseal_sha256": (
            file_sha256(args.supersedes_failed_preseal)
            if args.supersedes_failed_preseal else None
        ),
        "preflight_correction": (
            "The v1 RSTB ternary control exposed 7.450580596923828e-09 "
            "optimal-MSE reprojection roundoff. V2 accepts only a four-epsilon "
            "FP32 bound and records both observed delta and tolerance."
        ),
        "hypothesis": (
            "A role-specific FP16 island trained from the exact hard-ternary "
            "state improves quality over a same-location hard-ternary control."
        ),
        "design": {
            "paired_by": ["island", "seed", "data_order", "objective", "optimizer", "steps"],
            "discovery_seed": SEEDS[0],
            "confirmation_seeds": list(SEEDS[1:]),
            "confirmation_gate": (
                "Run confirmation only when FP16 minus ternary has positive "
                "ground-truth paired-bootstrap lower 95% bound and no negative "
                "edge or teacher mean on the discovery seed."
            ),
            "optimizer_scope": (
                "AdamW is held fixed to isolate precision; optimizer variants "
                "are a later study only after a precision location survives."
            ),
        },
        "base_study": str(args.base_study),
        "base_study_sha256": file_sha256(args.base_study),
        "base_sealed_sources": str(base_sources),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "development_bundle": str(bundle),
        "development_bundle_sha256": file_sha256(bundle),
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "q4_receipt": str(args.q4_receipt),
        "q4_receipt_sha256": file_sha256(args.q4_receipt),
        "training_data": dataset_fingerprints(args.data),
        "validation_data": str(args.validation_data),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "arms": arms,
        "overlay_source_sha256": source_sha256,
    }
    path = args.out / "study-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "study_manifest": str(path),
        "study_manifest_sha256": file_sha256(path),
        "discovery_arms": [name for name, arm in arms.items() if arm["stage"] == "discovery"],
        "confirmation_arms": sum(arm["stage"] != "discovery" for arm in arms.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
