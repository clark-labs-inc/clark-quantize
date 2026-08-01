#!/usr/bin/env python3
"""Evaluate one all-hard optimizer-transfer arm on development data."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from evaluate_hard_started_checkpoints import (
    verify_study_sources,
)
from evaluate_hard_started_stratified import (
    sealed_source_root,
    verify_development_inputs,
)
from evaluate_stratified import (
    file_sha256,
    grouped_summaries,
    load_bundle_records,
    sample_metrics,
    summarize,
)
from materialize_surface_states import load_materialized_state
from surface_model_state import pretrained
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


FORMAT = "backdropkit-swin2sr-optimizer-transfer-v1"
REPORT_FORMAT = "backdropkit-swin2sr-optimizer-transfer-stratified-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def verify_optimizer_study(study: dict) -> None:
    if study.get("format") != FORMAT:
        raise RuntimeError("unsupported optimizer-transfer study")
    if study.get("selection_source") != "development":
        raise RuntimeError("optimizer study is not development-only")
    if study.get("model_promotion_authorized"):
        raise RuntimeError("optimizer study improperly authorizes promotion")
    if not study.get("final_heldout_locked"):
        raise RuntimeError("optimizer study does not lock final heldout")
    if set(study.get("arms", {})) != {"qat8", "adamw"}:
        raise RuntimeError("optimizer study does not contain exact matched arms")


def arm_identity(
    payload: dict,
    study: dict,
    repository: Path,
) -> str:
    if int(payload.get("surface_step", -1)) != 30000:
        raise RuntimeError("optimizer checkpoint is not at step 30000")
    training = payload["surface_training"]
    quantization = payload["quantization"]
    surface_map = {
        name: int(size)
        for name, size in (
            quantization.get("surface_group_sizes") or {}
        ).items()
    }
    if surface_map != study["surface_group_sizes"]:
        raise RuntimeError("optimizer checkpoint surface map differs")
    optimizer = (training.get("optimizer") or {}).get("name")
    if optimizer == "bonsai_qat8_muon":
        resume = Path(training.get("resume_checkpoint", ""))
        if not resume.is_absolute():
            resume = repository / resume
        if (
            not resume.exists()
            or file_sha256(resume) != study["qat8_checkpoint_sha256"]
        ):
            raise RuntimeError("QAT8 arm does not continue selected output")
        return "qat8"
    if optimizer == "adamw":
        if training.get("base_checkpoint") is not None:
            raise RuntimeError("AdamW arm did not start from the FP32 teacher")
        if training.get("initialization") != "fp32":
            raise RuntimeError("AdamW arm has a different initialization")
        return "adamw"
    raise RuntimeError(f"unexpected optimizer-transfer arm: {optimizer}")


def verify_materialization_identity(
    checkpoint: Path,
    payload: dict,
    materialization: dict,
) -> dict:
    checkpoint_hash = file_sha256(checkpoint)
    if materialization.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("materialization uses another optimizer checkpoint")
    recorded_checkpoint = Path(materialization.get("checkpoint", ""))
    if recorded_checkpoint.resolve() != checkpoint.resolve():
        raise RuntimeError("materialization checkpoint path differs")
    if int(materialization.get("surface_step", -1)) != int(
        payload.get("surface_step", -2)
    ):
        raise RuntimeError("materialization surface step differs")
    if not materialization.get("all_target_surfaces_materialized_hard"):
        raise RuntimeError("materialization is not an all-hard identity")
    surfaces = materialization.get("surfaces") or {}
    if len(surfaces) != 204:
        raise RuntimeError("materialization does not cover 204 surfaces")
    if materialization.get("quantization") != payload.get("quantization"):
        raise RuntimeError("materialization quantization contract differs")
    if materialization.get("surface_training") != payload.get(
        "surface_training"
    ):
        raise RuntimeError("materialization training contract differs")
    return {
        "surfaces": len(surfaces),
        "started_surfaces": len(surfaces),
        "future_surfaces": 0,
        "transitioning_surfaces": 0,
        "fully_hard_surfaces": len(surfaces),
    }


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    verify_optimizer_study(study)
    bundle, records = load_bundle_records(args.bundle)
    q4 = json.loads(args.q4_reference.read_text())
    verify_development_inputs(
        study,
        args.bundle,
        bundle,
        args.q4_reference,
        q4,
        args.device,
    )
    sealed_root = sealed_source_root(
        study,
        args.study_manifest,
        repository,
    )
    verified_sources = verify_study_sources(
        study,
        repository,
        sealed_source_root=sealed_root,
    )
    complete_bundle = args.max_samples is None
    if args.max_samples is not None:
        records = records[:args.max_samples]
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    arm = arm_identity(payload, study, repository)
    materialization, dense_state = load_materialized_state(
        args.materialization,
        "dense_state",
    )
    surface_summary = verify_materialization_identity(
        args.checkpoint,
        payload,
        materialization,
    )
    student = pretrained(args.device).eval()
    student.load_state_dict(dense_state, strict=True)
    teacher = pretrained(args.device).eval()
    started = time.perf_counter()
    samples = []
    for index, record in enumerate(records, start=1):
        samples.append(sample_metrics(
            record,
            teacher,
            student,
            args.crop_size,
            args.device,
        ))
        if index % 100 == 0:
            print(
                json.dumps({"evaluated": index, "total": len(records)}),
                flush=True,
            )
    report = {
        "format": REPORT_FORMAT,
        "diagnostic_only": True,
        "selection_source": "development",
        "complete_bundle": complete_bundle,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "optimizer_arm": arm,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "materialization": str(args.materialization),
        "materialization_sha256": file_sha256(args.materialization),
        "state": materialization["dense_state"]["path"],
        "state_sha256": materialization["dense_state"]["file_sha256"],
        "state_tensor_sha256": materialization["dense_state"]["tensor_sha256"],
        "surface_step": int(payload["surface_step"]),
        "surface_group_sizes": study["surface_group_sizes"],
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "sealed_source_root": str(sealed_root),
        "verified_sealed_sources": verified_sources,
        "device": args.device,
        "crop_size": args.crop_size,
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "surface_summary": surface_summary,
        "overall": summarize(samples),
        "by_source": grouped_summaries(samples, "source"),
        "by_split": grouped_summaries(samples, "split"),
        "by_script": grouped_summaries(samples, "script"),
        "by_language": grouped_summaries(samples, "language"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "samples": samples,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(output),
        "optimizer_arm": arm,
        "samples": len(samples),
        "complete_bundle": complete_bundle,
        "overall": report["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
