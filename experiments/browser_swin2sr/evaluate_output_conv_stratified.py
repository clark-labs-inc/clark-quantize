#!/usr/bin/env python3
"""Evaluate one fully hard output-convolution arm on development data."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from evaluate_hard_started_checkpoints import (
    checkpoint_fork_provenance,
    load_checkpoint_student,
    surface_amount_summary,
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
from surface_curriculum import enter_hard_shadow
from surface_model_state import pretrained
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


GATE_FORMAT = "backdropkit-swin2sr-output-conv-development-gate-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def verify_output_gate_sources(
    gate: dict,
    repository: Path,
) -> dict[str, str]:
    if gate.get("format") != GATE_FORMAT:
        raise RuntimeError("unsupported output-convolution development gate")
    if gate.get("selection_source") != "development":
        raise RuntimeError("output-convolution gate is not development-only")
    if gate.get("model_promotion_authorized"):
        raise RuntimeError("output gate improperly authorizes promotion")
    if not gate.get("final_heldout_locked"):
        raise RuntimeError("output gate does not lock final heldout")
    verified = {}
    for relative, expected in gate["source_files"].items():
        actual = file_sha256(repository / relative)
        if actual != expected:
            raise RuntimeError(f"output-gate source changed: {relative}")
        verified[relative] = actual
    return verified


def main() -> None:
    args = parse_args()
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    gate = json.loads(args.gate_manifest.read_text())
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
    verified_gate_sources = verify_output_gate_sources(gate, repository)
    study_hash = file_sha256(args.study_manifest)
    if study_hash not in gate["study_manifests"].values():
        raise RuntimeError("study manifest is not sealed by output gate")
    if gate["bundle_sha256"] != file_sha256(args.bundle):
        raise RuntimeError("output gate and evaluator bundle differ")
    if gate["q4_reference_sha256"] != file_sha256(args.q4_reference):
        raise RuntimeError("output gate and evaluator q4 reference differ")
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
    student, surfaces, payload = load_checkpoint_student(
        args.checkpoint,
        args.device,
    )
    scheduled = surface_amount_summary(surfaces)
    enter_hard_shadow(surfaces)
    hard_started = surface_amount_summary(surfaces)
    if hard_started["transitioning_surfaces"]:
        raise RuntimeError("hard projection left transitioning surfaces")
    if hard_started["fully_hard_surfaces"] != scheduled["started_surfaces"]:
        raise RuntimeError("hard projection changed future surfaces")
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
    quantization = payload["quantization"]
    training = payload["surface_training"]
    report = {
        "format": "backdropkit-swin2sr-hard-started-stratified-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "complete_bundle": complete_bundle,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "state": str(args.checkpoint),
        "state_sha256": file_sha256(args.checkpoint),
        "surface_step": int(payload["surface_step"]),
        "surface_group_sizes": dict(sorted(
            (quantization.get("surface_group_sizes", {}) or {}).items()
        )),
        "convolution_learning_rate_multiplier": float(
            training.get("convolution_learning_rate_multiplier", 0.25)
        ),
        "fork_provenance": checkpoint_fork_provenance(
            payload,
            study,
            repository,
        ),
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": study_hash,
        "gate_manifest": str(args.gate_manifest),
        "gate_manifest_sha256": file_sha256(args.gate_manifest),
        "verified_gate_sources": verified_gate_sources,
        "sealed_source_root": str(sealed_root),
        "verified_sealed_sources": verified_sources,
        "device": args.device,
        "crop_size": args.crop_size,
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "scheduled_surface_summary": scheduled,
        "hard_started_surface_summary": hard_started,
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
        "samples": len(samples),
        "complete_bundle": complete_bundle,
        "surface_step": report["surface_step"],
        "overall": report["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
