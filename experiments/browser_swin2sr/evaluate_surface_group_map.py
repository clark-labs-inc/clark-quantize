#!/usr/bin/env python3
"""Evaluate a selected ternary surface-group map on disjoint development data."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from analyze_surface_group_sensitivity import paired_metric_evidence
from analyze_surface_restore_sensitivity import (
    load_sample,
    metric_delta,
    state_metrics,
    stratified_records,
)
from decide_surface_group_confirmation import FORMAT as DECISION_FORMAT
from evaluate_hard_started_checkpoints import load_checkpoint_student
from evaluate_stratified import (
    file_sha256,
    load_bundle_records,
    product_pixels,
    reconstruct,
)
from probe_sensitive_io_group_sizes import storage_receipt
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-surface-group-joint-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def joint_gate(
    evidence: dict[str, dict[str, float | int]],
    scale_bytes: int,
    byte_budget: int,
) -> tuple[bool, list[str]]:
    failures = []
    if evidence["psnr_vs_ground_truth"]["lower_95"] <= 0:
        failures.append("ground_truth_lower_95_not_positive")
    if evidence["edge_psnr_vs_ground_truth"]["mean"] < -0.02:
        failures.append("edge_mean_below_minus_0.02_db")
    if evidence["psnr_vs_teacher"]["mean"] < -0.1:
        failures.append("teacher_mean_below_minus_0.1_db")
    if scale_bytes > byte_budget:
        failures.append("scale_byte_budget_exceeded")
    return not failures, failures


def excluded_sample_ids(decision: dict) -> tuple[set[str], list[dict]]:
    identities = set()
    receipts = []
    for receipt in decision["reports"]:
        path = Path(receipt["path"])
        if file_sha256(path) != receipt["sha256"]:
            raise RuntimeError("confirmation report identity changed")
        report = json.loads(path.read_text())
        for sample in report["samples"]:
            identities.add(sample["sha256"])
        receipts.append({
            "path": str(path),
            "sha256": receipt["sha256"],
            "samples": len(report["samples"]),
        })
    return identities, receipts


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"joint evaluation exists: {args.out}")
    if min(
        args.max_samples,
        args.bootstrap_resamples,
        args.torch_threads,
    ) < 1:
        raise ValueError("samples, resamples, and threads must be positive")
    torch.set_num_threads(args.torch_threads)
    decision = json.loads(args.decision.read_text())
    if decision.get("format") != DECISION_FORMAT:
        raise RuntimeError("unsupported surface-group decision format")
    if decision.get("decision") != "evaluate_selected_map_jointly":
        raise RuntimeError("decision contains no surface map to evaluate")
    if decision.get("model_promotion_authorized"):
        raise RuntimeError("surface-group decision improperly authorizes promotion")
    if not decision.get("final_heldout_locked"):
        raise RuntimeError("surface-group decision did not lock final heldout")
    study_path = Path(decision["study_manifest"])
    if file_sha256(study_path) != decision["study_manifest_sha256"]:
        raise RuntimeError("surface-group study identity changed")
    study = json.loads(study_path.read_text())
    checkpoint = Path(study["checkpoint"])
    bundle_path = Path(study["bundle"])
    if file_sha256(checkpoint) != study["checkpoint_sha256"]:
        raise RuntimeError("surface-group checkpoint identity changed")
    if file_sha256(bundle_path) != study["bundle_sha256"]:
        raise RuntimeError("development bundle identity changed")
    excluded, exclusion_receipts = excluded_sample_ids(decision)
    bundle, records = load_bundle_records(bundle_path)
    available_records = [
        record for record in records
        if record["sha256"] not in excluded
    ]
    selected_records = stratified_records(available_records, args.max_samples)
    if len(selected_records) != args.max_samples:
        raise RuntimeError("insufficient disjoint development samples")
    samples = [load_sample(record, args.device) for record in selected_records]
    student, surfaces, payload = load_checkpoint_student(
        checkpoint,
        args.device,
    )
    if len(surfaces) != 204:
        raise RuntimeError("checkpoint does not cover all 204 surfaces")
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    available = {surface.name: surface for surface in surfaces}
    selected_map = decision["selected_surface_group_sizes"]
    unknown = sorted(set(selected_map) - set(available))
    if unknown:
        raise RuntimeError(f"decision contains unknown surfaces: {unknown}")
    total_values = sum(surface.module.weight.numel() for surface in surfaces)
    scale_bytes = 0
    storage_rows = []
    for name, group_size in sorted(selected_map.items()):
        surface = available[name]
        reference_group_size = surface.quantizer.group_size
        receipt = storage_receipt(
            surface.module.weight.numel(),
            group_size,
            total_values,
            reference_group_size,
        )
        reference_scales = (
            surface.module.weight.numel() + reference_group_size - 1
        ) // reference_group_size
        delta = (receipt["scale_count"] - reference_scales) * 2
        scale_bytes += delta
        storage_rows.append({
            "name": name,
            "group_size": group_size,
            "reference_group_size": reference_group_size,
            "scale_bytes_delta": delta,
        })
    teacher = pretrained(args.device).eval()
    with torch.inference_mode():
        teacher_outputs = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in samples
        ]
        baseline = state_metrics(student, samples, teacher_outputs, args.device)
        for name, group_size in selected_map.items():
            available[name].quantizer.group_size = group_size
        started = time.perf_counter()
        candidate = state_metrics(student, samples, teacher_outputs, args.device)
    bootstrap_indices = np.random.default_rng(args.bootstrap_seed).integers(
        0,
        len(samples),
        size=(args.bootstrap_resamples, len(samples)),
    )
    evidence = paired_metric_evidence(
        candidate["rows"],
        baseline["rows"],
        bootstrap_indices,
    )
    byte_budget = int(decision["scale_byte_budget"])
    passed, failures = joint_gate(evidence, scale_bytes, byte_budget)
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "disjoint_development_confirmation",
        "model_promotion_authorized": False,
        "recovery_study_eligible": passed,
        "final_heldout_locked": True,
        "decision": str(args.decision),
        "decision_sha256": file_sha256(args.decision),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": study["checkpoint_sha256"],
        "surface_step": int(payload["surface_step"]),
        "bundle": str(bundle_path),
        "bundle_sha256": study["bundle_sha256"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "excluded_confirmation_samples": len(excluded),
        "exclusion_receipts": exclusion_receipts,
        "sample_count": len(samples),
        "samples": [
            {
                key: sample.get(key)
                for key in ("sha256", "source", "script", "language")
            }
            for sample in samples
        ],
        "selected_surface_group_sizes": selected_map,
        "storage": {
            "scale_byte_budget": byte_budget,
            "selected_scale_bytes_delta": scale_bytes,
            "surfaces": storage_rows,
        },
        "hard": baseline["summary"],
        "candidate": candidate["summary"],
        "candidate_minus_hard": metric_delta(
            candidate["summary"],
            baseline["summary"],
        ),
        "paired_evidence": evidence,
        "bootstrap": {
            "method": "paired_nonparametric_mean",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
        "joint_gate": {
            "passed": passed,
            "failures": failures,
            "requirements": {
                "ground_truth_lower_95_db": "> 0",
                "edge_mean_delta_db": ">= -0.02",
                "teacher_mean_delta_db": ">= -0.1",
                "scale_bytes_delta": f"<= {byte_budget}",
            },
        },
        "device": args.device,
        "torch_threads": args.torch_threads,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "sample_count": len(samples),
        "selected_scale_bytes_delta": scale_bytes,
        "candidate_minus_hard": report["candidate_minus_hard"],
        "joint_gate": report["joint_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
