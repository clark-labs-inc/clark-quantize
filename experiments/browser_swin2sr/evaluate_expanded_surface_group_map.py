#!/usr/bin/env python3
"""Evaluate an expanded ternary surface map against hard and prior maps."""

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
from decide_broad_surface_group_confirmation import FORMAT as DECISION_FORMAT
from evaluate_hard_started_checkpoints import load_checkpoint_student
from evaluate_stratified import (
    file_sha256,
    load_bundle_records,
    product_pixels,
    reconstruct,
)
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-expanded-surface-group-joint-evaluation-v1"


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


def quality_gate(
    evidence: dict[str, dict[str, float | int]],
) -> tuple[bool, list[str]]:
    failures = []
    if evidence["psnr_vs_ground_truth"]["lower_95"] <= 0:
        failures.append("ground_truth_lower_95_not_positive")
    if evidence["edge_psnr_vs_ground_truth"]["mean"] < -0.02:
        failures.append("edge_mean_below_minus_0.02_db")
    if evidence["psnr_vs_teacher"]["mean"] < -0.1:
        failures.append("teacher_mean_below_minus_0.1_db")
    return not failures, failures


def effective_ternary_bits(
    values: int,
    packed_code_bytes: int,
    scale_bytes: int,
) -> float:
    if values < 1 or packed_code_bytes < 0 or scale_bytes < 0:
        raise ValueError("ternary storage inputs must be non-negative")
    return (packed_code_bytes + scale_bytes) * 8 / values


def sample_exclusions(
    paths_and_hashes: list[tuple[Path, str]],
) -> tuple[set[str], list[dict]]:
    identities = set()
    receipts = []
    seen_paths = set()
    for path, expected_sha256 in paths_and_hashes:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(f"exclusion report identity changed: {path}")
        report = json.loads(path.read_text())
        report_identities = {
            str(sample["sha256"])
            for sample in report["samples"]
        }
        identities.update(report_identities)
        receipts.append({
            "path": str(path),
            "sha256": expected_sha256,
            "sample_count": len(report["samples"]),
            "unique_sample_count": len(report_identities),
        })
    return identities, receipts


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"expanded map evaluation exists: {args.out}")
    if min(
        args.max_samples,
        args.bootstrap_resamples,
        args.torch_threads,
    ) < 1:
        raise ValueError("samples, resamples, and threads must be positive")
    torch.set_num_threads(args.torch_threads)
    decision = json.loads(args.decision.read_text())
    if decision.get("format") != DECISION_FORMAT:
        raise RuntimeError("unsupported broad surface decision")
    if decision.get("decision") != "evaluate_expanded_map_jointly":
        raise RuntimeError("decision contains no expanded map")
    if decision.get("model_promotion_authorized"):
        raise RuntimeError("broad decision improperly authorizes promotion")
    if not decision.get("final_heldout_locked"):
        raise RuntimeError("broad decision did not lock final heldout")
    study_path = Path(decision["study_manifest"])
    if file_sha256(study_path) != decision["study_manifest_sha256"]:
        raise RuntimeError("broad confirmation study identity changed")
    study = json.loads(study_path.read_text())
    protocol_path = Path(study["protocol"])
    if file_sha256(protocol_path) != study["protocol_sha256"]:
        raise RuntimeError("broad screen protocol identity changed")
    protocol = json.loads(protocol_path.read_text())
    checkpoint = Path(study["checkpoint"])
    bundle_path = Path(study["bundle"])
    if file_sha256(checkpoint) != study["checkpoint_sha256"]:
        raise RuntimeError("expanded-map checkpoint identity changed")
    if file_sha256(bundle_path) != study["bundle_sha256"]:
        raise RuntimeError("expanded-map development bundle changed")
    known_hashes = {
        str(record["path"]): record["sha256"]
        for record in protocol["excluded_reports"]
    }
    known_hashes.update({
        str(record["path"]): record["sha256"]
        for record in study["screens"]
    })
    known_hashes.update({
        str(record["path"]): record["sha256"]
        for record in decision["reports"]
    })
    exclusions, exclusion_receipts = sample_exclusions([
        (Path(path), known_hashes[path])
        for path in study["confirmation"]["excluded_reports"]
    ] + [
        (Path(record["path"]), record["sha256"])
        for record in decision["reports"]
    ])
    bundle, records = load_bundle_records(bundle_path)
    available_records = [
        record for record in records
        if record["sha256"] not in exclusions
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
    total_values = sum(
        surface.module.weight.numel()
        for surface in surfaces
    )
    packed_code_bytes = sum(
        (surface.module.weight.numel() + 4) // 5
        for surface in surfaces
    )
    baseline_scale_bytes = sum(
        (
            surface.module.weight.numel()
            + surface.quantizer.group_size
            - 1
        ) // surface.quantizer.group_size * 2
        for surface in surfaces
    )
    existing_map = decision["existing_surface_group_sizes"]
    added_map = decision["added_surface_group_sizes"]
    selected_map = decision["selected_surface_group_sizes"]
    if selected_map != dict(sorted({**existing_map, **added_map}.items())):
        raise RuntimeError("expanded map is not existing plus added surfaces")
    unknown = sorted(set(selected_map) - set(available))
    if unknown:
        raise RuntimeError(f"expanded map contains unknown surfaces: {unknown}")
    teacher = pretrained(args.device).eval()
    with torch.inference_mode():
        teacher_outputs = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in samples
        ]
        hard = state_metrics(student, samples, teacher_outputs, args.device)
        for name, group_size in existing_map.items():
            available[name].quantizer.group_size = group_size
        existing = state_metrics(student, samples, teacher_outputs, args.device)
        for name, group_size in added_map.items():
            available[name].quantizer.group_size = group_size
        started = time.perf_counter()
        expanded = state_metrics(student, samples, teacher_outputs, args.device)
    bootstrap_indices = np.random.default_rng(args.bootstrap_seed).integers(
        0,
        len(samples),
        size=(args.bootstrap_resamples, len(samples)),
    )
    existing_vs_hard = paired_metric_evidence(
        existing["rows"], hard["rows"], bootstrap_indices
    )
    expanded_vs_hard = paired_metric_evidence(
        expanded["rows"], hard["rows"], bootstrap_indices
    )
    expanded_vs_existing = paired_metric_evidence(
        expanded["rows"], existing["rows"], bootstrap_indices
    )
    overall_passed, overall_failures = quality_gate(expanded_vs_hard)
    incremental_passed, incremental_failures = quality_gate(
        expanded_vs_existing
    )
    size_passed = (
        decision["selected_scale_bytes_delta"]
        <= decision["scale_byte_budget"]
    )
    eligible = overall_passed and incremental_passed and size_passed
    report = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "second_disjoint_development_confirmation",
        "model_promotion_authorized": False,
        "recovery_study_eligible": eligible,
        "final_heldout_locked": True,
        "decision": str(args.decision),
        "decision_sha256": file_sha256(args.decision),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": study["checkpoint_sha256"],
        "surface_step": int(payload["surface_step"]),
        "bundle": str(bundle_path),
        "bundle_sha256": study["bundle_sha256"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "excluded_unique_sample_count": len(exclusions),
        "exclusion_receipts": exclusion_receipts,
        "sample_count": len(samples),
        "samples": [
            {
                key: sample.get(key)
                for key in ("sha256", "source", "script", "language")
            }
            for sample in samples
        ],
        "existing_surface_group_sizes": existing_map,
        "added_surface_group_sizes": added_map,
        "selected_surface_group_sizes": selected_map,
        "storage": {
            "scale_byte_budget": decision["scale_byte_budget"],
            "total_quantized_values": total_values,
            "packed_code_bytes": packed_code_bytes,
            "baseline_scale_bytes": baseline_scale_bytes,
            "existing_scale_bytes_delta": decision[
                "existing_scale_bytes_delta"
            ],
            "added_scale_bytes_delta": decision["added_scale_bytes_delta"],
            "selected_scale_bytes_delta": decision[
                "selected_scale_bytes_delta"
            ],
            "baseline_bits_per_value_including_scales": (
                effective_ternary_bits(
                    total_values,
                    packed_code_bytes,
                    baseline_scale_bytes,
                )
            ),
            "existing_bits_per_value_including_scales": (
                effective_ternary_bits(
                    total_values,
                    packed_code_bytes,
                    baseline_scale_bytes
                    + decision["existing_scale_bytes_delta"],
                )
            ),
            "expanded_bits_per_value_including_scales": (
                effective_ternary_bits(
                    total_values,
                    packed_code_bytes,
                    baseline_scale_bytes
                    + decision["selected_scale_bytes_delta"],
                )
            ),
        },
        "hard": hard["summary"],
        "existing": existing["summary"],
        "expanded": expanded["summary"],
        "existing_minus_hard": metric_delta(
            existing["summary"], hard["summary"]
        ),
        "expanded_minus_hard": metric_delta(
            expanded["summary"], hard["summary"]
        ),
        "expanded_minus_existing": metric_delta(
            expanded["summary"], existing["summary"]
        ),
        "paired_evidence": {
            "existing_minus_hard": existing_vs_hard,
            "expanded_minus_hard": expanded_vs_hard,
            "expanded_minus_existing": expanded_vs_existing,
        },
        "bootstrap": {
            "method": "paired_nonparametric_mean",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
        "joint_gate": {
            "passed": eligible,
            "overall_map": {
                "passed": overall_passed,
                "failures": overall_failures,
            },
            "incremental_expansion": {
                "passed": incremental_passed,
                "failures": incremental_failures,
            },
            "scale_byte_budget": {"passed": size_passed},
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
        "expanded_minus_hard": report["expanded_minus_hard"],
        "expanded_minus_existing": report["expanded_minus_existing"],
        "joint_gate": report["joint_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
