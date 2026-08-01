#!/usr/bin/env python3
"""Apply the frozen full-development gate to an expanded surface map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decide_pixel_objective import (
    METRICS,
    index_samples,
    paired_interval,
    source_deltas,
)
from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-expanded-surface-group-full-decision-v1"
PROTOCOL_FORMAT = (
    "backdropkit-swin2sr-expanded-surface-group-full-development-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fork-receipt", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def paired_report(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    rule: dict,
) -> tuple[dict, dict[str, float]]:
    if baseline.keys() != candidate.keys() or len(baseline) != 3040:
        raise RuntimeError("candidate and baseline do not cover exact 3,040 samples")
    ordered = sorted(baseline)
    seed = int(rule["bootstrap_seed"])
    bootstrap_samples = int(rule["bootstrap_samples"])
    paired = {}
    for offset, metric in enumerate(METRICS):
        paired[metric] = paired_interval(
            [
                float(candidate[identity][metric])
                - float(baseline[identity][metric])
                for identity in ordered
            ],
            seed=seed + offset,
            samples=bootstrap_samples,
        )
    return paired, source_deltas(baseline, candidate)


def validate_report_bundle(
    report: dict,
    protocol: dict,
    bundle: dict,
    label: str,
) -> None:
    report_bundle = report.get("bundle")
    if not report_bundle or Path(report_bundle).resolve() != Path(
        protocol["bundle"]
    ).resolve():
        raise RuntimeError(f"{label} report uses another development bundle")
    if report.get("bundle_manifest_sha256") != bundle.get("manifest_sha256"):
        raise RuntimeError(f"{label} report changed bundle identity")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"full-development decision exists: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported expanded-map protocol")
    if protocol.get("model_promotion_authorized"):
        raise RuntimeError("protocol improperly authorizes promotion")
    if not protocol.get("final_heldout_locked"):
        raise RuntimeError("protocol did not lock final heldout")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("surface_map", "surface_map_sha256"),
        ("surface_decision", "surface_decision_sha256"),
        ("disjoint_joint_report", "disjoint_joint_report_sha256"),
        ("bundle", "bundle_sha256"),
        ("q4_reference", "q4_reference_sha256"),
    ):
        if file_sha256(Path(protocol[path_key])) != protocol[hash_key]:
            raise RuntimeError(f"protocol input changed: {path_key}")
    baseline_report = json.loads(args.baseline.read_text())
    candidate_report = json.loads(args.candidate.read_text())
    if len(candidate_report.get("samples", [])) != protocol["sample_count"]:
        raise RuntimeError("candidate report changed sample count")
    bundle = json.loads(Path(protocol["bundle"]).read_text())
    validate_report_bundle(candidate_report, protocol, bundle, "candidate")
    validate_report_bundle(baseline_report, protocol, bundle, "baseline")
    fork = json.loads(args.fork_receipt.read_text())
    if fork.get("parent_checkpoint_sha256") != protocol["checkpoint_sha256"]:
        raise RuntimeError("surface-map fork uses another parent")
    selected_map = json.loads(Path(protocol["surface_map"]).read_text())
    if fork.get("added_surface_group_sizes") != selected_map:
        raise RuntimeError("surface-map fork changed selected map")
    materialization = json.loads(args.materialization.read_text())
    if materialization.get("checkpoint_sha256") != fork[
        "fork_checkpoint_sha256"
    ]:
        raise RuntimeError("materialization uses another surface-map fork")
    dense = materialization["dense_state"]
    if candidate_report.get("state_sha256") != dense["file_sha256"]:
        raise RuntimeError("candidate report uses another dense hard state")
    baseline = index_samples(baseline_report)
    candidate = index_samples(candidate_report)
    rule = protocol["decision_rule"]
    paired, by_source = paired_report(baseline, candidate, rule)
    q4 = index_samples(json.loads(Path(protocol["q4_reference"]).read_text()))
    if q4.keys() != candidate.keys():
        raise RuntimeError("candidate and q4 sample identities differ")
    ordered = sorted(candidate)
    candidate_vs_q4 = {
        metric: paired_interval(
            [
                float(candidate[identity][metric])
                - float(q4[identity][metric])
                for identity in ordered
            ],
            seed=int(rule["bootstrap_seed"]) + 10 + offset,
            samples=int(rule["bootstrap_samples"]),
        )
        for offset, metric in enumerate(METRICS)
    }
    joint = json.loads(Path(protocol["disjoint_joint_report"]).read_text())
    selected_scale_bytes = joint["storage"]["selected_scale_bytes_delta"]
    gates = {
        "ground_truth_positive_lower_95": (
            paired["psnr_vs_ground_truth"]["lower_95"]
            > float(rule["ground_truth_lower_95_min_db"])
        ),
        "edge_mean_guard": (
            paired["edge_psnr_vs_ground_truth"]["mean"]
            >= float(rule["edge_mean_min_db"])
        ),
        "teacher_mean_guard": (
            paired["psnr_vs_teacher"]["mean"]
            >= float(rule["teacher_mean_min_db"])
        ),
        "every_source_ground_truth_guard": (
            min(by_source.values())
            >= float(rule["every_source_ground_truth_mean_min_db"])
        ),
        "scale_byte_budget": (
            selected_scale_bytes <= int(rule["scale_byte_budget"])
        ),
    }
    selected = all(gates.values())
    decision = {
        "format": FORMAT,
        "selection_source": "development",
        "diagnostic_only": True,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "decision_source_sha256": file_sha256(Path(__file__)),
        "baseline_report": str(args.baseline),
        "baseline_report_sha256": file_sha256(args.baseline),
        "candidate_report": str(args.candidate),
        "candidate_report_sha256": file_sha256(args.candidate),
        "fork_receipt": str(args.fork_receipt),
        "fork_receipt_sha256": file_sha256(args.fork_receipt),
        "materialization": str(args.materialization),
        "materialization_sha256": file_sha256(args.materialization),
        "sample_count": len(candidate),
        "paired_candidate_minus_g2": paired,
        "candidate_minus_g2_by_source": by_source,
        "candidate_minus_q4": candidate_vs_q4,
        "selected_surface_group_sizes": selected_map,
        "selected_scale_bytes_delta": selected_scale_bytes,
        "decision_rule": rule,
        "gates": gates,
        "expanded_map_recovery_selected": selected,
        "selected_checkpoint": (
            fork["fork_checkpoint"] if selected else protocol["checkpoint"]
        ),
        "selected_checkpoint_sha256": (
            fork["fork_checkpoint_sha256"]
            if selected else protocol["checkpoint_sha256"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "expanded_map_recovery_selected": selected,
        "gates": gates,
        "candidate_minus_g2_ground_truth": paired[
            "psnr_vs_ground_truth"
        ],
        "candidate_minus_q4_ground_truth": candidate_vs_q4[
            "psnr_vs_ground_truth"
        ],
        "candidate_minus_g2_by_source": by_source,
    }, indent=2))


if __name__ == "__main__":
    main()
