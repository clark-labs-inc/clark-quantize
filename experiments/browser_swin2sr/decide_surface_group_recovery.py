#!/usr/bin/env python3
"""Decide a matched control versus refined surface-group recovery study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from decide_pixel_objective import (
    METRICS,
    index_samples,
    paired_interval,
    source_deltas,
)
from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-surface-group-recovery-study-v1"
DECISION_FORMAT = "backdropkit-swin2sr-surface-group-recovery-decision-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def verify_arm(
    study: dict,
    arm: str,
    report_path: Path,
    report: dict,
) -> dict:
    configured = study["arms"][arm]
    checkpoint = Path(configured["final_checkpoint"])
    materialization_path = Path(configured["materialization"])
    if not checkpoint.exists() or not materialization_path.exists():
        raise RuntimeError(f"{arm} output is incomplete")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("surface_step", -1)) != int(study["selection_step"]):
        raise RuntimeError(f"{arm} checkpoint is at the wrong step")
    actual_map = payload["quantization"].get("surface_group_sizes", {}) or {}
    if actual_map != configured["surface_group_sizes"]:
        raise RuntimeError(f"{arm} checkpoint surface map differs")
    if Path(payload["surface_training"]["resume_checkpoint"]) != Path(
        configured["fork_checkpoint"]
    ):
        raise RuntimeError(f"{arm} checkpoint uses another fork")
    materialization = json.loads(materialization_path.read_text())
    if materialization.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise RuntimeError(f"{arm} materialization uses another checkpoint")
    dense = materialization["dense_state"]
    if Path(report["state"]).resolve() != Path(dense["path"]).resolve():
        raise RuntimeError(f"{arm} report uses another dense state")
    if report["state_sha256"] != dense["file_sha256"]:
        raise RuntimeError(f"{arm} report state hash differs")
    if Path(report["bundle"]).resolve() != Path(study["bundle"]).resolve():
        raise RuntimeError(f"{arm} report uses another bundle")
    if report["bundle_manifest_sha256"] != study["bundle_manifest_sha256"]:
        raise RuntimeError(f"{arm} report bundle identity differs")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "fork_checkpoint": configured["fork_checkpoint"],
        "fork_checkpoint_sha256": configured["fork_checkpoint_sha256"],
        "materialization": str(materialization_path),
        "materialization_sha256": file_sha256(materialization_path),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "state_sha256": report["state_sha256"],
        "surface_group_sizes": actual_map,
    }


def decide(study: dict, control_report: dict, candidate_report: dict) -> dict:
    control = index_samples(control_report)
    candidate = index_samples(candidate_report)
    if control.keys() != candidate.keys() or len(control) != 3040:
        raise RuntimeError("matched reports do not cover exact 3,040 samples")
    ordered = sorted(control)
    rule = study["decision_rule"]
    seed = int(rule["bootstrap_seed"])
    bootstrap_samples = int(rule["bootstrap_samples"])
    paired = {}
    for offset, metric in enumerate(METRICS):
        paired[metric] = paired_interval(
            [
                float(candidate[identity][metric])
                - float(control[identity][metric])
                for identity in ordered
            ],
            seed=seed + offset,
            samples=bootstrap_samples,
        )
    by_source = source_deltas(control, candidate)
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
        "every_source_guard": (
            min(by_source.values())
            >= float(rule["every_source_mean_min_db"])
        ),
        "scale_byte_budget": (
            int(study["selected_scale_bytes_delta"])
            <= int(rule["scale_byte_budget"])
        ),
    }
    selected = "refined" if all(gates.values()) else "control"
    return {
        "sample_count": len(ordered),
        "paired_candidate_minus_control": paired,
        "candidate_minus_control_by_source": by_source,
        "gates": gates,
        "selected_arm": selected,
        "surface_group_map_selected": selected == "refined",
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"surface-group decision exists: {args.out}")
    study = json.loads(args.study_manifest.read_text())
    if study.get("format") != FORMAT:
        raise RuntimeError("unsupported surface-group recovery study")
    if study.get("selection_source") != "development":
        raise RuntimeError("surface-group recovery is not development-only")
    if study.get("model_promotion_authorized"):
        raise RuntimeError("study improperly authorizes promotion")
    if not study.get("final_heldout_locked"):
        raise RuntimeError("study does not lock final heldout")
    if file_sha256(Path(study["bundle"])) != study["bundle_sha256"]:
        raise RuntimeError("development bundle changed")
    if file_sha256(Path(study["q4_reference"])) != study["q4_reference_sha256"]:
        raise RuntimeError("q4 reference changed")
    control_report = json.loads(args.control.read_text())
    candidate_report = json.loads(args.candidate.read_text())
    receipts = {
        "control": verify_arm(
            study, "control", args.control, control_report
        ),
        "refined": verify_arm(
            study, "refined", args.candidate, candidate_report
        ),
    }
    result = decide(study, control_report, candidate_report)
    selected_receipt = receipts[result["selected_arm"]]
    q4 = index_samples(json.loads(Path(study["q4_reference"]).read_text()))
    selected_report = (
        candidate_report
        if result["selected_arm"] == "refined"
        else control_report
    )
    selected_samples = index_samples(selected_report)
    ordered = sorted(selected_samples)
    selected_vs_q4 = paired_interval(
        [
            float(selected_samples[identity]["psnr_vs_ground_truth"])
            - float(q4[identity]["psnr_vs_ground_truth"])
            for identity in ordered
        ],
        seed=int(study["decision_rule"]["bootstrap_seed"]) + 10,
        samples=int(study["decision_rule"]["bootstrap_samples"]),
    )
    decision = {
        "format": DECISION_FORMAT,
        "selection_source": "development",
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "receipts": receipts,
        **result,
        "selected_checkpoint": selected_receipt["checkpoint"],
        "selected_checkpoint_sha256": selected_receipt["checkpoint_sha256"],
        "selected_minus_q4_ground_truth": selected_vs_q4,
        "decision_rule": study["decision_rule"],
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_arm": result["selected_arm"],
        "gates": result["gates"],
        "ground_truth_delta_db": result[
            "paired_candidate_minus_control"
        ]["psnr_vs_ground_truth"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
