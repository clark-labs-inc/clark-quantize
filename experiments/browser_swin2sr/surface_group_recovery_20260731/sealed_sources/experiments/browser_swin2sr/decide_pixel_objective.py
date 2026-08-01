#!/usr/bin/env python3
"""Decide a matched Charbonnier versus PSNR-aligned development probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-pixel-objective-study-v1"
DECISION_FORMAT = "backdropkit-swin2sr-pixel-objective-decision-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def index_samples(report: dict) -> dict[str, dict]:
    indexed = {}
    for sample in report.get("samples", []):
        identity = sample.get("sha256")
        if not identity or identity in indexed:
            raise RuntimeError("report has missing or duplicate sample identities")
        indexed[identity] = sample
    return indexed


def paired_interval(
    values: list[float],
    *,
    seed: int,
    samples: int,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    bootstrapped = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "lower_95": float(np.quantile(bootstrapped, 0.025)),
        "upper_95": float(np.quantile(bootstrapped, 0.975)),
        "win_fraction": float((array > 0).mean()),
    }


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
    training = payload["surface_training"]
    objective = configured["pixel_objective"]
    if training.get("pixel_objective", "charbonnier") != objective:
        raise RuntimeError(f"{arm} checkpoint objective differs")
    if Path(training["resume_checkpoint"]) != Path(
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
        "materialization": str(materialization_path),
        "materialization_sha256": file_sha256(materialization_path),
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "state_sha256": report["state_sha256"],
        "pixel_objective": objective,
    }


def source_deltas(
    control: dict[str, dict],
    candidate: dict[str, dict],
) -> dict[str, float]:
    sources = sorted({
        sample.get("source")
        for sample in control.values()
        if sample.get("source")
    })
    result = {}
    for source in sources:
        identities = [
            identity
            for identity, sample in control.items()
            if sample.get("source") == source
        ]
        result[source] = float(np.mean([
            candidate[identity]["psnr_vs_ground_truth"]
            - control[identity]["psnr_vs_ground_truth"]
            for identity in identities
        ]))
    return result


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"decision already exists: {args.out}")
    study = json.loads(args.study_manifest.read_text())
    if study.get("format") != FORMAT:
        raise RuntimeError("unsupported pixel-objective study")
    if study.get("selection_source") != "development":
        raise RuntimeError("pixel-objective study is not development-only")
    if study.get("model_promotion_authorized"):
        raise RuntimeError("study improperly authorizes promotion")
    if not study.get("final_heldout_locked"):
        raise RuntimeError("study does not lock final heldout")
    if file_sha256(Path(study["q4_reference"])) != study["q4_reference_sha256"]:
        raise RuntimeError("q4 reference changed")
    if file_sha256(Path(study["bundle"])) != study["bundle_sha256"]:
        raise RuntimeError("development bundle changed")

    control_report = json.loads(args.control.read_text())
    candidate_report = json.loads(args.candidate.read_text())
    receipts = {
        "charbonnier": verify_arm(
            study, "charbonnier", args.control, control_report
        ),
        "psnr_aligned": verify_arm(
            study, "psnr_aligned", args.candidate, candidate_report
        ),
    }
    control = index_samples(control_report)
    candidate = index_samples(candidate_report)
    if control.keys() != candidate.keys() or len(control) != 3040:
        raise RuntimeError("matched reports do not cover exact 3,040 samples")
    ordered = sorted(control)
    seed = int(study["decision_rule"]["bootstrap_seed"])
    bootstrap_samples = int(
        study["decision_rule"]["bootstrap_samples"]
    )
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
    rule = study["decision_rule"]
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
    }
    selected = "psnr_aligned" if all(gates.values()) else "charbonnier"
    q4 = index_samples(json.loads(Path(study["q4_reference"]).read_text()))
    candidate_vs_q4 = paired_interval(
        [
            float(candidate[identity]["psnr_vs_ground_truth"])
            - float(q4[identity]["psnr_vs_ground_truth"])
            for identity in ordered
        ],
        seed=seed + 10,
        samples=bootstrap_samples,
    )
    decision = {
        "format": DECISION_FORMAT,
        "selection_source": "development",
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "sample_count": len(ordered),
        "receipts": receipts,
        "paired_candidate_minus_control": paired,
        "candidate_minus_control_by_source": by_source,
        "candidate_minus_q4_ground_truth": candidate_vs_q4,
        "decision_rule": rule,
        "gates": gates,
        "selected_pixel_objective": selected,
        "psnr_aligned_selected": selected == "psnr_aligned",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_pixel_objective": selected,
        "gates": gates,
        "candidate_minus_control_db": paired[
            "psnr_vs_ground_truth"
        ]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
