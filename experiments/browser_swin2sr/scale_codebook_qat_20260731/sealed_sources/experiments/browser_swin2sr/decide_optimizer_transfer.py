#!/usr/bin/env python3
"""Select one optimizer trajectory from canonical full-development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decide_sensitive_io_development import (
    indexed_samples,
    paired_analysis,
)
from evaluate_promotion import (
    indexed_samples as promotion_indexed_samples,
    paired_gate,
    stratified_gate,
)
from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-optimizer-transfer-v1"
REPORT_FORMAT = "backdropkit-swin2sr-optimizer-transfer-stratified-v1"
DECISION_FORMAT = "backdropkit-swin2sr-optimizer-transfer-decision-v1"
EXPECTED_ARMS = {"qat8", "adamw"}
RANKING_POLICY = [
    "worst_source_psnr_delta",
    "overall_psnr_delta",
    "edge_psnr_delta",
    "teacher_psnr_delta",
    "optimizer_arm_stable_tiebreak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("format") != REPORT_FORMAT:
        raise RuntimeError(f"unsupported optimizer report: {path}")
    if not report.get("diagnostic_only"):
        raise RuntimeError(f"optimizer report is not diagnostic-only: {path}")
    if report.get("selection_source") != "development":
        raise RuntimeError(f"optimizer report is not development-only: {path}")
    if not report.get("complete_bundle"):
        raise RuntimeError(f"optimizer report is not complete: {path}")
    if report.get("model_promotion_authorized"):
        raise RuntimeError(f"optimizer report authorizes promotion: {path}")
    if not report.get("final_heldout_locked"):
        raise RuntimeError(f"optimizer report unlocked final heldout: {path}")
    summary = report.get("surface_summary") or {}
    if (
        int(summary.get("surfaces", -1)) != 204
        or int(summary.get("fully_hard_surfaces", -1)) != 204
        or int(summary.get("transitioning_surfaces", -1)) != 0
    ):
        raise RuntimeError(f"optimizer report is not all-hard: {path}")
    return report


def report_identity(report: dict) -> dict:
    return {
        "model": report["model"],
        "model_revision": report["model_revision"],
        "study_manifest_sha256": report["study_manifest_sha256"],
        "device": report["device"],
        "crop_size": int(report["crop_size"]),
        "bundle_sha256": report["bundle_sha256"],
        "bundle_manifest_sha256": report["bundle_manifest_sha256"],
        "q4_reference_sha256": report["q4_reference_sha256"],
        "surface_step": int(report["surface_step"]),
        "surface_group_sizes": report["surface_group_sizes"],
        "sample_hashes": sorted(indexed_samples(report)),
    }


def canonical_evidence(report: dict) -> dict:
    checkpoint = Path(report["checkpoint"])
    materialization_path = Path(report["materialization"])
    if file_sha256(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError("optimizer checkpoint identity changed")
    if file_sha256(materialization_path) != report["materialization_sha256"]:
        raise RuntimeError("optimizer materialization identity changed")
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("format")
        != "backdropkit-swin2sr-surface-materialization-v1"
    ):
        raise RuntimeError("optimizer materialization format differs")
    if (
        materialization.get("checkpoint_sha256")
        != report["checkpoint_sha256"]
    ):
        raise RuntimeError("optimizer materialization checkpoint differs")
    if not materialization.get("all_target_surfaces_materialized_hard"):
        raise RuntimeError("optimizer materialization is not all-hard")
    dense = materialization.get("dense_state") or {}
    state_path = Path(report["state"])
    if (
        Path(dense.get("path", "")).resolve() != state_path.resolve()
        or dense.get("file_sha256") != report["state_sha256"]
        or dense.get("tensor_sha256") != report["state_tensor_sha256"]
        or file_sha256(state_path) != report["state_sha256"]
    ):
        raise RuntimeError("optimizer report does not use canonical dense state")
    runtime = materialization.get("runtime_state") or {}
    runtime_path = Path(runtime.get("path", ""))
    if file_sha256(runtime_path) != runtime.get("file_sha256"):
        raise RuntimeError("optimizer runtime state identity changed")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "materialization": str(materialization_path),
        "materialization_sha256": report["materialization_sha256"],
        "dense_state": str(state_path),
        "dense_state_sha256": report["state_sha256"],
        "dense_state_tensor_sha256": report["state_tensor_sha256"],
        "runtime_state": str(runtime_path),
        "runtime_state_sha256": runtime["file_sha256"],
    }


def development_quality_gate(
    report: dict,
    q4: dict,
    screen: dict,
) -> dict:
    reference = promotion_indexed_samples(q4)
    candidate = promotion_indexed_samples(report)
    if reference.keys() != candidate.keys():
        raise RuntimeError("optimizer report and q4 do not share exact samples")
    rng = np.random.default_rng(int(screen["seed"]))
    paired = {
        metric: paired_gate(
            reference,
            candidate,
            metric,
            config,
            int(screen["bootstrap_samples"]),
            rng,
        )
        for metric, config in screen["paired_quality_gates"].items()
    }
    stratified = [
        stratified_gate(reference, candidate, config)
        for config in screen["stratified_gates"]
    ]
    return {
        "paired_quality": paired,
        "stratified_quality": stratified,
        "passed": bool(
            all(result["passed"] for result in paired.values())
            and all(result["passed"] for result in stratified)
        ),
    }


def ranking_key(candidate: dict) -> tuple:
    versus_q4 = candidate["versus_q4"]
    means = versus_q4["metrics"]
    return (
        float(versus_q4["worst_source_psnr_delta"]),
        float(means["psnr_vs_ground_truth"]["mean"]),
        float(means["edge_psnr_vs_ground_truth"]["mean"]),
        float(means["psnr_vs_teacher"]["mean"]),
        candidate["optimizer_arm"],
    )


def decide(
    study_path: Path,
    report_paths: list[Path],
    q4_path: Path,
    policy_path: Path,
) -> dict:
    study = json.loads(study_path.read_text())
    if study.get("format") != FORMAT:
        raise RuntimeError("unsupported optimizer-transfer study")
    if study.get("selection_source") != "development":
        raise RuntimeError("optimizer study is not development-only")
    if study.get("model_promotion_authorized"):
        raise RuntimeError("optimizer study authorizes promotion")
    if not study.get("final_heldout_locked"):
        raise RuntimeError("optimizer study does not lock final heldout")
    if set(study.get("arms", {})) != EXPECTED_ARMS:
        raise RuntimeError("optimizer study does not contain exact arms")
    if file_sha256(q4_path) != study["q4_reference_sha256"]:
        raise RuntimeError("optimizer q4 reference differs")
    if file_sha256(policy_path) != study["promotion_policy_sha256"]:
        raise RuntimeError("optimizer promotion policy differs")
    policy = json.loads(policy_path.read_text())
    if (
        study["development_quality_screen"]
        != {
            "seed": int(policy["seed"]),
            "bootstrap_samples": int(policy["bootstrap_samples"]),
            "paired_quality_gates": policy["paired_quality_gates"],
            "stratified_gates": policy["stratified_gates"],
        }
    ):
        raise RuntimeError("optimizer quality screen differs from policy")
    if int(study["selection_step"]) != 30000:
        raise RuntimeError("optimizer selection step is not 30000")
    if study.get("ranking_policy") != RANKING_POLICY:
        raise RuntimeError("optimizer ranking policy differs")
    if len(report_paths) != len(EXPECTED_ARMS):
        raise RuntimeError("optimizer decision requires exactly two reports")

    study_hash = file_sha256(study_path)
    q4 = json.loads(q4_path.read_text())
    loaded = []
    seen = set()
    identity = None
    for path in report_paths:
        report = load_report(path)
        arm = report.get("optimizer_arm")
        if arm not in EXPECTED_ARMS or arm in seen:
            raise RuntimeError(f"unexpected optimizer report arm: {path}")
        seen.add(arm)
        current_identity = report_identity(report)
        if current_identity["study_manifest_sha256"] != study_hash:
            raise RuntimeError("optimizer report uses another study")
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise RuntimeError("optimizer reports use different protocols")
        if indexed_samples(report).keys() != indexed_samples(q4).keys():
            raise RuntimeError("optimizer report and q4 sample identities differ")
        versus_q4 = paired_analysis(
            report,
            q4,
            bootstrap_samples=int(
                study["development_quality_screen"]["bootstrap_samples"]
            ),
            rng=np.random.default_rng(
                int(study["development_quality_screen"]["seed"])
            ),
        )
        quality = development_quality_gate(
            report,
            q4,
            study["development_quality_screen"],
        )
        loaded.append({
            "optimizer_arm": arm,
            "report": str(path),
            "report_sha256": file_sha256(path),
            "canonical_evidence": canonical_evidence(report),
            "versus_q4": versus_q4,
            "development_quality_gate": quality,
            "eligible": quality["passed"],
            "_report": report,
        })
    if seen != EXPECTED_ARMS:
        raise RuntimeError("optimizer reports do not cover QAT8 and AdamW")

    for candidate in loaded:
        other = next(
            item for item in loaded
            if item["optimizer_arm"] != candidate["optimizer_arm"]
        )
        candidate["versus_other_optimizer"] = paired_analysis(
            candidate["_report"],
            other["_report"],
            bootstrap_samples=int(
                study["development_quality_screen"]["bootstrap_samples"]
            ),
            rng=np.random.default_rng(
                int(study["development_quality_screen"]["seed"])
            ),
        )
    eligible = [candidate for candidate in loaded if candidate["eligible"]]
    selected = max(eligible, key=ranking_key) if eligible else None
    candidates = [
        {key: value for key, value in candidate.items() if key != "_report"}
        for candidate in sorted(
            loaded,
            key=lambda item: item["optimizer_arm"],
        )
    ]
    selected_evidence = (
        selected["canonical_evidence"] if selected is not None else {}
    )
    return {
        "format": DECISION_FORMAT,
        "selection_source": "development",
        "study_manifest": str(study_path),
        "study_manifest_sha256": study_hash,
        "q4_reference": str(q4_path),
        "q4_reference_sha256": file_sha256(q4_path),
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "development_quality_screen": study["development_quality_screen"],
        "ranking_policy": study["ranking_policy"],
        "candidates": candidates,
        "selected_optimizer": (
            selected["optimizer_arm"] if selected is not None else None
        ),
        "selected_checkpoint": selected_evidence.get("checkpoint"),
        "selected_checkpoint_sha256": selected_evidence.get(
            "checkpoint_sha256"
        ),
        "selected_materialization": selected_evidence.get("materialization"),
        "selected_materialization_sha256": selected_evidence.get(
            "materialization_sha256"
        ),
        "selected_dense_state": selected_evidence.get("dense_state"),
        "selected_dense_state_sha256": selected_evidence.get(
            "dense_state_sha256"
        ),
        "selected_runtime_state": selected_evidence.get("runtime_state"),
        "selected_runtime_state_sha256": selected_evidence.get(
            "runtime_state_sha256"
        ),
        "optimizer_recipe_selected": selected is not None,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "required_next_gate": (
            "Build and verify the selected canonical browser artifact, then "
            "precommit the one final heldout evaluation."
            if selected is not None
            else "No optimizer arm passed the development quality screen; "
            "revise all-hard recovery without opening final heldout."
        ),
    }


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    study = json.loads(args.study_manifest.read_text())
    for relative, expected in study.get("source_files", {}).items():
        if file_sha256(repository / relative) != expected:
            raise RuntimeError(f"optimizer-study source changed: {relative}")
    receipt = decide(
        args.study_manifest,
        args.reports,
        args.q4_reference,
        args.promotion_policy,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "optimizer_recipe_selected": receipt["optimizer_recipe_selected"],
        "selected_optimizer": receipt["selected_optimizer"],
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
