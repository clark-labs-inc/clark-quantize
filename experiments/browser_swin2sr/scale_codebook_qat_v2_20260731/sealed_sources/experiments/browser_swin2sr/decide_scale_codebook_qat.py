#!/usr/bin/env python3
"""Apply frozen full-development, size, and endpoint gates to scale-QAT arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_promotion import indexed_samples, paired_gate, stratified_gate
from full_surface_provenance import file_sha256
from scale_codebook_state import CHECKPOINT_FORMAT, MATERIALIZATION_FORMAT
from seal_scale_codebook_qat_study import FORMAT as STUDY_FORMAT


FORMAT = "backdropkit-swin2sr-scale-codebook-qat-decision-v1"


def quality_gate(reference: dict, candidate: dict, policy: dict) -> dict:
    reference_samples = indexed_samples(reference)
    candidate_samples = indexed_samples(candidate)
    if reference_samples.keys() != candidate_samples.keys() or len(reference_samples) != 3040:
        raise RuntimeError("scale-QAT report does not cover the exact q4 sample set")
    generator = np.random.default_rng(int(policy["seed"]))
    paired = {
        metric: paired_gate(
            reference_samples,
            candidate_samples,
            metric,
            config,
            int(policy["bootstrap_samples"]),
            generator,
        )
        for metric, config in policy["paired_quality_gates"].items()
    }
    stratified = [
        stratified_gate(reference_samples, candidate_samples, config)
        for config in policy["stratified_gates"]
    ]
    return {
        "paired": paired,
        "stratified": stratified,
        "passed": bool(
            all(result["passed"] for result in paired.values())
            and all(result["passed"] for result in stratified)
        ),
    }


def verify_arm(study: dict, arm: str, configured: dict) -> dict:
    checkpoint_path = Path(configured["final_checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"{arm} has the wrong checkpoint format")
    if int(payload.get("scale_step", -1)) != int(study["steps"]):
        raise RuntimeError(f"{arm} stopped at the wrong scale step")
    if payload.get("parent_checkpoint_sha256") != study["parent_checkpoint_sha256"]:
        raise RuntimeError(f"{arm} uses another parent")
    if payload["scale_codebook"]["target_bits_by_category"] != configured[
        "bits_by_category"
    ]:
        raise RuntimeError(f"{arm} uses another scale map")
    materialization_path = Path(configured["materialization"])
    materialization = json.loads(materialization_path.read_text())
    if materialization.get("format") != MATERIALIZATION_FORMAT:
        raise RuntimeError(f"{arm} has the wrong materialization format")
    if materialization.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError(f"{arm} materializes another checkpoint")
    report_path = Path(configured["development_report"])
    report = json.loads(report_path.read_text())
    dense = materialization["dense_state"]
    if (
        Path(report["state"]).resolve() != Path(dense["path"]).resolve()
        or report["state_sha256"] != dense["file_sha256"]
    ):
        raise RuntimeError(f"{arm} evaluates another dense state")
    artifact_root = Path(configured["artifact"])
    manifest_path = artifact_root / "model.json"
    binary_path = artifact_root / "model.bin"
    artifact = json.loads(manifest_path.read_text())
    if artifact.get("format") != "clark-air-browser-v3":
        raise RuntimeError(f"{arm} is not a bit-packed scale artifact")
    if artifact.get("materialization_manifest_sha256") != file_sha256(
        materialization_path
    ):
        raise RuntimeError(f"{arm} artifact uses another materialization")
    total_bytes = manifest_path.stat().st_size + binary_path.stat().st_size
    ledger = artifact["storage_ledger"]
    if int(ledger["binary_bytes"]) != binary_path.stat().st_size:
        raise RuntimeError(f"{arm} artifact ledger differs from disk")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "materialization": str(materialization_path),
        "materialization_sha256": file_sha256(materialization_path),
        "development_report": str(report_path),
        "development_report_sha256": file_sha256(report_path),
        "report": report,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "artifact_binary": str(binary_path),
        "artifact_binary_sha256": file_sha256(binary_path),
        "artifact_total_bytes": total_bytes,
        "storage_ledger": ledger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"scale-QAT decision exists: {args.out}")
    study = json.loads(args.study_manifest.read_text())
    if study.get("format") != STUDY_FORMAT:
        raise RuntimeError("unsupported scale-QAT study")
    if study.get("model_promotion_authorized") or not study.get("final_heldout_locked"):
        raise RuntimeError("scale-QAT study violates promotion isolation")
    repository = Path(__file__).resolve().parents[2]
    for relative, expected in study["source_sha256"].items():
        if file_sha256(repository / relative) != expected:
            raise RuntimeError(f"sealed scale-QAT source changed: {relative}")
    for key in ("recovery_decision", "parent_checkpoint", "bundle", "q4_reference", "promotion_policy"):
        if file_sha256(Path(study[key])) != study[f"{key}_sha256"]:
            raise RuntimeError(f"sealed scale-QAT input changed: {key}")
    q4 = json.loads(Path(study["q4_reference"]).read_text())
    policy = json.loads(Path(study["promotion_policy"]).read_text())
    receipts = {
        arm: verify_arm(study, arm, configured)
        for arm, configured in study["arms"].items()
    }
    size_rule = study["size_rule"]
    candidates = {}
    for arm, receipt in receipts.items():
        quality = quality_gate(q4, receipt["report"], policy)
        reduction = 1 - receipt["artifact_total_bytes"] / float(
            size_rule["quality_model_declared_bytes"]
        )
        size_passed = reduction >= float(size_rule["minimum_reduction_fraction"])
        trunk_bits = float(
            receipt["storage_ledger"]["packed_trunk_bits_per_quantized_weight"]
        )
        endpoint_bits_passed = bool(
            arm == "uniform_q1"
            and trunk_bits <= float(size_rule["uniform_q1_max_packed_trunk_bits"])
        )
        candidates[arm] = {
            **{key: value for key, value in receipt.items() if key != "report"},
            "bits_by_category": study["arms"][arm]["bits_by_category"],
            "quality_vs_q4": quality,
            "size_reduction_vs_declared_q4": reduction,
            "size_passed": size_passed,
            "endpoint_bits_passed": endpoint_bits_passed,
            "endpoint_eligible": bool(
                quality["passed"] and size_passed and endpoint_bits_passed
            ),
        }
    endpoint = "uniform_q1" if candidates["uniform_q1"]["endpoint_eligible"] else None
    ranked = sorted(
        candidates,
        key=lambda arm: (
            candidates[arm]["quality_vs_q4"]["paired"][
                "psnr_vs_ground_truth"
            ]["candidate_minus_reference_mean"],
            candidates[arm]["quality_vs_q4"]["paired"][
                "edge_psnr_vs_ground_truth"
            ]["candidate_minus_reference_mean"],
            arm,
        ),
        reverse=True,
    )
    decision = {
        "format": FORMAT,
        "selection_source": "development",
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "candidates": candidates,
        "endpoint_candidate": endpoint,
        "approximately_1_58_endpoint_eligible": endpoint is not None,
        "next_stage_parent_arm": None if endpoint else ranked[0],
        "ranking": ranked,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "endpoint_candidate": endpoint,
        "next_stage_parent_arm": decision["next_stage_parent_arm"],
        "ranking": ranked,
    }, indent=2))


if __name__ == "__main__":
    main()
