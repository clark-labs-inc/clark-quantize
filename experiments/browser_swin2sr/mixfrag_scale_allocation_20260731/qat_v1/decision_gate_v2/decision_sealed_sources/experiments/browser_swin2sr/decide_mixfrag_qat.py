#!/usr/bin/env python3
"""Apply the preregistered quality and storage gates to MixFrag QAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_promotion import indexed_samples, paired_gate, stratified_gate
from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-mixfrag-scale-qat-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-mixfrag-scale-qat-decision-protocol-v1"
CHECKPOINT_FORMAT = "backdropkit-swin2sr-scale-codebook-checkpoint-v1"
MATERIALIZATION_FORMAT = "backdropkit-swin2sr-scale-codebook-materialization-v1"


def exact_input(record: dict) -> Path:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"sealed decision input changed: {record['name']}")
    return path


def quality_gate(reference: dict, candidate: dict, policy: dict) -> dict:
    reference_samples = indexed_samples(reference)
    candidate_samples = indexed_samples(candidate)
    if reference_samples.keys() != candidate_samples.keys() or len(reference_samples) != 3040:
        raise RuntimeError("quality reports do not cover the exact 3,040 samples")
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
            all(item["passed"] for item in paired.values())
            and all(item["passed"] for item in stratified)
        ),
    }


def means(report: dict) -> dict[str, float]:
    return {
        metric: float(report["overall"][metric]["mean"])
        for metric in (
            "psnr_vs_ground_truth",
            "edge_psnr_vs_ground_truth",
            "psnr_vs_teacher",
            "psnr_gain_over_bicubic",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"MixFrag QAT decision exists: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported MixFrag QAT decision protocol")
    if (
        protocol.get("model_promotion_authorized")
        or not protocol.get("final_heldout_locked")
    ):
        raise RuntimeError("decision protocol violates promotion isolation")
    for relative, expected in protocol["source_sha256"].items():
        if file_sha256(Path(protocol["sealed_source_root"]) / relative) != expected:
            raise RuntimeError(f"sealed decision source changed: {relative}")
    inputs = {
        record["name"]: exact_input(record) for record in protocol["inputs"]
    }
    study = json.loads(inputs["study_manifest"].read_text())
    target_map = json.loads(inputs["target_map"].read_text())
    checkpoint_path = Path(study["final_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or int(checkpoint.get("scale_step", -1)) != int(study["steps"])
        or checkpoint.get("parent_checkpoint_sha256") != study["parent_checkpoint_sha256"]
        or checkpoint["scale_codebook"].get("target_bits_by_surface")
        != target_map["target_bits_by_surface"]
    ):
        raise RuntimeError("trained checkpoint differs from the sealed study")
    materialization_path = Path(study["materialization"])
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("format") != MATERIALIZATION_FORMAT
        or materialization.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise RuntimeError("materialization differs from the trained checkpoint")
    candidate = json.loads(Path(study["development_report"]).read_text())
    dense = materialization["dense_state"]
    if (
        candidate["state_sha256"] != dense["file_sha256"]
        or Path(candidate["state"]).resolve() != Path(dense["path"]).resolve()
    ):
        raise RuntimeError("development report evaluates another dense state")
    artifact_root = Path(study["artifact"])
    artifact_manifest_path = artifact_root / "model.json"
    artifact_binary_path = artifact_root / "model.bin"
    artifact = json.loads(artifact_manifest_path.read_text())
    if artifact.get("materialization_manifest_sha256") != file_sha256(materialization_path):
        raise RuntimeError("browser artifact uses another materialization")
    storage = artifact["storage_ledger"]
    storage_passed = bool(
        int(storage["scale_bytes"]) == int(protocol["exact_scale_bytes"])
        and int(storage["binary_bytes"]) == artifact_binary_path.stat().st_size
    )
    policy = json.loads(inputs["promotion_policy"].read_text())
    q4 = json.loads(inputs["q4_reference"].read_text())
    parent = json.loads(inputs["refined_parent_report"].read_text())
    released = json.loads(inputs["released_ternary_report"].read_text())
    bridge = json.loads(inputs["sensitive_bridge_report"].read_text())
    leaders = [
        json.loads(inputs[f"all_ternary_leader_seed_{seed}"].read_text())
        for seed in protocol["leader_seeds"]
    ]
    candidate_means = means(candidate)
    references = {
        "q4": means(q4),
        "refined_parent": means(parent),
        "released_ternary": means(released),
        "sensitive_bridge": means(bridge),
        "all_ternary_leader_seed_mean": {
            metric: float(np.mean([means(report)[metric] for report in leaders]))
            for metric in candidate_means
        },
    }
    deltas = {
        name: {
            metric: candidate_means[metric] - values[metric]
            for metric in candidate_means
        }
        for name, values in references.items()
    }
    parent_gate = quality_gate(parent, candidate, policy)
    q4_gate = quality_gate(q4, candidate, policy)
    mechanism_passed = bool(
        storage_passed
        and parent_gate["passed"]
        and deltas["released_ternary"]["psnr_vs_ground_truth"] > 0
        and deltas["sensitive_bridge"]["psnr_vs_ground_truth"] > 0
    )
    leader_passed = bool(
        mechanism_passed
        and deltas["all_ternary_leader_seed_mean"]["psnr_vs_ground_truth"] > 0
        and deltas["all_ternary_leader_seed_mean"]["edge_psnr_vs_ground_truth"] > 0
    )
    decision = {
        "format": FORMAT,
        "selection_source": "development",
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "materialization": str(materialization_path),
        "materialization_sha256": file_sha256(materialization_path),
        "development_report": study["development_report"],
        "development_report_sha256": file_sha256(Path(study["development_report"])),
        "artifact_manifest": str(artifact_manifest_path),
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "artifact_binary": str(artifact_binary_path),
        "artifact_binary_sha256": file_sha256(artifact_binary_path),
        "storage_ledger": storage,
        "storage_passed": storage_passed,
        "candidate_means": candidate_means,
        "reference_means": references,
        "candidate_minus_reference": deltas,
        "noninferiority_vs_refined_parent": parent_gate,
        "official_quality_vs_q4": q4_gate,
        "mechanism_advances": mechanism_passed,
        "new_all_ternary_quality_leader": leader_passed,
        "next_action": (
            "runtime_and_robustness_gates"
            if leader_passed and q4_gate["passed"]
            else "transfer_allocator_to_strongest_all_ternary_parent"
            if mechanism_passed
            else "reject_or_revise_scale_qat"
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "storage_passed": storage_passed,
        "mechanism_advances": mechanism_passed,
        "new_all_ternary_quality_leader": leader_passed,
        "next_action": decision["next_action"],
    }, indent=2))


if __name__ == "__main__":
    main()
