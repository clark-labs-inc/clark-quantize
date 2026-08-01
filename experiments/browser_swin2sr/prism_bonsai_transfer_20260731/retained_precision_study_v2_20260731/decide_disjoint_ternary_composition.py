#!/usr/bin/env python3
"""Apply the preregistered gate to the disjoint ternary composition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "backdropkit-swin2sr-disjoint-ternary-composition-decision-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-disjoint-ternary-composition-protocol-v1"
REQUIRED_COMPARISONS = (
    "image_vs_parent",
    "rstb_vs_parent",
    "composition_vs_parent",
    "composition_vs_image",
    "composition_vs_rstb",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(comparison: dict) -> dict:
    metrics = comparison["metrics"]
    ground_truth = metrics["psnr_vs_ground_truth"]
    edge = metrics["edge_psnr_vs_ground_truth"]
    teacher = metrics["psnr_vs_teacher"]
    lower = float(ground_truth["paired_bootstrap_95ci"][0])
    edge_mean = float(edge["candidate_minus_reference_mean"])
    teacher_mean = float(teacher["candidate_minus_reference_mean"])
    checks = {
        "ground_truth_lower_95_positive": lower > 0.0,
        "edge_mean_nonnegative": edge_mean >= 0.0,
        "teacher_mean_nonnegative": teacher_mean >= 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ground_truth_lower_95": lower,
        "ground_truth_mean": float(ground_truth["candidate_minus_reference_mean"]),
        "edge_mean": edge_mean,
        "teacher_mean": teacher_mean,
    }


def verify_protocol_sources(protocol: dict, protocol_path: Path) -> None:
    root = protocol_path.parent
    paths = {
        "compose_disjoint_ternary_recovery.py": root
        / "compose_disjoint_ternary_recovery.py",
        "test_compose_disjoint_ternary_recovery.py": root
        / "test_compose_disjoint_ternary_recovery.py",
        "decide_disjoint_ternary_composition.py": root
        / "decide_disjoint_ternary_composition.py",
        "test_decide_disjoint_ternary_composition.py": root
        / "test_decide_disjoint_ternary_composition.py",
        "compare_evaluations.py": root
        / "sealed_overlay/experiments/browser_swin2sr/compare_evaluations.py",
        "evaluate_stratified.py": Path(
            "experiments/browser_swin2sr/scale_codebook_qat_v2_20260731/"
            "sealed_sources/experiments/browser_swin2sr/evaluate_stratified.py"
        ),
        "materialize_surface_states.py": Path(
            "experiments/browser_swin2sr/scale_codebook_qat_v2_20260731/"
            "sealed_sources/experiments/browser_swin2sr/materialize_surface_states.py"
        ),
        "bundle.json": Path(
            "/Users/stan/.cache/clark-quantize/"
            "swin2sr-heldout-bundle-v1/bundle.json"
        ),
    }
    if set(protocol["source_hashes"]) != set(paths):
        raise ValueError("sealed source hash inventory changed")
    for name, path in paths.items():
        if file_sha256(path) != protocol["source_hashes"][name]:
            raise ValueError(f"sealed source hash changed: {name}")
    source_study = Path(protocol["source_study"])
    if file_sha256(source_study) != protocol["source_study_sha256"]:
        raise ValueError("source study hash changed")
    parent = protocol["parent"]
    if file_sha256(Path(parent["state"])) != parent["state_file_sha256"]:
        raise ValueError("parent state hash changed")
    if file_sha256(Path(parent["materialization"])) != parent["materialization_sha256"]:
        raise ValueError("parent materialization hash changed")
    parent_report = json.loads(Path(parent["development_report"]).read_text())
    if parent_report["state_sha256"] != parent["state_file_sha256"]:
        raise ValueError("parent evaluation used a different state")
    image = protocol["sources"]["image_io_fp16"]
    if (
        file_sha256(Path(image["materialization"]))
        != image["materialization_sha256_at_seal"]
    ):
        raise ValueError("sealed image materialization hash changed")


def verify_composition(protocol: dict, protocol_path: Path) -> dict:
    composition_path = Path(protocol["composition_manifest"])
    composition = json.loads(composition_path.read_text())
    if composition.get("format") != "backdropkit-swin2sr-disjoint-ternary-composition-v1":
        raise ValueError("unsupported composition receipt")
    if composition.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("composition belongs to a different protocol")
    if Path(composition["protocol"]).resolve() != protocol_path.resolve():
        raise ValueError("composition protocol path changed")
    if composition.get("seed") != protocol["seed"]:
        raise ValueError("composition seed changed")
    if composition.get("storage_delta_bytes") != 0:
        raise ValueError("composition is not zero-storage")
    dense = composition["dense_state"]
    dense_path = Path(dense["path"])
    if file_sha256(dense_path) != dense["file_sha256"]:
        raise ValueError("composition dense file hash changed")
    report = json.loads(Path(protocol["composition_development_report"]).read_text())
    if report["state_sha256"] != dense["file_sha256"]:
        raise ValueError("composition evaluation used a different state")
    return composition


def verify_comparison(
    comparison: dict,
    reference_report_path: Path,
    candidate_report_path: Path,
) -> None:
    reference_report = json.loads(reference_report_path.read_text())
    candidate_report = json.loads(candidate_report_path.read_text())
    if Path(comparison["reference"]["path"]).resolve() != reference_report_path.resolve():
        raise ValueError("comparison reference path differs from protocol")
    if Path(comparison["candidate"]["path"]).resolve() != candidate_report_path.resolve():
        raise ValueError("comparison candidate path differs from protocol")
    if comparison["reference"]["state_sha256"] != reference_report["state_sha256"]:
        raise ValueError("comparison reference state hash differs from report")
    if comparison["candidate"]["state_sha256"] != candidate_report["state_sha256"]:
        raise ValueError("comparison candidate state hash differs from report")
    sample_counts = {
        int(comparison["metrics"][metric]["samples"])
        for metric in (
            "psnr_vs_ground_truth",
            "edge_psnr_vs_ground_truth",
            "psnr_vs_teacher",
        )
    }
    if sample_counts != {3040}:
        raise ValueError("composition comparison must cover all 3,040 samples")


def decide(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported composition protocol")
    if not protocol.get("final_heldout_locked"):
        raise ValueError("final heldout must remain locked")
    if protocol.get("model_promotion_authorized"):
        raise ValueError("development composition cannot authorize promotion")
    if set(protocol["comparisons"]) != set(REQUIRED_COMPARISONS):
        raise ValueError("composition protocol comparisons changed")
    verify_protocol_sources(protocol, protocol_path)
    composition = verify_composition(protocol, protocol_path)

    results = {}
    for name in REQUIRED_COMPARISONS:
        contract = protocol["comparisons"][name]
        comparison_path = Path(contract["output"])
        comparison = json.loads(comparison_path.read_text())
        verify_comparison(
            comparison,
            Path(contract["reference_report"]),
            Path(contract["candidate_report"]),
        )
        results[name] = {
            **gate(comparison),
            "comparison": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
        }
    accepted = all(result["passed"] for result in results.values())
    return {
        "format": FORMAT,
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "composition_manifest": protocol["composition_manifest"],
        "composition_manifest_sha256": file_sha256(
            Path(protocol["composition_manifest"])
        ),
        "composition_state_sha256": composition["dense_state"]["tensor_sha256"],
        "comparisons": results,
        "accepted_as_next_ternary_research_parent": accepted,
        "reason": (
            "both disjoint controls and their deterministic union passed every "
            "preregistered development non-regression and additivity gate"
            if accepted
            else "one or more preregistered development gates failed"
        ),
        "storage_delta_bytes": 0,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"composition decision already exists: {args.out}")
    decision = decide(args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "accepted": decision["accepted_as_next_ternary_research_parent"],
    }, indent=2))


if __name__ == "__main__":
    main()
