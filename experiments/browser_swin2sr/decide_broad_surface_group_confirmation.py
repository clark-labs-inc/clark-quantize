#!/usr/bin/env python3
"""Apply the frozen gate to broad surface-group confirmation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decide_surface_group_confirmation import passes_gate
from full_surface_provenance import file_sha256
from seal_broad_surface_group_confirmation import FORMAT as STUDY_FORMAT


FORMAT = "backdropkit-swin2sr-broad-surface-group-decision-v1"
REPORT_FORMAT = "backdropkit-swin2sr-disjoint-surface-group-sensitivity-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--existing-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def validate_report(report: dict, candidates: list[dict], study: dict) -> int:
    if report.get("format") != REPORT_FORMAT:
        raise RuntimeError("broad confirmation report has unsupported format")
    if report.get("model_promotion_authorized"):
        raise RuntimeError("broad confirmation improperly authorizes promotion")
    if not report.get("final_heldout_locked"):
        raise RuntimeError("broad confirmation did not lock final heldout")
    if report.get("checkpoint_sha256") != study["checkpoint_sha256"]:
        raise RuntimeError("broad confirmation changed checkpoint")
    if report.get("bundle_sha256") != study["bundle_sha256"]:
        raise RuntimeError("broad confirmation changed bundle")
    if report.get("sample_count") != study["confirmation"]["samples"]:
        raise RuntimeError("broad confirmation changed sample count")
    if report.get("bootstrap") != study["confirmation"]["bootstrap"]:
        raise RuntimeError("broad confirmation changed bootstrap contract")
    group_size = int(report["target_group_size"])
    expected_names = {
        row["name"]
        for row in candidates
        if row["group_size"] == group_size
    }
    actual_names = {row["name"] for row in report["surfaces"]}
    if actual_names != expected_names or len(actual_names) != len(
        report["surfaces"]
    ):
        raise RuntimeError("broad confirmation changed candidate surfaces")
    return group_size


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"broad surface decision exists: {args.out}")
    study = json.loads(args.study_manifest.read_text())
    if study.get("format") != STUDY_FORMAT:
        raise RuntimeError("unsupported broad surface confirmation study")
    existing = json.loads(args.existing_decision.read_text())
    existing_map = existing["selected_surface_group_sizes"]
    by_name = {row["name"]: row for row in study["candidates"]}
    reports = {}
    receipts = []
    for path in args.reports:
        report = json.loads(path.read_text())
        group_size = validate_report(report, study["candidates"], study)
        if group_size in reports:
            raise RuntimeError(f"duplicate broad confirmation: {group_size}")
        reports[group_size] = report
        receipts.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "group_size": group_size,
        })
    if set(reports) != {int(group) for group in study["commands"]}:
        raise RuntimeError("broad confirmation report set is incomplete")
    selected = []
    audit = []
    for group_size, report in sorted(reports.items()):
        for surface in report["surfaces"]:
            passed, failures = passes_gate(surface)
            candidate = by_name[surface["name"]]
            record = {
                "name": surface["name"],
                "group_size": group_size,
                "scale_bytes_delta": candidate["scale_bytes_delta"],
                "passed": passed,
                "failures": failures,
                "paired_evidence": surface["paired_evidence"],
            }
            audit.append(record)
            if passed:
                selected.append(record)
    selected.sort(key=lambda item: (
        -(
            item["paired_evidence"]["psnr_vs_ground_truth"]["mean"]
            / (item["scale_bytes_delta"] / 1024)
        ),
        item["scale_bytes_delta"],
        item["name"],
    ))
    added_bytes = sum(row["scale_bytes_delta"] for row in selected)
    total_bytes = study["existing_scale_bytes_delta"] + added_bytes
    if total_bytes > study["scale_byte_budget"]:
        raise RuntimeError("passing candidates exceed frozen byte budget")
    added_map = {row["name"]: row["group_size"] for row in selected}
    if set(added_map) & set(existing_map):
        raise RuntimeError("broad decision replaces an existing surface")
    expanded_map = dict(sorted({**existing_map, **added_map}.items()))
    decision = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "disjoint_development_confirmation",
        "model_promotion_authorized": False,
        "recovery_training_authorized": False,
        "final_heldout_locked": True,
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "existing_decision": str(args.existing_decision),
        "existing_decision_sha256": file_sha256(args.existing_decision),
        "reports": sorted(receipts, key=lambda row: row["group_size"]),
        "candidate_count": len(study["candidates"]),
        "passing_candidate_count": len(selected),
        "existing_surface_group_sizes": existing_map,
        "added_surface_group_sizes": added_map,
        "selected_surface_group_sizes": expanded_map,
        "existing_scale_bytes_delta": study["existing_scale_bytes_delta"],
        "added_scale_bytes_delta": added_bytes,
        "selected_scale_bytes_delta": total_bytes,
        "scale_byte_budget": study["scale_byte_budget"],
        "selected": selected,
        "audit": audit,
        "joint_evaluation_required": bool(selected),
        "decision": (
            "evaluate_expanded_map_jointly"
            if selected
            else "no_new_surface_group_candidate_passed"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "decision": decision["decision"],
        "passing_candidate_count": len(selected),
        "added_scale_bytes_delta": added_bytes,
        "selected_scale_bytes_delta": total_bytes,
        "added_surface_group_sizes": added_map,
    }, indent=2))


if __name__ == "__main__":
    main()
