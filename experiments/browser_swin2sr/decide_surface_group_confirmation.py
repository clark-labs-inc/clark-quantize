#!/usr/bin/env python3
"""Apply the frozen surface-group confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_surface_group_confirmation import FORMAT as STUDY_FORMAT


FORMAT = "backdropkit-swin2sr-surface-group-decision-v1"
REPORT_FORMAT = "backdropkit-swin2sr-surface-group-sensitivity-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def passes_gate(surface: dict) -> tuple[bool, list[str]]:
    evidence = surface["paired_evidence"]
    failures = []
    if evidence["psnr_vs_ground_truth"]["lower_95"] <= 0:
        failures.append("ground_truth_lower_95_not_positive")
    if evidence["edge_psnr_vs_ground_truth"]["mean"] < -0.02:
        failures.append("edge_mean_below_minus_0.02_db")
    if evidence["psnr_vs_teacher"]["mean"] < -0.1:
        failures.append("teacher_mean_below_minus_0.1_db")
    return not failures, failures


def choose_surface_options(
    reports: dict[int, dict],
    candidate_names: list[str],
    byte_budget: int,
) -> tuple[list[dict], list[dict], int]:
    by_group = {
        group_size: {row["name"]: row for row in report["surfaces"]}
        for group_size, report in reports.items()
    }
    choices = []
    audit = []
    for name in candidate_names:
        passing = []
        group_audit = []
        for group_size in sorted(by_group):
            row = by_group[group_size][name]
            passed, failures = passes_gate(row)
            scale_bytes = int(row["storage"]["scale_bytes_delta"])
            ground_truth = row["paired_evidence"][
                "psnr_vs_ground_truth"
            ]["mean"]
            efficiency = (
                ground_truth / (scale_bytes / 1024)
                if scale_bytes > 0
                else None
            )
            record = {
                "group_size": group_size,
                "passed": passed,
                "failures": failures,
                "scale_bytes_delta": scale_bytes,
                "ground_truth_db_per_added_kib": efficiency,
                "paired_evidence": {
                    metric: row["paired_evidence"][metric]
                    for metric in METRICS
                },
            }
            group_audit.append(record)
            if passed and scale_bytes > 0:
                passing.append(record)
        passing.sort(key=lambda item: (
            -item["ground_truth_db_per_added_kib"],
            item["scale_bytes_delta"],
            -item["group_size"],
        ))
        chosen = passing[0] if passing else None
        audit.append({
            "name": name,
            "groups": group_audit,
            "best_passing_group": chosen,
        })
        if chosen is not None:
            choices.append({"name": name, **chosen})
    choices.sort(key=lambda item: (
        -item["ground_truth_db_per_added_kib"],
        item["scale_bytes_delta"],
        item["name"],
    ))
    selected = []
    consumed = 0
    for choice in choices:
        if consumed + choice["scale_bytes_delta"] <= byte_budget:
            selected.append(choice)
            consumed += choice["scale_bytes_delta"]
    return selected, audit, consumed


def validate_report(
    report: dict,
    manifest: dict,
    candidate_names: list[str],
) -> int:
    if report.get("format") != REPORT_FORMAT:
        raise RuntimeError("confirmation report has an unsupported format")
    if report.get("model_promotion_authorized"):
        raise RuntimeError("confirmation report improperly authorizes promotion")
    if not report.get("final_heldout_locked"):
        raise RuntimeError("confirmation report did not lock final heldout")
    if report.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
        raise RuntimeError("confirmation report changed checkpoint")
    if report.get("bundle_sha256") != manifest["bundle_sha256"]:
        raise RuntimeError("confirmation report changed development bundle")
    expected = manifest["confirmation"]
    if report.get("sample_count") != expected["samples"]:
        raise RuntimeError("confirmation report changed sample count")
    if report.get("bootstrap") != expected["bootstrap"]:
        raise RuntimeError("confirmation report changed bootstrap contract")
    names = [row["name"] for row in report["surfaces"]]
    if set(names) != set(candidate_names) or len(names) != len(candidate_names):
        raise RuntimeError("confirmation report changed candidate surfaces")
    group_size = int(report["target_group_size"])
    if group_size not in expected["group_sizes"]:
        raise RuntimeError("confirmation report used an unregistered group size")
    return group_size


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"surface-group decision exists: {args.out}")
    manifest = json.loads(args.study_manifest.read_text())
    if manifest.get("format") != STUDY_FORMAT:
        raise RuntimeError("unsupported surface-group study format")
    if manifest.get("model_promotion_authorized"):
        raise RuntimeError("surface-group study improperly authorizes promotion")
    if not manifest.get("final_heldout_locked"):
        raise RuntimeError("surface-group study did not lock final heldout")
    candidate_names = [row["name"] for row in manifest["candidates"]]
    reports = {}
    report_receipts = []
    for path in args.reports:
        report = json.loads(path.read_text())
        group_size = validate_report(report, manifest, candidate_names)
        if group_size in reports:
            raise RuntimeError(f"duplicate group-size report: {group_size}")
        reports[group_size] = report
        report_receipts.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "group_size": group_size,
        })
    expected_groups = set(manifest["confirmation"]["group_sizes"])
    if set(reports) != expected_groups:
        raise RuntimeError("confirmation report set is incomplete")
    byte_budget = manifest["selection_rule"][
        "scale_byte_budget_at_group2"
    ]
    selected, audit, consumed = choose_surface_options(
        reports,
        candidate_names,
        byte_budget,
    )
    decision = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "development_confirmation",
        "model_promotion_authorized": False,
        "recovery_training_authorized": False,
        "final_heldout_locked": True,
        "study_manifest": str(args.study_manifest),
        "study_manifest_sha256": file_sha256(args.study_manifest),
        "decision_source_sha256": file_sha256(Path(__file__)),
        "reports": sorted(report_receipts, key=lambda row: row["group_size"]),
        "candidate_count": len(candidate_names),
        "passing_surface_count": len(selected),
        "scale_byte_budget": byte_budget,
        "selected_scale_bytes_delta": consumed,
        "selected_surface_group_sizes": {
            row["name"]: row["group_size"]
            for row in selected
        },
        "selected": selected,
        "audit": audit,
        "joint_candidate_required": bool(selected),
        "decision": (
            "evaluate_selected_map_jointly"
            if selected
            else "no_surface_group_candidate_passed"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "decision": decision["decision"],
        "passing_surface_count": len(selected),
        "selected_scale_bytes_delta": consumed,
        "selected_surface_group_sizes": decision[
            "selected_surface_group_sizes"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
