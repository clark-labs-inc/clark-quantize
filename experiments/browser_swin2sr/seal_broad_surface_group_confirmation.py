#!/usr/bin/env python3
"""Select and seal confirmation of broad group-3/group-6 surface screens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-broad-surface-group-confirmation-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-broad-surface-group-screen-v1"
SCREEN_FORMAT = "backdropkit-swin2sr-disjoint-surface-group-sensitivity-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--existing-decision", type=Path, required=True)
    parser.add_argument("--existing-joint-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale-byte-budget", type=int, default=65536)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def screen_options(
    reports: dict[int, dict],
    excluded_surfaces: set[str],
) -> list[dict]:
    by_surface: dict[str, list[dict]] = {}
    for group_size, report in reports.items():
        for surface in report["surfaces"]:
            if surface["name"] in excluded_surfaces:
                continue
            delta = surface["delta_vs_hard"]
            scale_bytes = int(surface["storage"]["scale_bytes_delta"])
            if (
                delta["psnr_vs_ground_truth"] > 0
                and delta["edge_psnr_vs_ground_truth"] > 0
                and delta["psnr_vs_teacher"] >= -0.1
                and scale_bytes > 0
            ):
                by_surface.setdefault(surface["name"], []).append({
                    "name": surface["name"],
                    "category": surface["category"],
                    "group_size": group_size,
                    "scale_bytes_delta": scale_bytes,
                    "screen_delta_vs_hard": delta,
                    "ground_truth_db_per_added_kib": (
                        delta["psnr_vs_ground_truth"]
                        / (scale_bytes / 1024)
                    ),
                })
    options = []
    for candidates in by_surface.values():
        candidates.sort(key=lambda item: (
            -item["ground_truth_db_per_added_kib"],
            item["scale_bytes_delta"],
            -item["group_size"],
        ))
        options.append(candidates[0])
    options.sort(key=lambda item: (
        -item["ground_truth_db_per_added_kib"],
        item["scale_bytes_delta"],
        item["name"],
    ))
    return options


def choose_with_budget(
    options: list[dict],
    existing_bytes: int,
    budget: int,
) -> tuple[list[dict], int]:
    if existing_bytes > budget:
        raise ValueError("existing surface map already exceeds byte budget")
    selected = []
    consumed = existing_bytes
    for option in options:
        if consumed + option["scale_bytes_delta"] <= budget:
            selected.append(option)
            consumed += option["scale_bytes_delta"]
    return selected, consumed


def confirmation_command(
    protocol: dict,
    out: Path,
    group_size: int,
    candidates: list[dict],
    exclusions: list[Path],
    samples: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[str]:
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repository / protocol["analysis_source"]),
        protocol["checkpoint"],
        "--bundle", protocol["bundle"],
        "--out", str(out / f"group{group_size}-confirmation-{samples}.json"),
        "--target-group-size", str(group_size),
        "--method", "current",
        "--max-samples", str(samples),
        "--bootstrap-resamples", str(bootstrap_resamples),
        "--bootstrap-seed", str(bootstrap_seed),
        "--device", "cpu",
        "--torch-threads", "1",
    ]
    for path in exclusions:
        command.extend(("--exclude-report", str(path)))
    for candidate in candidates:
        command.extend(("--surface", candidate["name"]))
    return command


def validate_screen(report: dict, protocol: dict) -> int:
    if report.get("format") != SCREEN_FORMAT:
        raise RuntimeError("broad screen has an unsupported format")
    if report.get("sample_count") != protocol["sample_count_per_group"]:
        raise RuntimeError("broad screen changed sample count")
    if report.get("surface_count") != protocol["surface_count"]:
        raise RuntimeError("broad screen changed surface count")
    if report.get("checkpoint_sha256") != protocol["checkpoint_sha256"]:
        raise RuntimeError("broad screen changed checkpoint")
    if report.get("bundle_sha256") != protocol["bundle_sha256"]:
        raise RuntimeError("broad screen changed bundle")
    if report.get("model_promotion_authorized"):
        raise RuntimeError("broad screen improperly authorizes promotion")
    if not report.get("final_heldout_locked"):
        raise RuntimeError("broad screen did not lock final heldout")
    group_size = int(report["target_group_size"])
    if group_size not in protocol["target_group_sizes"]:
        raise RuntimeError("broad screen used an unregistered group size")
    return group_size


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"broad confirmation exists: {args.out}")
    if min(
        args.scale_byte_budget,
        args.samples,
        args.bootstrap_resamples,
    ) < 1:
        raise ValueError("budget, samples, and resamples must be positive")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported broad-screen protocol")
    repository = Path(__file__).resolve().parents[2]
    if file_sha256(repository / protocol["analysis_source"]) != (
        protocol["analysis_source_sha256"]
    ):
        raise RuntimeError("broad-screen analysis source changed")
    reports = {}
    screen_receipts = []
    for path in args.screens:
        report = json.loads(path.read_text())
        group_size = validate_screen(report, protocol)
        if group_size in reports:
            raise RuntimeError(f"duplicate group-size screen: {group_size}")
        reports[group_size] = report
        screen_receipts.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "group_size": group_size,
        })
    if set(reports) != set(protocol["target_group_sizes"]):
        raise RuntimeError("broad-screen report set is incomplete")
    existing = json.loads(args.existing_decision.read_text())
    joint = json.loads(args.existing_joint_report.read_text())
    if not joint.get("recovery_study_eligible"):
        raise RuntimeError("existing surface map did not pass joint confirmation")
    if joint.get("decision_sha256") != file_sha256(args.existing_decision):
        raise RuntimeError("existing joint report uses another decision")
    existing_map = existing["selected_surface_group_sizes"]
    existing_bytes = int(joint["storage"]["selected_scale_bytes_delta"])
    options = screen_options(reports, set(existing_map))
    selected, consumed = choose_with_budget(
        options,
        existing_bytes,
        args.scale_byte_budget,
    )
    if not selected:
        raise RuntimeError("broad screens produced no confirmation candidates")
    exclusions = [
        Path(record["path"])
        for record in protocol["excluded_reports"]
    ] + list(args.screens)
    by_group = {
        group_size: [
            candidate for candidate in selected
            if candidate["group_size"] == group_size
        ]
        for group_size in protocol["target_group_sizes"]
    }
    commands = {
        str(group_size): confirmation_command(
            protocol,
            args.out,
            group_size,
            candidates,
            exclusions,
            args.samples,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        for group_size, candidates in by_group.items()
        if candidates
    }
    manifest = {
        "format": FORMAT,
        "diagnostic_only": True,
        "selection_source": "disjoint_development_confirmation",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "screens": sorted(screen_receipts, key=lambda row: row["group_size"]),
        "checkpoint": protocol["checkpoint"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "bundle": protocol["bundle"],
        "bundle_sha256": protocol["bundle_sha256"],
        "existing_surface_group_sizes": existing_map,
        "existing_scale_bytes_delta": existing_bytes,
        "scale_byte_budget": args.scale_byte_budget,
        "selected_total_scale_bytes_delta": consumed,
        "candidate_count": len(selected),
        "candidates": selected,
        "confirmation": {
            "samples": args.samples,
            "excluded_reports": [str(path) for path in exclusions],
            "bootstrap": {
                "method": "paired_nonparametric_mean",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence": 0.95,
            },
            "individual_pass_gate": {
                "ground_truth_lower_95_db": "> 0",
                "edge_mean_delta_db": ">= -0.02",
                "teacher_mean_delta_db": ">= -0.1",
            },
        },
        "commands": commands,
    }
    args.out.mkdir(parents=True)
    manifest_path = args.out / "study-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate_count": len(selected),
        "candidates_by_group": {
            str(group): len(candidates)
            for group, candidates in by_group.items()
        },
        "selected_total_scale_bytes_delta": consumed,
    }, indent=2))


if __name__ == "__main__":
    main()
