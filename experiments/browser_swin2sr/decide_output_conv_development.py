#!/usr/bin/env python3
"""Select an output-convolution group from full development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decide_sensitive_io_development import (
    indexed_samples,
    load_report,
    paired_analysis,
    report_identity,
)
from full_surface_provenance import file_sha256


TARGET_SURFACE = "upsample.conv"
EXPECTED_GROUPS = {2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--source-margin-db", type=float, default=0.10)
    parser.add_argument("--mean-margin-db", type=float, default=0.10)
    return parser.parse_args()


def group_size(report: dict) -> int:
    sizes = report.get("surface_group_sizes") or {}
    if TARGET_SURFACE not in sizes:
        raise RuntimeError(f"report has no {TARGET_SURFACE} group override")
    return int(sizes[TARGET_SURFACE])


def base_surface_map(report: dict) -> dict[str, int]:
    return {
        name: int(size)
        for name, size in (report.get("surface_group_sizes") or {}).items()
        if name != TARGET_SURFACE
    }


def require_all_hard(report: dict, expected_step: int) -> None:
    if int(report["surface_step"]) != expected_step:
        raise RuntimeError("output report is not at the sealed selection step")
    scheduled = report["scheduled_surface_summary"]
    if (
        int(scheduled["surfaces"]) != 204
        or int(scheduled["fully_hard_surfaces"]) != 204
        or int(scheduled["transitioning_surfaces"]) != 0
    ):
        raise RuntimeError("output report is not a fully hard 204-surface state")


def overall_mean(report: dict, metric: str) -> float:
    return float(report["overall"][metric]["mean"])


def decide(
    gate_path: Path,
    control_path: Path,
    candidate_paths: list[Path],
    q4_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    source_margin_db: float,
    mean_margin_db: float,
) -> dict:
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("format")
        != "backdropkit-swin2sr-output-conv-development-gate-v1"
    ):
        raise RuntimeError("unsupported output-convolution development gate")
    if gate.get("selection_source") != "development":
        raise RuntimeError("output gate is not development-only")
    if gate.get("model_promotion_authorized"):
        raise RuntimeError("output gate improperly authorizes promotion")
    if not gate.get("final_heldout_locked"):
        raise RuntimeError("output gate does not lock final heldout")
    if gate["q4_reference_sha256"] != file_sha256(q4_path):
        raise RuntimeError("output gate q4 reference differs")
    frozen = {
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "source_margin_db": source_margin_db,
        "mean_margin_db": mean_margin_db,
    }
    for name, value in frozen.items():
        if gate[name] != value:
            raise RuntimeError(f"selector differs from sealed {name}")
    if set(map(int, gate["candidate_group_sizes"])) != EXPECTED_GROUPS:
        raise RuntimeError("output gate does not seal groups 2 and 3")
    if int(gate["control_group_size"]) != 9:
        raise RuntimeError("output gate control is not group 9")

    expected_step = int(gate["selection_step"])
    gate_hash = file_sha256(gate_path)
    q4 = json.loads(q4_path.read_text())
    control = load_report(control_path)
    require_all_hard(control, expected_step)
    if control["gate_manifest_sha256"] != gate_hash:
        raise RuntimeError("control was not evaluated under the output gate")
    if group_size(control) != 9:
        raise RuntimeError("output control does not use group 9")
    if indexed_samples(control).keys() != indexed_samples(q4).keys():
        raise RuntimeError("output control and q4 do not share exact samples")
    identity = report_identity(control)
    shared_surface_map = base_surface_map(control)
    expected_parent = gate["parent_checkpoint_sha256"]
    control_parent = (
        control.get("fork_provenance") or {}
    ).get("parent_checkpoint_sha256")
    if control_parent != expected_parent:
        raise RuntimeError("output control does not descend from sealed parent")

    if len(candidate_paths) != len(EXPECTED_GROUPS):
        raise RuntimeError("output gate requires exactly groups 2 and 3")
    rng = np.random.default_rng(seed)
    loaded = []
    seen = set()
    control_gt = overall_mean(control, "psnr_vs_ground_truth")
    control_teacher = overall_mean(control, "psnr_vs_teacher")
    for path in candidate_paths:
        report = load_report(path)
        require_all_hard(report, expected_step)
        if report_identity(report) != identity:
            raise RuntimeError(f"output candidate protocol differs: {path}")
        current_group = group_size(report)
        if current_group not in EXPECTED_GROUPS or current_group in seen:
            raise RuntimeError(f"unexpected output candidate group: {path}")
        seen.add(current_group)
        if base_surface_map(report) != shared_surface_map:
            raise RuntimeError("output candidates differ outside target surface")
        parent = (
            report.get("fork_provenance") or {}
        ).get("parent_checkpoint_sha256")
        if parent != expected_parent:
            raise RuntimeError(f"output candidate parent differs: {path}")
        gt = overall_mean(report, "psnr_vs_ground_truth")
        teacher = overall_mean(report, "psnr_vs_teacher")
        loaded.append({
            "_report": report,
            "report": str(path),
            "report_sha256": file_sha256(path),
            "checkpoint": report["state"],
            "checkpoint_sha256": report["state_sha256"],
            "output_group_size": current_group,
            "development_ground_truth_db": gt,
            "development_teacher_db": teacher,
            "beats_control_ground_truth": gt > control_gt,
            "beats_control_teacher": teacher > control_teacher,
            "versus_control": paired_analysis(
                report,
                control,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "versus_q4": paired_analysis(
                report,
                q4,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
        })
    if seen != EXPECTED_GROUPS:
        raise RuntimeError("output reports do not cover groups 2 and 3")

    control_beaters = [
        arm
        for arm in loaded
        if arm["beats_control_ground_truth"] and arm["beats_control_teacher"]
    ]
    best = (
        max(
            control_beaters,
            key=lambda arm: arm["development_ground_truth_db"],
        )
        if control_beaters
        else None
    )
    for arm in loaded:
        versus_best = (
            paired_analysis(
                arm["_report"],
                best["_report"],
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            if best
            else None
        )
        within_mean = (
            arm["development_ground_truth_db"]
            >= best["development_ground_truth_db"] - mean_margin_db
            if best
            else False
        )
        every_source = (
            versus_best["worst_source_psnr_delta"] >= -source_margin_db
            if versus_best
            else False
        )
        arm["versus_best_development_arm"] = versus_best
        arm["gates"] = {
            "beats_control_ground_truth": arm["beats_control_ground_truth"],
            "beats_control_teacher": arm["beats_control_teacher"],
            "overall_within_best_margin": within_mean,
            "every_source_within_best_margin": every_source,
        }
        arm["eligible"] = all(arm["gates"].values())
    eligible = sorted(
        (arm for arm in loaded if arm["eligible"]),
        key=lambda arm: (
            -arm["output_group_size"],
            -arm["development_ground_truth_db"],
        ),
    )
    selected = eligible[0] if eligible else None
    return {
        "format": "backdropkit-swin2sr-output-conv-decision-v1",
        "selection_source": "development",
        "gate_manifest": str(gate_path),
        "gate_manifest_sha256": gate_hash,
        "control_report": str(control_path),
        "control_report_sha256": file_sha256(control_path),
        "q4_reference": str(q4_path),
        "q4_reference_sha256": file_sha256(q4_path),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "source_margin_db": source_margin_db,
        "mean_margin_db": mean_margin_db,
        "parent_checkpoint_sha256": expected_parent,
        "shared_non_output_surface_map": shared_surface_map,
        "control_ground_truth_db": control_gt,
        "control_teacher_db": control_teacher,
        "best_development_group_size": (
            best["output_group_size"] if best else None
        ),
        "candidates": sorted(
            (
                {key: value for key, value in arm.items() if key != "_report"}
                for arm in loaded
            ),
            key=lambda arm: arm["output_group_size"],
        ),
        "selected_output_group_size": (
            selected["output_group_size"] if selected else None
        ),
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": (
            selected["checkpoint_sha256"] if selected else None
        ),
        "output_recipe_selected": selected is not None,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "required_next_gate": (
            "Run the preregistered full-surface optimizer-transfer study."
            if selected
            else "No output-convolution recipe passed; revise training."
        ),
    }


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_manifest.read_text())
    repository = Path(__file__).resolve().parents[2]
    for relative, expected in gate.get("source_files", {}).items():
        if file_sha256(repository / relative) != expected:
            raise RuntimeError(f"output-gate source changed: {relative}")
    receipt = decide(
        args.gate_manifest,
        args.control,
        args.candidates,
        args.q4_reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        source_margin_db=args.source_margin_db,
        mean_margin_db=args.mean_margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "output_recipe_selected": receipt["output_recipe_selected"],
        "selected_output_group_size": receipt["selected_output_group_size"],
        "model_promotion_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
