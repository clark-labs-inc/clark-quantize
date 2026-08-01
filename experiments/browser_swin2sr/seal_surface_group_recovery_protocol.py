#!/usr/bin/env python3
"""Preregister the matched surface-group recovery before parent selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_surface_provenance import file_sha256
from seal_surface_group_recovery_study import (
    PROTOCOL_FORMAT,
    SOURCE_FILES,
)


SURFACE_FORMAT = "backdropkit-swin2sr-surface-group-decision-v1"
JOINT_FORMAT = "backdropkit-swin2sr-surface-group-joint-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-decision", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--optimizer-decision", type=Path, required=True)
    parser.add_argument("--objective-decision", type=Path, required=True)
    parser.add_argument("--objective-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"recovery protocol exists: {args.out}")
    surface = json.loads(args.surface_decision.read_text())
    joint = json.loads(args.joint_report.read_text())
    if surface.get("format") != SURFACE_FORMAT:
        raise RuntimeError("unsupported surface decision")
    if joint.get("format") != JOINT_FORMAT:
        raise RuntimeError("unsupported joint report")
    if not joint.get("recovery_study_eligible"):
        raise RuntimeError("joint surface map is not recovery eligible")
    if joint.get("decision_sha256") != file_sha256(args.surface_decision):
        raise RuntimeError("joint report uses another surface decision")
    if (
        joint["selected_surface_group_sizes"]
        != surface["selected_surface_group_sizes"]
    ):
        raise RuntimeError("joint report changed selected surface map")
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {
        relative: file_sha256(repository / relative)
        for relative in SOURCE_FILES
    }
    protocol = {
        "format": PROTOCOL_FORMAT,
        "status": "preregistered_before_parent_optimizer_or_objective_selection",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "surface_decision": str(args.surface_decision),
        "surface_decision_sha256": file_sha256(args.surface_decision),
        "joint_report": str(args.joint_report),
        "joint_report_sha256": file_sha256(args.joint_report),
        "optimizer_decision": str(args.optimizer_decision),
        "objective_decision": str(args.objective_decision),
        "objective_study": str(args.objective_study),
        "parent_selection": (
            "use q4-eligible optimizer winner when present; otherwise use "
            "the frozen pixel-objective winner"
        ),
        "steps_after_parent": 2000,
        "device": "mps",
        "selected_surface_group_sizes": surface[
            "selected_surface_group_sizes"
        ],
        "selected_scale_bytes_delta": joint["storage"][
            "selected_scale_bytes_delta"
        ],
        "arms": {
            "control": "exact parent continuation",
            "refined": "same continuation plus selected surface group map",
        },
        "decision_rule": {
            "bootstrap_seed": 20260731,
            "bootstrap_samples": 10000,
            "ground_truth_lower_95_min_db": 0.0,
            "edge_mean_min_db": -0.1,
            "teacher_mean_min_db": -0.1,
            "every_source_mean_min_db": -0.1,
            "scale_byte_budget": 65536,
        },
        "source_sha256": source_hashes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "source_count": len(source_hashes),
        "selected_surface_group_sizes": protocol[
            "selected_surface_group_sizes"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
