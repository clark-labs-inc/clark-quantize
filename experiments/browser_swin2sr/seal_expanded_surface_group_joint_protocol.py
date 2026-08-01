#!/usr/bin/env python3
"""Preregister disjoint joint evaluation of an expanded surface map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decide_broad_surface_group_confirmation import FORMAT as DECISION_FORMAT
from full_surface_provenance import file_sha256


FORMAT = "backdropkit-swin2sr-expanded-surface-group-joint-protocol-v1"
EVALUATOR = (
    "experiments/browser_swin2sr/evaluate_expanded_surface_group_map.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"expanded joint protocol exists: {args.out}")
    decision = json.loads(args.decision.read_text())
    if decision.get("format") != DECISION_FORMAT:
        raise RuntimeError("unsupported broad surface decision")
    if not decision.get("joint_evaluation_required"):
        raise RuntimeError("broad decision has no map requiring evaluation")
    if decision.get("model_promotion_authorized"):
        raise RuntimeError("broad decision improperly authorizes promotion")
    if not decision.get("final_heldout_locked"):
        raise RuntimeError("broad decision did not lock final heldout")
    study_path = Path(decision["study_manifest"])
    if file_sha256(study_path) != decision["study_manifest_sha256"]:
        raise RuntimeError("broad confirmation study identity changed")
    repository = Path(__file__).resolve().parents[2]
    protocol = {
        "format": FORMAT,
        "status": "preregistered_before_joint_results",
        "diagnostic_only": True,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "decision": str(args.decision),
        "decision_sha256": file_sha256(args.decision),
        "study_manifest": str(study_path),
        "study_manifest_sha256": decision["study_manifest_sha256"],
        "evaluation_source": EVALUATOR,
        "evaluation_source_sha256": file_sha256(repository / EVALUATOR),
        "existing_surface_group_sizes": decision[
            "existing_surface_group_sizes"
        ],
        "added_surface_group_sizes": decision["added_surface_group_sizes"],
        "selected_surface_group_sizes": decision[
            "selected_surface_group_sizes"
        ],
        "selected_scale_bytes_delta": decision[
            "selected_scale_bytes_delta"
        ],
        "scale_byte_budget": decision["scale_byte_budget"],
        "sample_count": 256,
        "sample_source": (
            "balanced development samples excluding every screen, individual "
            "confirmation, and prior joint-evaluation sample"
        ),
        "bootstrap": {
            "method": "paired_nonparametric_mean",
            "resamples": 10000,
            "seed": 20260731,
            "confidence": 0.95,
        },
        "overall_and_incremental_gate": {
            "ground_truth_lower_95_db": "> 0",
            "edge_mean_delta_db": ">= -0.02",
            "teacher_mean_delta_db": ">= -0.1",
            "scale_byte_budget": f"<= {decision['scale_byte_budget']}",
        },
        "passing_result": "expanded map is eligible for matched recovery study",
        "failing_result": "retain the previously confirmed three-surface map",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "evaluation_source_sha256": protocol[
            "evaluation_source_sha256"
        ],
        "added_surface_count": len(protocol["added_surface_group_sizes"]),
        "selected_scale_bytes_delta": protocol[
            "selected_scale_bytes_delta"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
