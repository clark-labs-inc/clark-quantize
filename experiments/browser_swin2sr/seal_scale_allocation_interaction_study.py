#!/usr/bin/env python3
"""Seal the combined-output gate for MixFrag per-surface allocations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from full_surface_provenance import file_sha256
from scale_allocation_interactions import precision_map_storage, uniform_map


FORMAT = "backdropkit-swin2sr-scale-allocation-interaction-study-v1"
SOURCES = (
    "experiments/browser_swin2sr/evaluate_scale_allocation_interactions.py",
    "experiments/browser_swin2sr/scale_allocation_interactions.py",
    "experiments/browser_swin2sr/seal_scale_allocation_interaction_study.py",
    "experiments/browser_swin2sr/mixfrag_scale_allocation_20260731/INTERACTION_PROTOCOL.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-study", type=Path, required=True)
    parser.add_argument("--allocation-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def storage_erratum(study: dict, report: dict) -> dict:
    q8 = precision_map_storage(report, uniform_map(report, "q8"))
    expected = int(study["expected_uniform_storage_bytes"]["q8"])
    observed = int(q8["calibration_storage_bytes"])
    return {
        "affected_field": "expected_uniform_storage_bytes.q8",
        "sealed_preflight_estimate_bytes": expected,
        "observed_calibration_bytes": observed,
        "difference_bytes": observed - expected,
        "browser_physical_aligned_span_bytes": int(
            q8["physical_aligned_span_bytes"]
        ),
        "disposition": (
            "The observed per-surface receipts override the four-byte preflight "
            "estimate. q8 is a diagnostic control, not an advancing allocation."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"interaction study exists: {args.out}")
    study = json.loads(args.allocation_study.read_text())
    report = json.loads(args.allocation_report.read_text())
    if study.get("format") != "backdropkit-swin2sr-output-fragility-study-v1":
        raise ValueError("unsupported allocation study")
    if report.get("format") != "backdropkit-swin2sr-output-fragility-calibration-v1":
        raise ValueError("unsupported allocation report")
    if (
        report.get("development_accessed")
        or report.get("final_heldout_accessed")
        or report.get("selection_source") != "training_calibration_only"
    ):
        raise RuntimeError("allocation report is not isolated training evidence")
    allocation_manifest = Path(report["manifest"])
    if file_sha256(allocation_manifest) != report["manifest_sha256"]:
        raise RuntimeError("allocation calibration manifest changed")
    parent = Path(report["parent"])
    if file_sha256(parent) != report["parent_sha256"]:
        raise RuntimeError("allocation parent changed")
    for budget in (10012, 20024):
        allocation = report["allocations"].get(str(budget))
        if not allocation or int(allocation["storage_bytes"]) != budget:
            raise RuntimeError(f"allocation budget is unavailable: {budget}")
    repository = Path(__file__).resolve().parents[2]
    sealed_root = args.out / "sealed_sources"
    for relative in SOURCES:
        source = repository / relative
        destination = sealed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    allocation_overlay = args.allocation_study.parent / "sealed_overlay"
    base_sources = Path(study["base_sealed_sources"])
    pythonpath = ":".join((
        str(sealed_root / "experiments/browser_swin2sr"),
        str(allocation_overlay / "experiments/browser_swin2sr"),
        str(base_sources / "experiments/browser_swin2sr"),
        str(base_sources),
    ))
    result_path = args.out / "result.json"
    protocol_path = args.out / "protocol.json"
    candidates = {
        "mixfrag_10012": {
            "budget_bytes": 10012,
            "uniform_control": "uniform_q2",
        },
        "mixfrag_20024": {
            "budget_bytes": 20024,
            "uniform_control": "uniform_q4",
        },
    }
    candidate_storage = {
        name: precision_map_storage(
            report, report["allocations"][str(spec["budget_bytes"])]["choices"]
        )
        for name, spec in candidates.items()
    }
    for name, storage in candidate_storage.items():
        budget = candidates[name]["budget_bytes"]
        if (
            storage["calibration_storage_bytes"] != budget
            or storage["physical_aligned_span_bytes"] > budget
        ):
            raise RuntimeError(f"candidate violates physical byte budget: {name}")
    command = [
        "/usr/bin/env",
        f"PYTHONPATH={pythonpath}",
        sys.executable,
        str(
            sealed_root
            / "experiments/browser_swin2sr/evaluate_scale_allocation_interactions.py"
        ),
        "--protocol", str(protocol_path),
        "--out", str(result_path),
        "--device", "mps",
    ]
    protocol = {
        "format": FORMAT,
        "status": "sealed_before_combined_interaction_results",
        "selection_source": "training_only",
        "model_promotion_authorized": False,
        "development_access_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "allocation_study_manifest": str(args.allocation_study),
        "allocation_study_manifest_sha256": file_sha256(args.allocation_study),
        "allocation_report": str(args.allocation_report),
        "allocation_report_sha256": file_sha256(args.allocation_report),
        "allocation_manifest": str(allocation_manifest),
        "allocation_manifest_sha256": file_sha256(allocation_manifest),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "training_data": report["training_data"],
        "allocation_samples": int(report["calibration_samples"]),
        "confirmation_samples": 64,
        "confirmation_seed": 20260732,
        "confirmation_disjoint_from_allocation": True,
        "batch_size": 8,
        "crop_size": int(report["crop_size"]),
        "uniform_controls": ["q2", "q4", "fp16"],
        "candidates": candidates,
        "candidate_storage_preflight": candidate_storage,
        "minimum_relative_gain": 0.10,
        "advancement_rule": (
            "Select the smallest exact-byte candidate with at least 10% lower "
            "combined output fragility than its matched uniform control on both "
            "the allocation shard and a disjoint training-only confirmation shard."
        ),
        "interaction_diagnostic": (
            "Measured combined fragility divided by summed isolated fragility; "
            "measured combined output overrides the additive prediction."
        ),
        "storage_erratum": storage_erratum(study, report),
        "sealed_source_root": str(sealed_root.resolve()),
        "source_sha256": {
            relative: file_sha256(sealed_root / relative) for relative in SOURCES
        },
        "evaluation_command": command,
        "result": str(result_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "candidate_storage": candidate_storage,
        "storage_erratum": protocol["storage_erratum"],
        "evaluation_command": command,
    }, indent=2))


if __name__ == "__main__":
    main()
