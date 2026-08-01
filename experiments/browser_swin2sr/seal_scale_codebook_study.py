#!/usr/bin/env python3
"""Preregister a disjoint development study for compressed ternary scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_stratified import file_sha256, load_bundle_records
from materialize_surface_states import load_materialized_state
from residual_gain import balanced_folds


FORMAT = "backdropkit-swin2sr-scale-codebook-protocol-v1"
SOURCE_FILES = (
    "experiments/browser_swin2sr/scale_codebook.py",
    "experiments/browser_swin2sr/evaluate_scale_codebooks.py",
    "experiments/browser_swin2sr/evaluate_stratified.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/residual_gain.py",
    "experiments/browser_swin2sr/rewrite_onnx_ternary.py",
    "experiments/browser_swin2sr/rotation_control.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--prior-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sample_receipts(rows: list[dict]) -> list[dict[str, str]]:
    return [
        {"sha256": str(row["sha256"]), "source": str(row["source"])}
        for row in rows
    ]


def source_counts(rows: list[dict]) -> dict[str, int]:
    return {
        source: sum(str(row["source"]) == source for row in rows)
        for source in sorted({str(row["source"]) for row in rows})
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"scale-codebook protocol exists: {args.out}")
    repository = Path(__file__).resolve().parents[2]
    materialization, _runtime = load_materialized_state(
        args.materialization, "runtime_state"
    )
    _dense_manifest, _dense = load_materialized_state(
        args.materialization, "dense_state"
    )
    bundle, records = load_bundle_records(args.bundle)
    q4 = json.loads(args.q4_reference.read_text())
    if q4.get("bundle_manifest_sha256") != bundle["manifest_sha256"]:
        raise RuntimeError("q4 reference and development bundle differ")
    prior = json.loads(args.prior_protocol.read_text())
    if prior.get("bundle_sha256") != file_sha256(args.bundle):
        raise RuntimeError("prior protocol used a different development bundle")
    prior_screen, prior_confirmation = balanced_folds(
        records,
        screen_per_source=int(prior["screen_per_source"]),
        confirmation_per_source=int(prior["confirmation_per_source"]),
        seed=int(prior["fold_seed"]),
    )
    excluded = {
        str(row["sha256"])
        for row in prior_screen + prior_confirmation
    }
    available = [row for row in records if str(row["sha256"]) not in excluded]
    screen, confirmation = balanced_folds(
        available,
        screen_per_source=32,
        confirmation_per_source=64,
        seed=20260801,
    )
    screen_ids = {row["sha256"] for row in screen}
    confirmation_ids = {row["sha256"] for row in confirmation}
    if screen_ids & confirmation_ids or (screen_ids | confirmation_ids) & excluded:
        raise RuntimeError("scale-codebook folds are not disjoint")
    protocol = {
        "format": FORMAT,
        "status": "preregistered_before_scale_codebook_image_results",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "materialization": str(args.materialization),
        "materialization_sha256": file_sha256(args.materialization),
        "runtime_state_sha256": materialization["runtime_state"]["file_sha256"],
        "dense_state_sha256": materialization["dense_state"]["file_sha256"],
        "bundle": str(args.bundle),
        "bundle_sha256": file_sha256(args.bundle),
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_reference": str(args.q4_reference),
        "q4_reference_sha256": file_sha256(args.q4_reference),
        "excluded_prior_protocol": str(args.prior_protocol),
        "excluded_prior_protocol_sha256": file_sha256(args.prior_protocol),
        "excluded_prior_sample_count": len(excluded),
        "fold_seed": 20260801,
        "screen": sample_receipts(screen),
        "screen_source_counts": source_counts(screen),
        "confirmation": sample_receipts(confirmation),
        "confirmation_source_counts": source_counts(confirmation),
        "candidate_scale_index_bits": [1, 2, 3, 4, 6, 8],
        "reference_codec": "fp16_per_group",
        "selection_order": "smallest_passing_packed_trunk_bits",
        "decision_rule": {
            "bootstrap_seed": 20260801,
            "bootstrap_samples": 5000,
            "ground_truth_lower_95_min_db": -0.02,
            "edge_mean_min_db": -0.02,
            "teacher_lower_95_min_db": -0.03,
            "every_source_ground_truth_mean_min_db": -0.03,
            "minimum_scale_storage_reduction_fraction": 0.5,
        },
        "passing_result": (
            "smallest confirmed codec advances to packed-runtime integration; "
            "it does not authorize model promotion"
        ),
        "failing_result": "retain FP16 per-group scales",
        "source_sha256": {
            name: file_sha256(repository / name) for name in SOURCE_FILES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "screen_source_counts": protocol["screen_source_counts"],
        "confirmation_source_counts": protocol["confirmation_source_counts"],
        "excluded_prior_sample_count": len(excluded),
    }, indent=2))


if __name__ == "__main__":
    main()
