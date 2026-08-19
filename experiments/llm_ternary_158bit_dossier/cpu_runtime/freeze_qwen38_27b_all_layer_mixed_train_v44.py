#!/usr/bin/env python3
"""Freeze the v44 all-layer training preregistration from a passed v43 receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DIRECT_SCHEMA = "qwen38_27b_all_layer_mixed_direct_receipt_v43"
FULL_CONTEXT_RECEIPT_SHA256 = (
    "8653749c9616d88cec3509cfd3c87262145d370da2fc613aab88ee02263aca28"
)
PILOT_TRAINING_RECEIPT_SHA256 = (
    "4454761a8519db03ca88a91c21cf8344cab12445266f1936767c04ebe3b76128"
)
PILOT_LAYERS = (0, 32, 63)
DIRECT_NMSE_LIMIT = 0.12
MAXIMUM_SELECTED_PAYLOAD_BYTES = 5_100_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_direct_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    layers = receipt.get("layers", [])
    gate = receipt.get("gate", {})
    if (
        receipt.get("schema") != DIRECT_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("full_context_receipt_sha256")
        != FULL_CONTEXT_RECEIPT_SHA256
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or gate.get("passed") is not True
        or len(layers) != 64
        or [record.get("layer") for record in layers] != list(range(64))
    ):
        raise ValueError("v43 terminal receipt identity or denominator changed")
    values = [
        record.get("direct_mixed_calibration_complete_block_nmse")
        for record in layers
    ]
    if (
        not all(isinstance(value, (int, float)) for value in values)
        or not all(record.get("passed") is True for record in layers)
        or max(values) > DIRECT_NMSE_LIMIT
        or gate.get("maximum_observed_calibration_nmse") != max(values)
        or gate.get("all_layers_below_maximum") is not True
        or gate.get("total_selected_payload_bytes", MAXIMUM_SELECTED_PAYLOAD_BYTES + 1)
        > MAXIMUM_SELECTED_PAYLOAD_BYTES
        or gate.get("payload_storage_within_maximum") is not True
    ):
        raise ValueError("v43 terminal receipt failed the frozen numeric gate")
    for layer in PILOT_LAYERS:
        record = layers[layer]
        if (
            record.get("aff3_pilot_algebra_exact") is not True
            or record.get("cq3_pilot_algebra_exact") is not True
            or record.get("pilot_score_exact_within_1e_8") is not True
        ):
            raise ValueError("v43 pilot reproduction changed")
    return {
        "maximum_observed_calibration_nmse": max(values),
        "maximum_observed_layer": values.index(max(values)),
        "mean_direct_calibration_nmse": sum(values) / len(values),
        "total_selected_payload_bytes": gate["total_selected_payload_bytes"],
    }


def build_preregistration(
    *,
    direct_receipt: dict[str, Any],
    direct_receipt_sha256: str,
    runner_sha256: str,
    source_commit: str,
    frozen_at_utc: str,
) -> dict[str, Any]:
    direct_summary = validate_direct_receipt(direct_receipt)
    return {
        "schema": "qwen38_27b_all_layer_mixed_training_preregistration_v44",
        "experiment_id": "qwen38_27b_all_layer_mixed_train_v44",
        "goal_id": "qwen38_27b_amd_cpu_breakthrough_v1",
        "status": "frozen_after_v43_pass_before_training",
        "frozen_at_utc": frozen_at_utc,
        "objective": (
            "Train the authenticated v43 AFF3/CQ3 algebra at all 64 Qwen3.8-27B "
            "text layers with the already-passed calibration-only optimizers, while "
            "keeping validation, confirmation, and benchmark text sealed."
        ),
        "scope": {
            "active_model_only": MODEL_ID,
            "revision": MODEL_REVISION,
            "target_layers": list(range(64)),
            "host": "quant",
            "cpu": "AMD EPYC Milan",
            "physical_cores": 4,
            "qwen_4b_compute_forbidden": True,
        },
        "parent_direct": {
            "receipt_path": (
                "/workspace/quant-state/"
                "qwen38-all-layer-mixed-direct-v43/receipt.json"
            ),
            "receipt_sha256": direct_receipt_sha256,
            "repository_commit": direct_receipt["repository_commit"],
            "preregistration_sha256": direct_receipt[
                "preregistration_sha256"
            ],
            **direct_summary,
        },
        "parent_evidence": {
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "pilot_training_receipt_sha256": PILOT_TRAINING_RECEIPT_SHA256,
            "pilot_layers": list(PILOT_LAYERS),
        },
        "data_protocol": {
            "calibration_sequences": 8,
            "tokens_per_sequence": 64,
            "selected_calibration_rows_per_layer": 64,
            "internal_validation_sequences_used": 0,
            "fresh_confirmation_sequences_used": 0,
            "benchmark_examples_used": 0,
        },
        "training": {
            "seed": 20260818,
            "fixed_layer_order": list(range(64)),
            "restart_boundary": "one authenticated terminal layer",
            "mlp_cq3": {
                "method": (
                    "activation-aware row-sharded hard-forward "
                    "straight-through training"
                ),
                "row_shard": 128,
                "group_size": 128,
                "steps_per_row_shard": 6,
                "learning_rate": 0.0001,
                "gradient_clip_norm": 1.0,
                "weight_reconstruction_weight": 0.01,
                "maximum_weighted_code_churn_fraction": 0.05,
            },
            "attention_output_aff3": {
                "method": "continuous-only hard-FP16 trust-region descent",
                "outer_steps": 8,
                "initial_global_l2_radius": 0.02,
                "backtracking_factors": [
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                    0.03125,
                ],
            },
            "no_sweep": (
                "Reuse the frozen pilot hyperparameters exactly once at every "
                "layer; no score-dependent retuning or fallback is permitted."
            ),
        },
        "gate": {
            "required_layers": 64,
            "all_layers_must_not_regress_vs_direct": True,
            "minimum_mean_relative_calibration_nmse_reduction_vs_direct": 0.01,
            "maximum_selected_payload_bytes": MAXIMUM_SELECTED_PAYLOAD_BYTES,
            "all_payload_roundtrips_exact": True,
            "maximum_weighted_code_churn_fraction": 0.05,
            "pilot_payloads_and_scores_must_reproduce": True,
        },
        "resources": {
            "threads": 4,
            "cpu_quota_percent": 400,
            "memory_max_bytes": 31_000_000_000,
            "memory_swap_max_bytes": 0,
            "minimum_free_disk_bytes": 20 * 1024**3,
            "maximum_restarts": 2,
        },
        "frozen_inputs": {
            "runner": {
                "path": "run_qwen38_27b_all_layer_mixed_train_v44.py",
                "sha256": runner_sha256,
                "repository_commit": source_commit,
            },
            "direct_receipt_sha256": direct_receipt_sha256,
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "pilot_training_receipt_sha256": PILOT_TRAINING_RECEIPT_SHA256,
        },
        "promotion_rules": [
            "a pass requires exactly 64 ordered terminal trained layers",
            "no layer may regress against its frozen v43 direct calibration score",
            "the mean relative calibration NMSE reduction must be at least 0.01",
            "all three pilot payload algebras and scores must reproduce exactly",
            "internal validation opens only after the complete trained base is frozen",
            "whole-model benchmarks remain sealed until representation selection ends",
        ],
        "claim_boundary": (
            "A v44 pass proves calibration-only all-layer training of one frozen "
            "representation. It is not whole-model capability, deployable export, "
            "CPU throughput, energy, novelty, or scientific-breakthrough evidence."
        ),
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-receipt", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=RUNTIME / "run_qwen38_27b_all_layer_mixed_train_v44.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_mixed_train_v44.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a frozen v44 preregistration")
    tracked_status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if tracked_status.strip():
        raise SystemExit("refusing to freeze v44 from a dirty tracked worktree")
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = json.loads(args.direct_receipt.read_text())
    prereg = build_preregistration(
        direct_receipt=receipt,
        direct_receipt_sha256=sha256_file(args.direct_receipt),
        runner_sha256=sha256_file(args.runner),
        source_commit=source_commit,
        frozen_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    atomic_json(prereg, args.output)
    print(
        json.dumps(
            {
                "event": "v44_preregistration_frozen",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "direct_receipt_sha256": prereg["parent_direct"][
                    "receipt_sha256"
                ],
                "terminal": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
