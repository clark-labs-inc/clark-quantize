#!/usr/bin/env python3
"""Freeze the v45 residual preregistration from a passed v44 receipt."""

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
TRAINING_SCHEMA = "qwen38_27b_all_layer_mixed_training_receipt_v44"
FULL_CONTEXT_RECEIPT_SHA256 = (
    "8653749c9616d88cec3509cfd3c87262145d370da2fc613aab88ee02263aca28"
)
PILOT_RESIDUAL_PREREGISTRATION_SHA256 = (
    "4d5b8646887ccee1959fcd12fb8056deaa57ce06941a0f721c57843ba15dbd36"
)
PILOT_RESIDUAL_CALIBRATION_SHA256 = (
    "6871879dd4413419bfb36e233275481bce291d6bac1c884770c2514c876eeb73"
)
PILOT_LAYERS = (0, 32, 63)
RANK = 16
RIDGE_FRACTION = 0.01
SENSITIVITY_THRESHOLD = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    layers = receipt.get("layers", [])
    gate = receipt.get("gate", {})
    if (
        receipt.get("schema") != TRAINING_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("full_context_receipt_sha256")
        != FULL_CONTEXT_RECEIPT_SHA256
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or gate.get("passed") is not True
        or len(layers) != 64
        or [record.get("layer") for record in layers] != list(range(64))
        or not all(record.get("passed") is True for record in layers)
    ):
        raise ValueError("v44 terminal receipt identity or denominator changed")
    for layer in PILOT_LAYERS:
        record = layers[layer]
        if (
            record.get("pilot_payload_algebra_exact") is not True
            or record.get("pilot_score_exact_within_1e_8") is not True
        ):
            raise ValueError("v44 pilot training reproduction changed")
    if (
        gate.get("observed_layers") != 64
        or gate.get("all_layers_calibration_not_regressed_vs_direct") is not True
        or gate.get("all_payload_roundtrips_exact") is not True
        or gate.get("all_values_finite") is not True
        or gate.get("all_pilot_payload_algebras_exact") is not True
        or gate.get("all_pilot_scores_exact_within_1e_8") is not True
        or gate.get("payload_storage_within_maximum") is not True
    ):
        raise ValueError("v44 terminal receipt failed the frozen training gate")
    return {
        "mean_relative_calibration_nmse_reduction_vs_direct": gate[
            "mean_relative_calibration_nmse_reduction_vs_direct"
        ],
        "total_selected_payload_bytes": gate["total_selected_payload_bytes"],
    }


def build_preregistration(
    *,
    training_receipt: dict[str, Any],
    training_receipt_sha256: str,
    runner_sha256: str,
    source_commit: str,
    frozen_at_utc: str,
) -> dict[str, Any]:
    training_summary = validate_training_receipt(training_receipt)
    return {
        "schema": (
            "qwen38_27b_all_layer_sensitivity_residual_preregistration_v45"
        ),
        "experiment_id": "qwen38_27b_all_layer_sensitivity_residual_v45",
        "goal_id": "qwen38_27b_amd_cpu_breakthrough_v1",
        "status": "frozen_after_v44_pass_before_residual_fit",
        "frozen_at_utc": frozen_at_utc,
        "objective": (
            "Apply the already-passed v37 sensitivity hypothesis to every frozen "
            "v44 Qwen3.8-27B text block: classify by sealed internal-validation "
            "base error, fit one fixed calibration-only rank-16 INT8 correction "
            "where admitted, and preserve exact bypass everywhere else."
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
        "parent_training": {
            "receipt_path": (
                "/workspace/quant-state/"
                "qwen38-all-layer-mixed-trained-v44/receipt.json"
            ),
            "receipt_sha256": training_receipt_sha256,
            "repository_commit": training_receipt["repository_commit"],
            "preregistration_sha256": training_receipt[
                "preregistration_sha256"
            ],
            "direct_receipt_sha256": training_receipt[
                "direct_receipt_sha256"
            ],
            **training_summary,
        },
        "parent_evidence": {
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "pilot_residual_preregistration_sha256": (
                PILOT_RESIDUAL_PREREGISTRATION_SHA256
            ),
            "pilot_residual_calibration_sha256": (
                PILOT_RESIDUAL_CALIBRATION_SHA256
            ),
            "pilot_layers": list(PILOT_LAYERS),
        },
        "data_protocol": {
            "calibration_rows_per_layer": 64,
            "sealed_internal_validation_rows_per_layer": 32,
            "calibration_sequences": 8,
            "internal_validation_sequences": 4,
            "fresh_confirmation_examples_used": 0,
            "benchmark_examples_used": 0,
            "selection_uses_only_base_internal_validation_error": True,
            "adapter_fit_uses_only_calibration_rows": True,
        },
        "sensitivity_rule": {
            "input_statistic": "frozen v44 base internal-validation block NMSE",
            "minimum_base_validation_nmse": SENSITIVITY_THRESHOLD,
            "active_mode": "rank16_int8_residual",
            "inactive_mode": "exact_bypass",
            "post_freeze_admission_changes_forbidden": True,
        },
        "adapter": {
            "rank": RANK,
            "ridge_fraction": RIDGE_FRACTION,
            "fit_rows": 64,
            "persistent_bytes_per_active_layer": 174_144,
            "runtime_form": (
                "candidate_output + input @ left_INT8 @ right_INT8 + bias_FP16"
            ),
            "no_grid_or_optimizer_sweep": True,
        },
        "execution": {
            "seed": 20260819,
            "fixed_layer_order": list(range(64)),
            "restart_boundary": "one authenticated terminal layer",
            "calibration_and_validation_forward_widths": [64, 32],
            "one_candidate_per_layer": True,
        },
        "gate": {
            "required_layers": 64,
            "all_active_layers_calibration_not_regressed": True,
            "all_active_layers_validation_not_regressed": True,
            "all_bypass_layers_exact": True,
            "minimum_mean_active_relative_validation_nmse_reduction": 0.1,
            "maximum_final_validation_nmse_per_layer": 0.05,
            "maximum_residual_payload_bytes": 12_000_000,
            "pilot_modes_scores_and_payloads_must_reproduce": True,
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
            "runner_sha256": runner_sha256,
            "runner_repository_commit": source_commit,
            "training_receipt_sha256": training_receipt_sha256,
            "direct_receipt_sha256": training_receipt[
                "direct_receipt_sha256"
            ],
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "pilot_residual_preregistration_sha256": (
                PILOT_RESIDUAL_PREREGISTRATION_SHA256
            ),
            "pilot_residual_calibration_sha256": (
                PILOT_RESIDUAL_CALIBRATION_SHA256
            ),
        },
        "promotion_rules": [
            "all 64 layers must reach one ordered terminal classification",
            "no sensitive layer may regress on sealed internal validation",
            "every insensitive layer must remain an exact no-payload bypass",
            "the three pilot layers must reproduce their frozen modes and algebra",
            "fresh confirmation and whole-model benchmarks stay sealed until v45 terminates",
        ],
        "claim_boundary": (
            "A v45 pass establishes one complete sensitivity map and compact "
            "activation-subspace residual set for the frozen trained base. It is "
            "not whole-model capability, llama.cpp execution, CPU throughput, "
            "energy, novelty, or scientific-breakthrough evidence."
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
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=RUNTIME
        / "run_qwen38_27b_all_layer_sensitivity_residual_v45.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_sensitivity_residual_v45.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a frozen v45 preregistration")
    tracked_status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if tracked_status.strip():
        raise SystemExit("refusing to freeze v45 from a dirty tracked worktree")
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = json.loads(args.training_receipt.read_text())
    prereg = build_preregistration(
        training_receipt=receipt,
        training_receipt_sha256=sha256_file(args.training_receipt),
        runner_sha256=sha256_file(args.runner),
        source_commit=source_commit,
        frozen_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    atomic_json(prereg, args.output)
    print(
        json.dumps(
            {
                "event": "v45_preregistration_frozen",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "training_receipt_sha256": prereg["parent_training"][
                    "receipt_sha256"
                ],
                "terminal": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
