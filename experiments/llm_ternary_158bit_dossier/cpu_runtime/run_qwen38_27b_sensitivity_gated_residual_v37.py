#!/usr/bin/env python3
"""Materialize and confirm the fixed v37 sensitivity-gated residual."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import run_qwen38_27b_calibration_span_residual_v36 as v36
import run_qwen38_27b_mixed_train_blocks_v1 as prior_training

from clark_quantize.qwen38_calibration_span_residual import (
    fit_float_residual_adapter,
    load_int8_adapter,
    normalized_mse,
    quantize_residual_adapter,
    write_int8_adapter,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
ACTIVE_LAYERS = (0, 63)
BYPASS_LAYERS = (32,)
RANK = 16
RIDGE_FRACTION = 0.01
CALIBRATION_SCHEMA = "qwen38_27b_sensitivity_gated_residual_fit_result_v37"
CONFIRMATION_SCHEMA = "qwen38_27b_sensitivity_gated_residual_confirmation_result_v37"


def validate_common(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str, str]:
    prereg = json.loads(args.preregistration.read_text())
    training = json.loads(args.training_receipt.read_text())
    v36_receipt = json.loads(args.v36_receipt.read_text())
    prereg_sha = v36.sha256_file(args.preregistration)
    training_sha = v36.sha256_file(args.training_receipt)
    v36_sha = v36.sha256_file(args.v36_receipt)
    if (
        prereg.get("schema")
        != "qwen38_27b_sensitivity_gated_residual_preregistration_v37"
        or prereg.get("status")
        != "frozen_after_terminal_v36_before_any_fresh_holdout_collection"
        or prereg["scope"].get("active_model_only") != MODEL_ID
        or prereg["scope"].get("revision") != MODEL_REVISION
        or tuple(prereg["scope"].get("pilot_layers", [])) != TARGET_LAYERS
        or prereg["scope"].get("qwen_4b_compute_forbidden") is not True
        or tuple(prereg["sensitivity_rule"].get("active_layers", []))
        != ACTIVE_LAYERS
        or tuple(prereg["sensitivity_rule"].get("exact_bypass_layers", []))
        != BYPASS_LAYERS
        or prereg["fixed_adapter"].get("rank") != RANK
        or prereg["fixed_adapter"].get("ridge_fraction") != RIDGE_FRACTION
        or prereg["fixed_adapter"].get("new_grid_or_hyperparameter_search_forbidden")
        is not True
        or training_sha != prereg["base_candidate"]["training_receipt_sha256"]
        or training.get("schema") != prior_training.SCHEMA
        or training.get("gate", {}).get("passed") is not True
        or training.get("complete") is not True
        or training.get("terminal") is not True
        or v36_sha != prereg["parent_evidence"]["v36_receipt"]["sha256"]
        or v36_receipt.get("schema") != v36.CALIBRATION_SCHEMA
        or v36_receipt.get("gate", {}).get("passed") is not False
        or v36_receipt.get("selected") is not None
        or v36_receipt.get("terminal") is not True
        or len(v36_receipt.get("grid", [])) != 12
    ):
        raise ValueError("v37 frozen identity or v36 parent evidence changed")
    v36.resource_guard(
        Path("/workspace"),
        int(prereg["execution_policy"]["minimum_free_disk_bytes"]),
    )
    return prereg, training, v36_receipt, prereg_sha, training_sha, v36_sha


def load_model_context(args: argparse.Namespace) -> tuple[Any, dict[str, str]]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != 5120
        or tuple(config.layer_types[layer] for layer in TARGET_LAYERS)
        != ("linear_attention", "linear_attention", "full_attention")
    ):
        raise ValueError("v37 Qwen3.8-27B architecture changed")
    return config, weight_map


def base_layer_data(
    *,
    args: argparse.Namespace,
    config: Any,
    weight_map: dict[str, str],
    cache: dict[str, Any],
    selected: torch.Tensor,
    layer: int,
    training: dict[str, Any],
    training_sha: str,
    prereg_sha: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    return v36.trained_candidate_output(
        config=config,
        model_dir=args.model_dir,
        weight_map=weight_map,
        cache=cache,
        selected=selected,
        layer_index=layer,
        training_receipt=training,
        training_receipt_sha=training_sha,
        prereg_sha=prereg_sha,
        parent_artifact_root=args.parent_artifact_root,
        trained_artifact_root=args.trained_artifact_root,
    )


def validate_existing(path: Path, schema: str, prereg_sha: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    existing = json.loads(path.read_text())
    if (
        existing.get("schema") != schema
        or existing.get("preregistration_sha256") != prereg_sha
        or existing.get("complete") is not True
        or existing.get("terminal") is not True
    ):
        raise ValueError("existing v37 terminal receipt changed")
    return existing


def run_calibration(args: argparse.Namespace) -> int:
    prereg, training, _, prereg_sha, training_sha, v36_sha = validate_common(args)
    existing = validate_existing(args.out, CALIBRATION_SCHEMA, prereg_sha)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing["gate"]["passed"] else 2
    if v36.sha256_file(args.training_cache) != prereg["data"]["training_cache"][
        "sha256"
    ]:
        raise ValueError("v37 training cache identity changed")
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    selected = cache["calibration_flat_rows"].long()
    if selected.numel() != 64 or int(selected.max()) >= 8 * 64:
        raise ValueError("v37 calibration denominator changed")
    config, weight_map = load_model_context(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    layers = []
    payload_roundtrips = True
    for layer in TARGET_LAYERS:
        inputs, candidates, targets, base_nmse = base_layer_data(
            args=args,
            config=config,
            weight_map=weight_map,
            cache=cache,
            selected=selected,
            layer=layer,
            training=training,
            training_sha=training_sha,
            prereg_sha=prereg_sha,
        )
        payload_path = args.artifact_root / f"layer-{layer}.csr"
        descriptor_path = args.artifact_root / f"layer-{layer}.json"
        if layer in ACTIVE_LAYERS:
            adapter = quantize_residual_adapter(
                fit_float_residual_adapter(
                    inputs,
                    candidates,
                    targets,
                    rank=RANK,
                    ridge_fraction=RIDGE_FRACTION,
                )
            )
            write_int8_adapter(payload_path, adapter)
            restored = load_int8_adapter(
                payload_path,
                input_features=inputs.shape[1],
                output_features=targets.shape[1],
                rank=RANK,
                ridge_fraction=RIDGE_FRACTION,
            )
            payload_roundtrips = payload_roundtrips and all(
                torch.equal(getattr(adapter, field), getattr(restored, field))
                for field in (
                    "left_codes",
                    "left_scales",
                    "right_codes",
                    "right_scales",
                    "bias",
                )
            )
            corrected_nmse = normalized_mse(
                candidates + restored.correction(inputs), targets
            )
            descriptor = v36.adapter_descriptor(
                payload_path,
                restored,
                layer=layer,
                prereg_sha=prereg_sha,
                training_sha=training_sha,
            )
            descriptor["sensitivity_admitted"] = True
            descriptor["sensitivity_threshold"] = prereg["sensitivity_rule"][
                "minimum_base_nmse_for_residual"
            ]
            v36.atomic_json(descriptor_path, descriptor)
            payload = descriptor["payload"]
            descriptor_record: dict[str, Any] | None = {
                "path": str(descriptor_path),
                "sha256": v36.sha256_file(descriptor_path),
            }
            mode = "rank16_int8_residual"
        else:
            if payload_path.exists() or descriptor_path.exists():
                raise ValueError("v37 bypass layer unexpectedly has a payload")
            corrected_nmse = base_nmse
            payload = None
            descriptor_record = None
            mode = "exact_bypass"
        relative = (base_nmse - corrected_nmse) / max(base_nmse, 1e-30)
        layers.append(
            {
                "layer": layer,
                "mode": mode,
                "calibration_rows": selected.numel(),
                "base_calibration_nmse": base_nmse,
                "corrected_calibration_nmse": corrected_nmse,
                "relative_reduction": relative,
                "payload": payload,
                "descriptor": descriptor_record,
            }
        )
        print(
            json.dumps({"event": "v37_calibration_layer", **layers[-1]}, sort_keys=True),
            flush=True,
        )
        del inputs, candidates, targets
        gc.collect()

    active = [row for row in layers if row["layer"] in ACTIVE_LAYERS]
    bypass = [row for row in layers if row["layer"] in BYPASS_LAYERS]
    projection = prereg["storage_and_compute"]["conservative_all_64_layers_projection"]
    base_image = int(
        prereg["base_candidate"]["projected_complete_compute_image_bytes_before_residual"]
    )
    checks = {
        "both_active_layers_not_regressed": all(
            row["corrected_calibration_nmse"] <= row["base_calibration_nmse"]
            for row in active
        ),
        "mean_active_relative_reduction_at_least_0_2": (
            sum(row["relative_reduction"] for row in active) / len(active)
            >= float(
                prereg["gates"]["calibration_materialization"][
                    "minimum_mean_active_layer_relative_nmse_reduction"
                ]
            )
        ),
        "bypass_layer_has_no_payload_and_is_exact": all(
            row["payload"] is None
            and row["corrected_calibration_nmse"] == row["base_calibration_nmse"]
            for row in bypass
        ),
        "all_payload_roundtrips_exact": payload_roundtrips,
        "payload_count_is_two": len(list(args.artifact_root.glob("*.csr"))) == 2,
        "conservative_residual_bytes_at_most_12m": projection["persistent_bytes"]
        <= int(
            prereg["gates"]["deployment"][
                "maximum_residual_bytes_complete_64_layer_conservative_projection"
            ]
        ),
        "conservative_compute_image_at_most_12_42b": projection[
            "complete_compute_image_bytes"
        ]
        <= int(prereg["gates"]["deployment"]["maximum_complete_compute_image_bytes"]),
        "conservative_weight_fraction_at_most_0_001": projection[
            "persistent_bytes"
        ]
        / base_image
        <= float(
            prereg["gates"]["deployment"][
                "maximum_residual_weight_traffic_fraction"
            ]
        ),
        "conservative_macs_at_most_11m": projection["macs_per_token"]
        <= int(
            prereg["gates"]["deployment"][
                "maximum_residual_macs_per_token_complete_64_layer_conservative_projection"
            ]
        ),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "v36_receipt_sha256": v36_sha,
        "training_cache_sha256": v36.sha256_file(args.training_cache),
        "fixed_rank": RANK,
        "fixed_ridge_fraction": RIDGE_FRACTION,
        "active_layers": list(ACTIVE_LAYERS),
        "bypass_layers": list(BYPASS_LAYERS),
        "layers": layers,
        "conservative_projection": projection,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "next_stage": "collect_untouched_holdout" if gate["passed"] else None,
        "claim_boundary": prereg["claim_boundary"],
    }
    v36.atomic_json(args.out, receipt)
    print(json.dumps({"event": "v37_calibration_terminal", "gate": gate}, sort_keys=True))
    return 0 if gate["passed"] else 2


def run_confirmation(args: argparse.Namespace) -> int:
    prereg, training, _, prereg_sha, training_sha, v36_sha = validate_common(args)
    existing = validate_existing(args.out, CONFIRMATION_SCHEMA, prereg_sha)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing["gate"]["passed"] else 2
    calibration = json.loads(args.calibration_receipt.read_text())
    cache_receipt = json.loads(args.confirmation_cache_receipt.read_text())
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("gate", {}).get("passed") is not True
        or calibration.get("preregistration_sha256") != prereg_sha
        or calibration.get("v36_receipt_sha256") != v36_sha
        or cache_receipt.get("schema")
        != "qwen38_27b_mixed_confirmation_cache_receipt_v1"
        or cache_receipt.get("preregistration_sha256") != prereg_sha
        or cache_receipt.get("cache", {}).get("sha256")
        != v36.sha256_file(args.confirmation_cache)
        or cache_receipt.get("confirmation_rows") != 32
        or cache_receipt.get("complete") is not True
    ):
        raise ValueError("v37 calibration-to-confirmation handoff changed")
    cache = torch.load(args.confirmation_cache, map_location="cpu", weights_only=False)
    selected = cache["confirmation_flat_rows"].long()
    if (
        selected.numel() != 32
        or cache.get("confirmation_texts")
        != prereg["data"]["fresh_confirmation"]["texts"]
    ):
        raise ValueError("v37 untouched confirmation cache changed")
    config, weight_map = load_model_context(args)
    started = time.monotonic()
    layers = []
    for layer in TARGET_LAYERS:
        inputs, candidates, targets, base_nmse = base_layer_data(
            args=args,
            config=config,
            weight_map=weight_map,
            cache=cache,
            selected=selected,
            layer=layer,
            training=training,
            training_sha=training_sha,
            prereg_sha=prereg_sha,
        )
        payload_path = args.artifact_root / f"layer-{layer}.csr"
        descriptor_path = args.artifact_root / f"layer-{layer}.json"
        if layer in ACTIVE_LAYERS:
            calibration_layer = next(
                row for row in calibration["layers"] if row["layer"] == layer
            )
            descriptor = json.loads(descriptor_path.read_text())
            if (
                descriptor.get("rank") != RANK
                or descriptor.get("ridge_fraction") != RIDGE_FRACTION
                or descriptor.get("sensitivity_admitted") is not True
                or descriptor.get("payload") != calibration_layer["payload"]
                or v36.sha256_file(descriptor_path)
                != calibration_layer["descriptor"]["sha256"]
                or v36.sha256_file(payload_path) != descriptor["payload"]["sha256"]
            ):
                raise ValueError("v37 frozen active adapter changed")
            adapter = load_int8_adapter(
                payload_path,
                input_features=inputs.shape[1],
                output_features=targets.shape[1],
                rank=RANK,
                ridge_fraction=RIDGE_FRACTION,
            )
            corrected_candidates = candidates + adapter.correction(inputs)
            corrected_nmse = normalized_mse(corrected_candidates, targets)
            bypass_exact = None
            mode = "rank16_int8_residual"
        else:
            if payload_path.exists() or descriptor_path.exists():
                raise ValueError("v37 bypass payload appeared before confirmation")
            corrected_candidates = candidates
            corrected_nmse = normalized_mse(corrected_candidates, targets)
            bypass_exact = bool(torch.equal(corrected_candidates, candidates))
            mode = "exact_bypass"
        relative = (base_nmse - corrected_nmse) / max(base_nmse, 1e-30)
        layers.append(
            {
                "layer": layer,
                "mode": mode,
                "confirmation_rows": selected.numel(),
                "base_trained_mixed_nmse": base_nmse,
                "sensitivity_candidate_nmse": corrected_nmse,
                "relative_reduction": relative,
                "bypass_tensor_exact": bypass_exact,
                "candidates_scored": [
                    "frozen_trained_mixed",
                    (
                        "frozen_trained_mixed_plus_rank16_int8_residual"
                        if layer in ACTIVE_LAYERS
                        else "frozen_trained_mixed_exact_bypass"
                    ),
                ],
            }
        )
        print(
            json.dumps({"event": "v37_confirmation_layer", **layers[-1]}, sort_keys=True),
            flush=True,
        )
        del inputs, candidates, targets, corrected_candidates
        gc.collect()

    active = [row for row in layers if row["layer"] in ACTIVE_LAYERS]
    bypass = [row for row in layers if row["layer"] in BYPASS_LAYERS]
    gate_config = prereg["gates"]["fresh_confirmation"]
    checks = {
        "both_active_layers_not_regressed": all(
            row["sensitivity_candidate_nmse"] <= row["base_trained_mixed_nmse"]
            for row in active
        ),
        "bypass_layer_exactly_identical": all(
            row["bypass_tensor_exact"] is True
            and row["sensitivity_candidate_nmse"] == row["base_trained_mixed_nmse"]
            for row in bypass
        ),
        "mean_relative_reduction_all_three_at_least_0_1": (
            sum(row["relative_reduction"] for row in layers) / len(layers)
            >= float(
                gate_config[
                    "minimum_mean_relative_nmse_reduction_across_all_three_layers"
                ]
            )
        ),
        "all_final_nmse_at_most_0_05": all(
            row["sensitivity_candidate_nmse"]
            <= float(gate_config["maximum_final_complete_block_nmse_per_layer"])
            for row in layers
        ),
        "exactly_two_candidates_per_layer": all(
            len(row["candidates_scored"])
            == int(gate_config["exact_candidate_count_per_layer"])
            for row in layers
        ),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    receipt = {
        "schema": CONFIRMATION_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "v36_receipt_sha256": v36_sha,
        "calibration_receipt_sha256": v36.sha256_file(args.calibration_receipt),
        "confirmation_cache_receipt_sha256": v36.sha256_file(
            args.confirmation_cache_receipt
        ),
        "confirmation_cache_sha256": v36.sha256_file(args.confirmation_cache),
        "layers": layers,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "next_stage": (
            "implement_fused_avx2_rank16_residual_with_exact_bypass"
            if gate["passed"]
            else None
        ),
        "claim_boundary": prereg["claim_boundary"],
    }
    v36.atomic_json(args.out, receipt)
    print(json.dumps({"event": "v37_confirmation_terminal", "gate": gate}, sort_keys=True))
    return 0 if gate["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "confirmation"), required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--v36-receipt", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--parent-artifact-root", type=Path, required=True)
    parser.add_argument("--trained-artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--confirmation-cache", type=Path)
    parser.add_argument("--confirmation-cache-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        parser.error("v37 is frozen to four CPU threads")
    if args.stage == "confirmation" and not all(
        (
            args.calibration_receipt,
            args.confirmation_cache,
            args.confirmation_cache_receipt,
        )
    ):
        parser.error("v37 confirmation requires calibration and cache receipts")
    return args


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260819)
    return run_calibration(args) if args.stage == "calibration" else run_confirmation(args)


if __name__ == "__main__":
    raise SystemExit(main())
