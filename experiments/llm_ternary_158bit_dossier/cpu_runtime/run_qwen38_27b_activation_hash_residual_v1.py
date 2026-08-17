#!/usr/bin/env python3
"""Activation-weighted Clark-Hash residual screen on real Qwen3.8-27B rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.qwen38_activation_hash_residual import (
    activation_weighted_hash_coefficients,
    output_error_stats,
    train_int4_activation_hash_coefficients,
)
from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    fit_hash_ternary_residual,
    projected_hash_residual_bits_per_weight,
    quantize_hash_coefficients_int4,
    sketch_rows,
)
from clark_quantize.ternary import ternary_codes_scales


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_frozen_sources(preregistration: dict[str, Any]) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen activation-hash source changed: {label}")


def full_output_stats(
    source: torch.Tensor,
    ternary: torch.Tensor,
    coefficients: torch.Tensor,
    inputs: torch.Tensor,
    projection: Any,
) -> dict[str, float]:
    target = inputs.float() @ source.float().T
    candidate = (
        inputs.float() @ ternary.float().T
        + sketch_rows(inputs.float(), projection) @ coefficients.float().T
    )
    difference = candidate - target
    squared_error = float(difference.square().sum(dtype=torch.float64).item())
    target_energy = float(target.square().sum(dtype=torch.float64).item())
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def symmetric_baseline_stats(
    source: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group: int,
) -> dict[str, float]:
    codes, scales = ternary_codes_scales(source, group=group, method="optimal")
    frozen = scales.to(torch.float16).to(torch.float32)
    ternary = (
        codes.reshape(source.shape[0], -1, group).float() * frozen[:, :, None]
    ).reshape_as(source)
    target = inputs.float() @ source.float().T
    candidate = inputs.float() @ ternary.T
    difference = candidate - target
    squared_error = float(difference.square().sum(dtype=torch.float64).item())
    target_energy = float(target.square().sum(dtype=torch.float64).item())
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def aggregate(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    squared_error = sum(float(record[key]["squared_error"]) for record in records)
    target_energy = sum(float(record[key]["target_squared_sum"]) for record in records)
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-27b-layer0-mlp-activation-qat-v1/activations.pt"),
    )
    parser.add_argument(
        "--activation-cache-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json"),
    )
    parser.add_argument(
        "--hash-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-activation-hash-residual-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_activation_hash_residual_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    if (
        prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != REVISION
    ):
        raise ValueError("activation-hash Qwen3.8-27B preregistration identity changed")

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download.get("model_id") != MODEL_ID or download.get("revision") != REVISION:
        raise ValueError("activation-hash checkpoint receipt identity mismatch")
    index_path = args.model_dir / "model.safetensors.index.json"
    index_content = index_path.read_bytes()
    if hashlib.sha256(index_content).hexdigest() != INDEX_SHA256:
        raise ValueError("activation-hash Qwen3.8-27B index changed")
    weight_map = json.loads(index_content)["weight_map"]

    hash_receipt = json.loads(args.hash_receipt.read_text())
    if (
        hash_receipt.get("schema") != "qwen38_27b_hash_residual_screen_receipt_v1"
        or hash_receipt.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("activation-hash parent is not a passing terminal receipt")
    selected_name = hash_receipt["gate"]["selected_candidate"]
    parent_prereg_path = runtime / "qwen38_27b_hash_residual_screen_preregistration_v1.json"
    parent_prereg = json.loads(parent_prereg_path.read_text())
    if hash_receipt.get("preregistration_sha256") != sha256_file(parent_prereg_path):
        raise ValueError("activation-hash parent preregistration identity changed")
    candidates = {
        item["name"]: item for item in parent_prereg["candidate_family"]["candidates"]
    }
    if selected_name not in candidates:
        raise ValueError("activation-hash selected parent candidate changed")
    selected = candidates[selected_name]

    cache_receipt = json.loads(args.activation_cache_receipt.read_text())
    activation_qat_prereg_path = runtime / (
        "qwen38_27b_layer0_mlp_activation_qat_preregistration_v1.json"
    )
    activation_qat_prereg = json.loads(activation_qat_prereg_path.read_text())
    activation_qat_prereg_sha256 = sha256_file(activation_qat_prereg_path)
    if (
        cache_receipt.get("schema") != "qwen38_27b_layer0_mlp_activation_cache_receipt_v1"
        or cache_receipt.get("model_id") != MODEL_ID
        or cache_receipt.get("model_revision") != REVISION
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("preregistration", {}).get("sha256")
        != activation_qat_prereg_sha256
        or cache_receipt.get("cache", {}).get("sha256") != sha256_file(args.activation_cache)
    ):
        raise ValueError("activation-hash cache receipt is invalid")
    cache = torch.load(args.activation_cache, map_location="cpu", weights_only=False)
    required_cache = {
        "mlp_input_calibration",
        "mlp_input_validation",
        "down_input_calibration",
        "down_input_validation",
    }
    if (
        cache.get("schema") != "qwen38_27b_layer0_mlp_activation_cache_v1"
        or cache.get("model_id") != MODEL_ID
        or cache.get("model_revision") != REVISION
        or cache.get("preregistration_sha256") != activation_qat_prereg_sha256
        or cache.get("calibration_texts")
        != activation_qat_prereg["activation_cache"]["calibration_texts"]
        or cache.get("validation_texts")
        != activation_qat_prereg["activation_cache"]["validation_texts"]
        or not required_cache.issubset(cache)
    ):
        raise ValueError("activation-hash cache payload is invalid")

    family = parent_prereg["candidate_family"]
    training = prereg["training"]
    records: list[dict[str, Any]] = []
    parent_records = {
        (record["tensor"], record["row_start"], record["row_stop"]): record
        for record in hash_receipt["records"]
    }
    for tensor_index, tensor_spec in enumerate(parent_prereg["scope"]["tensors"]):
        name = tensor_spec["name"]
        shard = weight_map.get(name)
        if shard != tensor_spec["shard"]:
            raise ValueError(f"activation-hash tensor shard changed: {name}")
        start, stop = tensor_spec["row_span"]
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            if list(view.get_shape()) != tensor_spec["shape"]:
                raise ValueError(f"activation-hash tensor shape changed: {name}")
            source = view[start:stop].contiguous().reshape(stop - start, -1).float()
        activation_prefix = "down_input" if ".down_proj." in name else "mlp_input"
        calibration = cache[f"{activation_prefix}_calibration"].float().contiguous()
        validation = cache[f"{activation_prefix}_validation"].float().contiguous()
        if not bool(torch.isfinite(calibration).all()) or not bool(torch.isfinite(validation).all()):
            raise ValueError("activation-hash cache contains non-finite inputs")

        sketch_dim = source.shape[1] // int(selected["input_to_sketch_ratio"])
        fit = fit_hash_ternary_residual(
            source,
            sketch_dim=sketch_dim,
            hashes_per_input=int(family["hashes_per_input"]),
            seed=int(family["seed"]),
            group=int(family["group_size"]),
            nonzero_per_group=int(selected["nonzero_per_group"]),
            iterations=int(family["alternating_iterations"]),
        )
        parent_record = parent_records.get((name, start, stop))
        if parent_record is None:
            raise ValueError("activation-hash parent row record is absent")
        parent_candidate = parent_record.get("candidates", {}).get(selected_name)
        if parent_candidate is None or not math.isclose(
            fit.exported_squared_error,
            float(parent_candidate["squared_error"]),
            rel_tol=2e-6,
            abs_tol=1e-7,
        ):
            raise ValueError("activation-hash parent candidate replay changed")
        group = int(family["group_size"])
        frozen_scales = fit.ternary_scales.to(torch.float16).to(torch.float32)
        ternary = (
            fit.ternary_codes.reshape(source.shape[0], -1, group).float()
            * frozen_scales[:, :, None]
        ).reshape_as(source)
        residual = source - ternary
        _, data_free_scales, data_free_export = quantize_hash_coefficients_int4(
            fit.hash_coefficients
        )
        activation_float = activation_weighted_hash_coefficients(
            residual,
            calibration,
            fit.projection,
            reference_coefficients=fit.hash_coefficients,
            relative_ridge=float(training["relative_ridge"]),
        )
        _, activation_scales, activation_export = quantize_hash_coefficients_int4(
            activation_float
        )
        trained = train_int4_activation_hash_coefficients(
            residual,
            calibration,
            fit.projection,
            activation_float,
            reference_coefficients=fit.hash_coefficients,
            steps=int(training["steps"]),
            learning_rate=float(training["learning_rate"]),
            anchor_weight=float(training["data_free_anchor_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )

        data_free_calibration = full_output_stats(
            source, ternary, data_free_export, calibration, fit.projection
        )
        data_free_validation = full_output_stats(
            source, ternary, data_free_export, validation, fit.projection
        )
        activation_float_calibration = output_error_stats(
            residual, activation_float, calibration, fit.projection
        )
        activation_int4_calibration = output_error_stats(
            residual, activation_export, calibration, fit.projection
        )
        trained_calibration = full_output_stats(
            source, ternary, trained.coefficients, calibration, fit.projection
        )
        trained_validation = full_output_stats(
            source, ternary, trained.coefficients, validation, fit.projection
        )
        symmetric_validation = symmetric_baseline_stats(
            source, validation, group=group
        )
        implicit = (
            validation @ ternary.T
            + sketch_rows(validation, fit.projection) @ trained.coefficients.T
        )
        expanded_weight = ternary + expand_sketch_rows(trained.coefficients, fit.projection)
        explicit = validation @ expanded_weight.T
        identity_error = float((implicit - explicit).abs().max().item())
        if identity_error > float(prereg["gate"]["maximum_kernel_identity_error"]):
            raise RuntimeError(f"activation-hash runtime identity failed: {identity_error}")

        records.append(
            {
                "tensor": name,
                "row_start": start,
                "row_stop": stop,
                "parameters": source.numel(),
                "activation_prefix": activation_prefix,
                "source_sha256": tensor_sha256(source),
                "sketch_dim": sketch_dim,
                "nonzero_per_group": int(selected["nonzero_per_group"]),
                "projected_component_bits_per_weight": projected_hash_residual_bits_per_weight(
                    input_dim=source.shape[1],
                    sketch_dim=sketch_dim,
                    nonzero_per_group=int(selected["nonzero_per_group"]),
                    group=group,
                ),
                "data_free_calibration": data_free_calibration,
                "data_free_validation": data_free_validation,
                "activation_float_calibration_residual": activation_float_calibration,
                "activation_int4_calibration_residual": activation_int4_calibration,
                "trained_calibration": trained_calibration,
                "trained_validation": trained_validation,
                "symmetric_validation": symmetric_validation,
                "relative_validation_error_reduction_vs_data_free": (
                    1.0
                    - trained_validation["squared_error"]
                    / max(data_free_validation["squared_error"], 1e-30)
                ),
                "relative_validation_error_reduction_vs_symmetric": (
                    1.0
                    - trained_validation["squared_error"]
                    / max(symmetric_validation["squared_error"], 1e-30)
                ),
                "data_free_int4_scales_sha256": tensor_sha256(data_free_scales),
                "activation_int4_scales_sha256": tensor_sha256(activation_scales),
                "trained_int4_codes_sha256": tensor_sha256(trained.codes),
                "trained_int4_scales_sha256": tensor_sha256(trained.scales),
                "training_best_step": trained.best_step,
                "training_history": list(trained.calibration_history),
                "kernel_identity_max_abs_error": identity_error,
            }
        )

    summary = {
        "data_free_calibration": aggregate(records, "data_free_calibration"),
        "data_free_validation": aggregate(records, "data_free_validation"),
        "trained_calibration": aggregate(records, "trained_calibration"),
        "trained_validation": aggregate(records, "trained_validation"),
        "symmetric_validation": aggregate(records, "symmetric_validation"),
    }
    summary["relative_validation_error_reduction_vs_data_free"] = (
        1.0
        - summary["trained_validation"]["squared_error"]
        / max(summary["data_free_validation"]["squared_error"], 1e-30)
    )
    summary["relative_validation_error_reduction_vs_symmetric"] = (
        1.0
        - summary["trained_validation"]["squared_error"]
        / max(summary["symmetric_validation"]["squared_error"], 1e-30)
    )
    maximum_tensor_regression = max(
        0.0,
        max(
            -float(record["relative_validation_error_reduction_vs_data_free"])
            for record in records
        ),
    )
    maximum_rate = max(float(record["projected_component_bits_per_weight"]) for record in records)
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "selected_parent_candidate": selected_name,
        "observed_tensor_spans": len(records),
        "observed_relative_validation_error_reduction_vs_data_free": summary[
            "relative_validation_error_reduction_vs_data_free"
        ],
        "observed_relative_validation_error_reduction_vs_symmetric": summary[
            "relative_validation_error_reduction_vs_symmetric"
        ],
        "observed_maximum_tensor_relative_validation_regression_vs_data_free": maximum_tensor_regression,
        "observed_maximum_projected_component_bits_per_weight": maximum_rate,
        "observed_maximum_kernel_identity_error": max(
            float(record["kernel_identity_max_abs_error"]) for record in records
        ),
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_tensor_spans"])
        and summary["relative_validation_error_reduction_vs_data_free"]
        >= float(gate_spec["minimum_relative_validation_error_reduction_vs_data_free"])
        and summary["relative_validation_error_reduction_vs_symmetric"] >= 0.0
        and maximum_tensor_regression
        <= float(gate_spec["maximum_tensor_relative_validation_regression_vs_data_free"])
        and maximum_rate <= float(gate_spec["maximum_projected_component_bits_per_weight"])
        and gate["observed_maximum_kernel_identity_error"]
        <= float(gate_spec["maximum_kernel_identity_error"])
    )

    receipt = {
        "schema": "qwen38_27b_activation_hash_residual_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent_hash_receipt": {
            "path": str(args.hash_receipt),
            "sha256": sha256_file(args.hash_receipt),
        },
        "activation_cache_receipt": {
            "path": str(args.activation_cache_receipt),
            "sha256": sha256_file(args.activation_cache_receipt),
        },
        "records": records,
        "summary": summary,
        "gate": gate,
        "complete": True,
        "claim_boundary": {
            "real_Qwen38_27B_weights_measured": True,
            "real_disjoint_calibration_and_validation_activations_measured": True,
            "sampled_rows_only": True,
            "ternary_codes_and_scales_frozen_from_data_free_parent": True,
            "hash_coefficients_hard_INT4_trained": True,
            "capability_preservation_proven": False,
            "complete_storage_proven": False,
            "native_fused_kernel_proven": False,
            "semi_lossless_layer_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
