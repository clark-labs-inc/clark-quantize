#!/usr/bin/env python3
"""Activation-weighted gain-routed Clark-Hash screen on Qwen3.8-27B."""
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

from clark_quantize.qwen38_activation_gain_hash_residual import (
    activation_weighted_gain_hash_coefficients,
    output_error_stats,
    train_int4_activation_gain_hash_coefficients,
)
from clark_quantize.qwen38_gain_hash_residual import (
    GainHashProjection,
    expand_gain_hash,
    fit_fixed_gain_hash_ternary_residual,
    gain_hash_sketch_rows,
    gain_hash_storage_projection,
    learn_gain_hash_routing,
)
from clark_quantize.qwen38_hash_residual import (
    fixed_support_ternary,
    quantize_hash_coefficients_int4,
)
from clark_quantize.qwen38_learned_hash_residual import (
    LearnedHashProjection,
    learn_signed_hash_routing,
)


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
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


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
            raise RuntimeError(f"frozen activation gain-hash source changed: {label}")


def route_digest(projection: GainHashProjection) -> str:
    digest = hashlib.sha256()
    digest.update(projection.buckets.numpy().astype("<u2", copy=False).tobytes())
    digest.update(projection.signs.numpy().astype("i1", copy=False).tobytes())
    digest.update(projection.gain_codes.numpy().astype("u1", copy=False).tobytes())
    digest.update(
        projection.gain_codebook.numpy().astype("<f2", copy=False).tobytes()
    )
    return digest.hexdigest()


def load_span(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    tensor: dict[str, Any],
    span_name: str,
) -> torch.Tensor:
    name = tensor["name"]
    shard = weight_map.get(name)
    if shard != tensor["shard"]:
        raise RuntimeError(f"activation gain-hash tensor shard changed: {name}")
    start, stop = tensor[span_name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        if list(view.get_shape()) != tensor["shape"]:
            raise RuntimeError(f"activation gain-hash tensor shape changed: {name}")
        return view[start:stop].contiguous().reshape(stop - start, -1).float()


def unity_gain_projection(projection: LearnedHashProjection) -> GainHashProjection:
    return GainHashProjection(
        input_dim=projection.input_dim,
        sketch_dim=projection.sketch_dim,
        gain_bits=1,
        buckets=projection.buckets.clone(),
        signs=projection.signs.clone(),
        gain_codes=torch.zeros(projection.input_dim, dtype=torch.uint8),
        gain_codebook=torch.ones(2, dtype=torch.float16),
    )


def ternary_export(
    codes: torch.Tensor, scales: torch.Tensor, *, group: int
) -> torch.Tensor:
    frozen = scales.to(torch.float16).to(torch.float32)
    return (
        codes.to(torch.float32).reshape(codes.shape[0], -1, group)
        * frozen[:, :, None]
    ).reshape_as(codes)


def full_output_stats(
    source: torch.Tensor,
    ternary: torch.Tensor,
    coefficients: torch.Tensor,
    inputs: torch.Tensor,
    projection: GainHashProjection,
) -> dict[str, float]:
    target = inputs.float() @ source.float().T
    candidate = (
        inputs.float() @ ternary.float().T
        + gain_hash_sketch_rows(inputs.float(), projection)
        @ coefficients.float().T
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


def half_support_baseline_stats(
    source: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group: int,
    support: int,
) -> dict[str, float]:
    codes, scales, _ = fixed_support_ternary(
        source, group=group, nonzero_per_group=support
    )
    ternary = ternary_export(codes, scales, group=group)
    target = inputs.float() @ source.float().T
    difference = inputs.float() @ ternary.T - target
    squared_error = float(difference.square().sum(dtype=torch.float64).item())
    target_energy = float(target.square().sum(dtype=torch.float64).item())
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def fit_activation_candidate(
    source: torch.Tensor,
    calibration: torch.Tensor,
    validation: torch.Tensor,
    projection: GainHashProjection,
    *,
    group: int,
    support: int,
    coordinate_iterations: int,
    training: dict[str, Any],
) -> dict[str, Any]:
    fit = fit_fixed_gain_hash_ternary_residual(
        source,
        projection,
        group=group,
        nonzero_per_group=support,
        iterations=coordinate_iterations,
    )
    ternary = ternary_export(fit.ternary_codes, fit.ternary_scales, group=group)
    residual = source - ternary
    data_free_codes, data_free_scales, data_free_export = (
        quantize_hash_coefficients_int4(fit.hash_coefficients)
    )
    activation_float = activation_weighted_gain_hash_coefficients(
        residual,
        calibration,
        projection,
        reference_coefficients=fit.hash_coefficients,
        relative_ridge=float(training["relative_ridge"]),
    )
    activation_codes, activation_scales, activation_export = (
        quantize_hash_coefficients_int4(activation_float)
    )
    trained = train_int4_activation_gain_hash_coefficients(
        residual,
        calibration,
        projection,
        activation_float,
        reference_coefficients=fit.hash_coefficients,
        steps=int(training["steps"]),
        learning_rate=float(training["learning_rate"]),
        anchor_weight=float(training["data_free_anchor_weight"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    data_free_calibration = full_output_stats(
        source, ternary, data_free_export, calibration, projection
    )
    data_free_validation = full_output_stats(
        source, ternary, data_free_export, validation, projection
    )
    activation_float_calibration = output_error_stats(
        residual, activation_float, calibration, projection
    )
    activation_int4_calibration = output_error_stats(
        residual, activation_export, calibration, projection
    )
    trained_calibration = full_output_stats(
        source, ternary, trained.coefficients, calibration, projection
    )
    trained_validation = full_output_stats(
        source, ternary, trained.coefficients, validation, projection
    )
    implicit = (
        validation @ ternary.T
        + gain_hash_sketch_rows(validation, projection) @ trained.coefficients.T
    )
    expanded = validation @ (
        ternary + expand_gain_hash(trained.coefficients, projection)
    ).T
    return {
        "fit": fit,
        "ternary": ternary,
        "data_free_calibration": data_free_calibration,
        "data_free_validation": data_free_validation,
        "activation_float_calibration_residual": activation_float_calibration,
        "activation_int4_calibration_residual": activation_int4_calibration,
        "trained_calibration": trained_calibration,
        "trained_validation": trained_validation,
        "data_free_int4_codes_sha256": tensor_sha256(data_free_codes),
        "data_free_int4_scales_sha256": tensor_sha256(data_free_scales),
        "activation_int4_codes_sha256": tensor_sha256(activation_codes),
        "activation_int4_scales_sha256": tensor_sha256(activation_scales),
        "trained_int4_codes_sha256": tensor_sha256(trained.codes),
        "trained_int4_scales_sha256": tensor_sha256(trained.scales),
        "training_best_step": trained.best_step,
        "training_history": list(trained.calibration_history),
        "kernel_identity_max_abs_error": float((implicit - expanded).abs().max().item()),
    }


def aggregate(records: list[dict[str, Any]], path: tuple[str, str]) -> dict[str, float]:
    outer, inner = path
    squared_error = sum(
        float(record[outer][inner]["squared_error"]) for record in records
    )
    target_energy = sum(
        float(record[outer][inner]["target_squared_sum"]) for record in records
    )
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_energy,
        "normalized_mse": squared_error / max(target_energy, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-27b-layer0-mlp-activation-qat-v1/activations.pt"
        ),
    )
    parser.add_argument(
        "--activation-cache-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json"
        ),
    )
    parser.add_argument(
        "--gain-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-gain-hash-routing-v1.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-activation-gain-hash-residual-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / (
        "qwen38_27b_activation_gain_hash_residual_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    if (
        prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != REVISION
    ):
        raise ValueError("activation gain-hash Qwen3.8-27B identity changed")

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download.get("model_id") != MODEL_ID or download.get("revision") != REVISION:
        raise ValueError("activation gain-hash checkpoint receipt mismatch")
    index_path = args.model_dir / "model.safetensors.index.json"
    index_content = index_path.read_bytes()
    if hashlib.sha256(index_content).hexdigest() != INDEX_SHA256:
        raise ValueError("activation gain-hash Qwen3.8-27B index changed")
    weight_map = json.loads(index_content)["weight_map"]

    parent_prereg_path = runtime / "qwen38_27b_gain_hash_routing_preregistration_v1.json"
    parent_prereg = json.loads(parent_prereg_path.read_text())
    gain_receipt = json.loads(args.gain_receipt.read_text())
    if (
        gain_receipt.get("schema") != "qwen38_27b_gain_hash_routing_receipt_v1"
        or gain_receipt.get("complete") is not True
        or gain_receipt.get("terminal") is not True
        or gain_receipt.get("gate", {}).get("passed") is not True
        or gain_receipt.get("preregistration_sha256")
        != sha256_file(parent_prereg_path)
    ):
        raise ValueError("activation gain-hash parent is not a passing receipt")

    activation_qat_prereg_path = runtime / (
        "qwen38_27b_layer0_mlp_activation_qat_preregistration_v1.json"
    )
    activation_qat_prereg = json.loads(activation_qat_prereg_path.read_text())
    activation_qat_prereg_sha256 = sha256_file(activation_qat_prereg_path)
    cache_receipt = json.loads(args.activation_cache_receipt.read_text())
    if (
        cache_receipt.get("schema")
        != "qwen38_27b_layer0_mlp_activation_cache_receipt_v1"
        or cache_receipt.get("model_id") != MODEL_ID
        or cache_receipt.get("model_revision") != REVISION
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("preregistration", {}).get("sha256")
        != activation_qat_prereg_sha256
        or cache_receipt.get("cache", {}).get("sha256")
        != sha256_file(args.activation_cache)
    ):
        raise ValueError("activation gain-hash cache receipt is invalid")
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
        raise ValueError("activation gain-hash cache payload is invalid")

    family = parent_prereg["candidate"]
    group = int(family["group_size"])
    support = int(family["nonzero_per_group"])
    ratio = int(family["input_to_sketch_ratio"])
    gain_bits = int(family["gain_bits"])
    training = prereg["training"]
    train: dict[str, torch.Tensor] = {}
    evaluation: dict[str, torch.Tensor] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for tensor in parent_prereg["scope"]["tensors"]:
        name = tensor["name"]
        tensors[name] = tensor
        train[name] = load_span(
            model_dir=args.model_dir,
            weight_map=weight_map,
            tensor=tensor,
            span_name="train_row_span",
        )
        evaluation[name] = load_span(
            model_dir=args.model_dir,
            weight_map=weight_map,
            tensor=tensor,
            span_name="evaluation_row_span",
        )

    parent_routing = gain_receipt["routing_training"]
    gain_routes: dict[str, GainHashProjection] = {}
    control_routes: dict[str, GainHashProjection] = {}
    replay = {}
    for route_spec in parent_prereg["scope"]["shared_routing_groups"]:
        route_name = route_spec["name"]
        source = torch.cat([train[name] for name in route_spec["members"]], dim=0)
        _, _, base = fixed_support_ternary(
            source, group=group, nonzero_per_group=support
        )
        residual = source - base
        gain = learn_gain_hash_routing(
            residual,
            sketch_dim=source.shape[1] // ratio,
            gain_bits=gain_bits,
            iterations=int(family["routing_iterations"]),
        )
        control = learn_signed_hash_routing(
            residual,
            sketch_dim=source.shape[1] // ratio,
            iterations=int(family["routing_iterations"]),
        )
        digest = route_digest(gain.projection)
        if digest != parent_routing[route_name]["route_sha256"]:
            raise ValueError(f"activation gain-hash route replay changed: {route_name}")
        gain_routes[route_name] = gain.projection
        control_routes[route_name] = unity_gain_projection(control.projection)
        replay[route_name] = {
            "route_sha256": digest,
            "training_rows": source.shape[0],
            "input_dim": source.shape[1],
            "sketch_dim": gain.projection.sketch_dim,
        }

    route_by_tensor = {
        member: route["name"]
        for route in parent_prereg["scope"]["shared_routing_groups"]
        for member in route["members"]
    }
    parent_records = {record["tensor"]: record for record in gain_receipt["records"]}
    records: list[dict[str, Any]] = []
    for name, source in evaluation.items():
        route_name = route_by_tensor[name]
        activation_prefix = "down_input" if ".down_proj." in name else "mlp_input"
        calibration = cache[f"{activation_prefix}_calibration"].float().contiguous()
        validation = cache[f"{activation_prefix}_validation"].float().contiguous()
        if (
            not bool(torch.isfinite(calibration).all())
            or not bool(torch.isfinite(validation).all())
        ):
            raise ValueError("activation gain-hash cache contains non-finite values")
        gain = fit_activation_candidate(
            source,
            calibration,
            validation,
            gain_routes[route_name],
            group=group,
            support=support,
            coordinate_iterations=int(family["coordinate_iterations"]),
            training=training,
        )
        control = fit_activation_candidate(
            source,
            calibration,
            validation,
            control_routes[route_name],
            group=group,
            support=support,
            coordinate_iterations=int(family["coordinate_iterations"]),
            training=training,
        )
        parent = parent_records[name]
        if (
            not math.isclose(
                gain["fit"].exported_squared_error,
                float(parent["gain_hash"]["squared_error"]),
                rel_tol=2e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                control["fit"].exported_squared_error,
                float(parent["learned_sign_only_control"]["squared_error"]),
                rel_tol=2e-6,
                abs_tol=1e-7,
            )
        ):
            raise ValueError(f"activation gain-hash parent replay changed: {name}")
        storage = gain_hash_storage_projection(
            input_dim=source.shape[1],
            output_rows=int(tensors[name]["shape"][0]),
            sketch_dim=gain_routes[route_name].sketch_dim,
            gain_bits=gain_bits,
            nonzero_per_group=support,
            group=group,
        )
        baseline = half_support_baseline_stats(
            source, validation, group=group, support=support
        )
        gain_record = {key: value for key, value in gain.items() if key not in {"fit", "ternary"}}
        control_record = {
            key: value for key, value in control.items() if key not in {"fit", "ternary"}
        }
        records.append(
            {
                "tensor": name,
                "route_group": route_name,
                "train_row_span": tensors[name]["train_row_span"],
                "evaluation_row_span": tensors[name]["evaluation_row_span"],
                "parameters": source.numel(),
                "source_sha256": tensor_sha256(source),
                "activation_prefix": activation_prefix,
                "gain_hash": gain_record,
                "activation_trained_sign_only_control": control_record,
                "half_support_ternary_validation": baseline,
                "complete_component_storage_projection": storage,
                "relative_validation_error_reduction_vs_data_free_gain": (
                    1.0
                    - gain_record["trained_validation"]["squared_error"]
                    / max(gain_record["data_free_validation"]["squared_error"], 1e-30)
                ),
                "relative_validation_error_reduction_vs_trained_sign_only": (
                    1.0
                    - gain_record["trained_validation"]["squared_error"]
                    / max(control_record["trained_validation"]["squared_error"], 1e-30)
                ),
                "relative_validation_error_reduction_vs_half_support": (
                    1.0
                    - gain_record["trained_validation"]["squared_error"]
                    / max(baseline["squared_error"], 1e-30)
                ),
            }
        )

    summary = {
        "data_free_gain_validation": aggregate(
            records, ("gain_hash", "data_free_validation")
        ),
        "trained_gain_calibration": aggregate(
            records, ("gain_hash", "trained_calibration")
        ),
        "trained_gain_validation": aggregate(
            records, ("gain_hash", "trained_validation")
        ),
        "trained_sign_only_validation": aggregate(
            records, ("activation_trained_sign_only_control", "trained_validation")
        ),
    }
    baseline_error = sum(
        float(record["half_support_ternary_validation"]["squared_error"])
        for record in records
    )
    summary["relative_validation_error_reduction_vs_data_free_gain"] = (
        1.0
        - summary["trained_gain_validation"]["squared_error"]
        / max(summary["data_free_gain_validation"]["squared_error"], 1e-30)
    )
    summary["relative_validation_error_reduction_vs_trained_sign_only"] = (
        1.0
        - summary["trained_gain_validation"]["squared_error"]
        / max(summary["trained_sign_only_validation"]["squared_error"], 1e-30)
    )
    summary["relative_validation_error_reduction_vs_half_support"] = (
        1.0
        - summary["trained_gain_validation"]["squared_error"]
        / max(baseline_error, 1e-30)
    )

    maximum_tensor_regression = max(
        0.0,
        max(
            -float(record["relative_validation_error_reduction_vs_data_free_gain"])
            for record in records
        ),
    )
    maximum_rate = max(
        float(record["complete_component_storage_projection"]["bits_per_weight"])
        for record in records
    )
    maximum_identity = max(
        max(
            float(record[family_name]["kernel_identity_max_abs_error"])
            for family_name in ("gain_hash", "activation_trained_sign_only_control")
        )
        for record in records
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_tensor_spans": len(records),
        "observed_relative_validation_error_reduction_vs_data_free_gain": summary[
            "relative_validation_error_reduction_vs_data_free_gain"
        ],
        "observed_relative_validation_error_reduction_vs_trained_sign_only": summary[
            "relative_validation_error_reduction_vs_trained_sign_only"
        ],
        "observed_relative_validation_error_reduction_vs_half_support": summary[
            "relative_validation_error_reduction_vs_half_support"
        ],
        "observed_maximum_tensor_relative_validation_regression_vs_data_free_gain": maximum_tensor_regression,
        "observed_maximum_component_bits_per_weight": maximum_rate,
        "observed_maximum_kernel_identity_error": maximum_identity,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_tensor_spans"])
        and summary["relative_validation_error_reduction_vs_data_free_gain"]
        >= float(gate_spec["minimum_relative_validation_error_reduction_vs_data_free_gain"])
        and summary["relative_validation_error_reduction_vs_trained_sign_only"]
        >= float(gate_spec["minimum_relative_validation_error_reduction_vs_trained_sign_only"])
        and summary["relative_validation_error_reduction_vs_half_support"] >= 0.0
        and maximum_tensor_regression
        <= float(gate_spec["maximum_tensor_relative_validation_regression_vs_data_free_gain"])
        and maximum_rate <= float(gate_spec["maximum_component_bits_per_weight"])
        and maximum_identity <= float(gate_spec["maximum_kernel_identity_error"])
    )

    receipt = {
        "schema": "qwen38_27b_activation_gain_hash_residual_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent_gain_receipt": {
            "path": str(args.gain_receipt),
            "sha256": sha256_file(args.gain_receipt),
        },
        "activation_cache_receipt": {
            "path": str(args.activation_cache_receipt),
            "sha256": sha256_file(args.activation_cache_receipt),
        },
        "route_replay": replay,
        "records": records,
        "summary": summary,
        "gate": gate,
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "real_Qwen38_27B_weights_measured": True,
            "real_disjoint_calibration_and_validation_activations_measured": True,
            "passing_parent_gain_route_replayed_and_authenticated": True,
            "gain_and_sign_only_controls_trained_identically": True,
            "sampled_rows_only": True,
            "full_layer_or_model_fitted": False,
            "capability_preservation_proven": False,
            "complete_storage_proven": False,
            "native_fused_gain_kernel_proven": False,
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
