#!/usr/bin/env python3
"""Fit and validate a complete activation-calibrated Qwen3.8-27B MLP."""
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
import tempfile
from typing import Any

from safetensors import safe_open
import torch
import torch.nn.functional as functional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_activation_binary_mlp import (
    complete_mlp_storage_projection,
    error_metrics,
    exported_int4_left,
    exported_row_normalized_ternary,
    fit_activation_left_int4,
)
from clark_quantize.qwen38_binary_low_rank_residual import (
    BinaryRightFactor,
    binary_low_rank_outputs,
    binary_right_values,
    fit_binary_low_rank_residual,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
GATE = "model.language_model.layers.0.mlp.gate_proj.weight"
UP = "model.language_model.layers.0.mlp.up_proj.weight"
DOWN = "model.language_model.layers.0.mlp.down_proj.weight"
SHAPES = {
    GATE: (17_408, 5_120),
    UP: (17_408, 5_120),
    DOWN: (5_120, 17_408),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: torch.Tensor, *, chunk_rows: int = 256) -> str:
    digest = hashlib.sha256()
    for start in range(0, values.shape[0], chunk_rows):
        block = values[start : start + chunk_rows].contiguous()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_frozen_inputs(preregistration: dict[str, Any]) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen activation-binary source changed: {label}")


def load_weight(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    shard = weight_map.get(name)
    if not shard:
        raise RuntimeError(f"missing Qwen3.8 tensor: {name}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        source = handle.get_tensor(name)
    if tuple(source.shape) != SHAPES[name] or source.dtype != torch.bfloat16:
        raise RuntimeError(f"Qwen3.8 source identity changed: {name}")
    record = {
        "name": name,
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "shard": shard,
        "source_BF16_sha256": tensor_sha256(source),
    }
    return source.float().contiguous(), record


def reduction(candidate_error: float, control_error: float) -> float:
    return 1.0 - candidate_error / max(control_error, 1e-30)


def factor_identity_error(
    inputs: torch.Tensor,
    coefficients: torch.Tensor,
    right: BinaryRightFactor,
) -> float:
    rows = min(8, inputs.shape[0])
    outputs = min(32, coefficients.shape[0])
    implicit = binary_low_rank_outputs(
        inputs[:rows], coefficients[:outputs], right
    )
    expanded = inputs[:rows].to(torch.float32) @ (
        coefficients[:outputs].to(torch.float32) @ binary_right_values(right)
    ).T
    return float((implicit - expanded).abs().max().item())


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json"),
    )
    parser.add_argument(
        "--activation-cache-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.pt"),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_activation_binary_mlp_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if (
        download.get("model_id") != MODEL_ID
        or download.get("revision") != REVISION
        or download.get("index_sha256") != INDEX_SHA256
    ):
        raise RuntimeError("Qwen3.8 checkpoint receipt changed")
    index_path = args.model_dir / "model.safetensors.index.json"
    if sha256_file(index_path) != INDEX_SHA256:
        raise RuntimeError("Qwen3.8 tensor index changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema") != "qwen38_27b_binary_low_rank_residual_receipt_v1"
        or parent.get("model") != MODEL_ID
        or parent.get("revision") != REVISION
        or parent.get("complete") is not True
        or parent.get("terminal") is not True
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("binary-low-rank parent did not pass its frozen gate")
    parent_artifact_path = Path(parent["candidate"]["factor_artifact"])
    if sha256_file(parent_artifact_path) != parent["candidate"]["factor_artifact_sha256"]:
        raise RuntimeError("binary-low-rank parent artifact changed")
    parent_artifact = torch.load(
        parent_artifact_path, map_location="cpu", weights_only=False
    )
    if (
        parent_artifact.get("schema")
        != "qwen38_27b_binary_low_rank_factor_artifact_v1"
        or parent_artifact.get("tensor_names") != [GATE, UP]
        or parent_artifact.get("split_rows") != [17_408, 17_408]
    ):
        raise RuntimeError("binary-low-rank factor artifact identity changed")
    gate_up_right = BinaryRightFactor(
        signs=parent_artifact["right_signs"],
        scales=parent_artifact["right_scales"],
    )
    parent_left = exported_int4_left(
        parent_artifact["left_codes"], parent_artifact["left_scales"]
    )
    parent_left_parts = parent_left.split([17_408, 17_408], dim=0)

    cache_receipt_content = args.activation_cache_receipt.read_bytes()
    cache_receipt = json.loads(cache_receipt_content)
    cache_path = Path(cache_receipt.get("cache", {}).get("path", ""))
    if (
        cache_receipt.get("schema")
        != "qwen38_27b_layer0_mlp_activation_cache_receipt_v1"
        or cache_receipt.get("model_id") != MODEL_ID
        or cache_receipt.get("model_revision") != REVISION
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("finite") is not True
        or not cache_path.is_file()
        or sha256_file(cache_path) != cache_receipt.get("cache", {}).get("sha256")
    ):
        raise RuntimeError("Qwen3.8 activation-cache receipt is invalid")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    calibration = cache["mlp_input_calibration"].to(torch.float32)
    validation = cache["mlp_input_validation"].to(torch.float32)
    calibration_rows = cache_receipt.get("rows", {}).get("mlp_calibration")
    validation_rows = cache_receipt.get("rows", {}).get("mlp_validation")
    if (
        not isinstance(calibration_rows, int)
        or not 0 < calibration_rows <= 256
        or not isinstance(validation_rows, int)
        or not 0 < validation_rows <= 96
        or tuple(calibration.shape) != (calibration_rows, 5_120)
        or tuple(validation.shape) != (validation_rows, 5_120)
        or not bool(torch.isfinite(calibration).all())
        or not bool(torch.isfinite(validation).all())
    ):
        raise RuntimeError("Qwen3.8 activation-cache matrix dimensions changed")

    candidate = prereg["candidate"]
    rank = int(candidate["rank"])
    group = int(candidate["group_size"])
    support = int(candidate["nonzero_per_group"])
    ridge = float(candidate["activation_relative_ridge"])
    right_values = binary_right_values(gate_up_right)
    gate_up_features_cal = calibration @ right_values.T
    gate_up_features_val = validation @ right_values.T

    projection_records: dict[str, Any] = {}
    activation_left_fits = []
    stage_outputs: dict[str, dict[str, torch.Tensor]] = {}
    source_records = []
    identity_errors = []
    for name, parent_coefficients in zip((GATE, UP), parent_left_parts):
        source, source_record = load_weight(args.model_dir, weight_map, name)
        ternary = exported_row_normalized_ternary(
            source, group=group, nonzero_per_group=support
        )
        reference_cal = functional.linear(calibration, source)
        reference_val = functional.linear(validation, source)
        direct_cal = functional.linear(calibration, ternary.weights)
        direct_val = functional.linear(validation, ternary.weights)
        target_cal = reference_cal - direct_cal
        fit = fit_activation_left_int4(
            gate_up_features_cal, target_cal, relative_ridge=ridge
        )
        candidate_cal = direct_cal + gate_up_features_cal @ fit.coefficients.T
        candidate_val = direct_val + gate_up_features_val @ fit.coefficients.T
        weight_val = direct_val + gate_up_features_val @ parent_coefficients.T
        direct_metrics = error_metrics(reference_val, direct_val)
        candidate_metrics = error_metrics(reference_val, candidate_val)
        weight_metrics = error_metrics(reference_val, weight_val)
        projection_records[name] = {
            "source": source_record,
            "scale_export": ternary.scale_record,
            "validation": {
                "direct_ternary": direct_metrics,
                "weight_space_binary": weight_metrics,
                "activation_calibrated_binary": candidate_metrics,
                "activation_error_reduction_vs_direct": reduction(
                    candidate_metrics["squared_error"],
                    direct_metrics["squared_error"],
                ),
                "activation_error_reduction_vs_weight_space": reduction(
                    candidate_metrics["squared_error"],
                    weight_metrics["squared_error"],
                ),
            },
        }
        activation_left_fits.append(fit)
        source_records.append(source_record)
        identity_errors.append(
            factor_identity_error(validation, fit.coefficients, gate_up_right)
        )
        stage_outputs[name] = {
            "reference_cal": reference_cal,
            "reference_val": reference_val,
            "direct_cal": direct_cal,
            "direct_val": direct_val,
            "candidate_cal": candidate_cal,
            "candidate_val": candidate_val,
        }
        del source, ternary, target_cal, weight_val

    reference_intermediate_cal = functional.silu(
        stage_outputs[GATE]["reference_cal"]
    ) * stage_outputs[UP]["reference_cal"]
    reference_intermediate_val = functional.silu(
        stage_outputs[GATE]["reference_val"]
    ) * stage_outputs[UP]["reference_val"]
    direct_intermediate_cal = functional.silu(
        stage_outputs[GATE]["direct_cal"]
    ) * stage_outputs[UP]["direct_cal"]
    direct_intermediate_val = functional.silu(
        stage_outputs[GATE]["direct_val"]
    ) * stage_outputs[UP]["direct_val"]
    candidate_intermediate_cal = functional.silu(
        stage_outputs[GATE]["candidate_cal"]
    ) * stage_outputs[UP]["candidate_cal"]
    candidate_intermediate_val = functional.silu(
        stage_outputs[GATE]["candidate_val"]
    ) * stage_outputs[UP]["candidate_val"]
    intermediate_direct_metrics = error_metrics(
        reference_intermediate_val, direct_intermediate_val
    )
    intermediate_candidate_metrics = error_metrics(
        reference_intermediate_val, candidate_intermediate_val
    )

    down_source, down_source_record = load_weight(args.model_dir, weight_map, DOWN)
    down_ternary = exported_row_normalized_ternary(
        down_source, group=group, nonzero_per_group=support
    )
    down_weight_fit = fit_binary_low_rank_residual(
        down_source - down_ternary.weights,
        rank=rank,
        iterations=int(candidate["down_alternating_iterations"]),
        pca_iterations=int(candidate["down_randomized_pca_iterations"]),
        seed=int(candidate["down_seed"]),
        relative_ridge=float(candidate["weight_relative_ridge"]),
    )
    down_right_values = binary_right_values(down_weight_fit.right)
    down_features_cal = candidate_intermediate_cal @ down_right_values.T
    down_features_val = candidate_intermediate_val @ down_right_values.T
    reference_output_cal = functional.linear(
        reference_intermediate_cal, down_source
    )
    reference_output_val = functional.linear(
        reference_intermediate_val, down_source
    )
    base_candidate_cal = functional.linear(
        candidate_intermediate_cal, down_ternary.weights
    )
    base_candidate_val = functional.linear(
        candidate_intermediate_val, down_ternary.weights
    )
    down_activation_fit = fit_activation_left_int4(
        down_features_cal,
        reference_output_cal - base_candidate_cal,
        relative_ridge=ridge,
    )
    candidate_output_cal = (
        base_candidate_cal + down_features_cal @ down_activation_fit.coefficients.T
    )
    candidate_output_val = (
        base_candidate_val + down_features_val @ down_activation_fit.coefficients.T
    )
    weight_output_val = (
        base_candidate_val
        + down_features_val @ down_weight_fit.left_coefficients.T
    )
    direct_output_val = functional.linear(
        direct_intermediate_val, down_ternary.weights
    )
    direct_output_metrics = error_metrics(reference_output_val, direct_output_val)
    weight_output_metrics = error_metrics(reference_output_val, weight_output_val)
    candidate_output_metrics = error_metrics(
        reference_output_val, candidate_output_val
    )
    calibration_output_metrics = error_metrics(
        reference_output_cal, candidate_output_cal
    )
    down_direct_weight_error = down_weight_fit.initial_squared_error
    down_binary_weight_error = min(down_weight_fit.objective_history)
    down_weight_reduction = reduction(
        down_binary_weight_error, down_direct_weight_error
    )
    identity_errors.append(
        factor_identity_error(
            candidate_intermediate_val,
            down_activation_fit.coefficients,
            down_weight_fit.right,
        )
    )
    source_records.append(down_source_record)

    storage = complete_mlp_storage_projection(
        hidden=5_120,
        intermediate=17_408,
        rank=rank,
        group=group,
        nonzero_per_group=support,
    )
    full_reduction = reduction(
        candidate_output_metrics["squared_error"],
        direct_output_metrics["squared_error"],
    )
    gate_spec = prereg["gate"]
    per_projection_reductions = [
        projection_records[name]["validation"][
            "activation_error_reduction_vs_direct"
        ]
        for name in (GATE, UP)
    ]
    maximum_identity_error = max(identity_errors)
    gate = {
        **gate_spec,
        "observed_full_tensors": len(source_records),
        "observed_gate_up_minimum_validation_error_reduction_vs_direct": min(
            per_projection_reductions
        ),
        "observed_intermediate_validation_error_reduction_vs_direct": reduction(
            intermediate_candidate_metrics["squared_error"],
            intermediate_direct_metrics["squared_error"],
        ),
        "observed_down_weight_error_reduction_vs_direct": down_weight_reduction,
        "observed_full_mlp_validation_error_reduction_vs_direct": full_reduction,
        "observed_full_mlp_validation_relative_rmse": candidate_output_metrics[
            "relative_rmse"
        ],
        "observed_complete_mlp_bits_per_weight": storage["bits_per_weight"],
        "observed_kernel_identity_max_abs_error": maximum_identity_error,
    }
    gate["passed"] = bool(
        len(source_records) == int(gate_spec["required_full_tensors"])
        and min(per_projection_reductions)
        >= float(gate_spec["minimum_gate_up_validation_error_reduction_vs_direct"])
        and down_weight_reduction
        >= float(gate_spec["minimum_down_weight_error_reduction_vs_direct"])
        and full_reduction
        >= float(gate_spec["minimum_full_mlp_validation_error_reduction_vs_direct"])
        and candidate_output_metrics["relative_rmse"]
        <= float(gate_spec["maximum_full_mlp_validation_relative_rmse"])
        and storage["bits_per_weight"]
        <= float(gate_spec["maximum_complete_mlp_bits_per_weight"])
        and maximum_identity_error
        <= float(gate_spec["maximum_kernel_identity_error"])
    )

    artifact = {
        "schema": "qwen38_27b_activation_binary_mlp_factor_artifact_v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
        "activation_cache_receipt_sha256": hashlib.sha256(
            cache_receipt_content
        ).hexdigest(),
        "rank": rank,
        "gate_up": {
            "tensor_names": [GATE, UP],
            "right_signs": gate_up_right.signs.cpu(),
            "right_scales": gate_up_right.scales.cpu(),
            "left_codes": torch.cat(
                [fit.codes for fit in activation_left_fits], dim=0
            ).cpu(),
            "left_scales": torch.cat(
                [fit.scales for fit in activation_left_fits], dim=0
            ).cpu(),
            "split_rows": [17_408, 17_408],
        },
        "down": {
            "tensor_name": DOWN,
            "right_signs": down_weight_fit.right.signs.cpu(),
            "right_scales": down_weight_fit.right.scales.cpu(),
            "left_codes": down_activation_fit.codes.cpu(),
            "left_scales": down_activation_fit.scales.cpu(),
        },
    }
    atomic_torch_save(artifact, args.artifact)
    artifact_sha256 = sha256_file(args.artifact)
    receipt = {
        "schema": "qwen38_27b_activation_binary_mlp_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "model": MODEL_ID,
        "revision": REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parents": {
            "binary_low_rank_receipt": str(args.parent_receipt),
            "binary_low_rank_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
            "activation_cache_receipt": str(args.activation_cache_receipt),
            "activation_cache_receipt_sha256": hashlib.sha256(
                cache_receipt_content
            ).hexdigest(),
        },
        "activation_rows": {
            "calibration": int(calibration.shape[0]),
            "validation": int(validation.shape[0]),
        },
        "projections": projection_records,
        "intermediate_validation": {
            "direct_ternary": intermediate_direct_metrics,
            "activation_calibrated_binary": intermediate_candidate_metrics,
        },
        "down": {
            "source": down_source_record,
            "scale_export": down_ternary.scale_record,
            "weight_fit": {
                "objective_history": list(down_weight_fit.objective_history),
                "best_iteration": down_weight_fit.best_iteration,
                "direct_ternary_squared_error": down_direct_weight_error,
                "binary_squared_error": down_binary_weight_error,
                "error_reduction_vs_direct": down_weight_reduction,
            },
        },
        "complete_mlp": {
            "calibration": calibration_output_metrics,
            "validation": {
                "direct_complete_ternary": direct_output_metrics,
                "weight_space_binary": weight_output_metrics,
                "activation_calibrated_binary": candidate_output_metrics,
                "activation_error_reduction_vs_direct": full_reduction,
                "activation_error_reduction_vs_weight_space": reduction(
                    candidate_output_metrics["squared_error"],
                    weight_output_metrics["squared_error"],
                ),
            },
        },
        "storage": storage,
        "factor_artifact": {
            "path": str(args.artifact),
            "bytes": args.artifact.stat().st_size,
            "sha256": artifact_sha256,
            "contains_ternary_backbone": False,
        },
        "gate": gate,
        "claim_boundary": {
            "full_real_layer0_gate_up_down_weights": True,
            "real_disjoint_layer0_activations": True,
            "whole_model_artifact_proven": False,
            "free_generation_or_capability_proven": False,
            "end_to_end_CPU_throughput_proven": False,
            "scientific_novelty_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
