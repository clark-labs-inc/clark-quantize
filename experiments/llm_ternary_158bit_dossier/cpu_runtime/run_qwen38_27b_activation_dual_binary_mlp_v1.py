#!/usr/bin/env python3
"""Fit and validate activation-dual binary factors on a complete Qwen MLP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

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
)
from clark_quantize.qwen38_activation_dual_binary import (
    activation_dual_outputs,
    fit_activation_dual_binary_residual,
)
from clark_quantize.qwen38_binary_low_rank_residual import BinaryRightFactor
from run_qwen38_27b_activation_binary_mlp_v1 import (
    DOWN,
    GATE,
    INDEX_SHA256,
    MODEL_ID,
    REVISION,
    SHAPES,
    UP,
    atomic_torch_save,
    factor_identity_error,
    load_weight,
    reduction,
    sha256_file,
)


def validate_frozen_inputs(descriptors: dict[str, Any]) -> None:
    for label, descriptor in descriptors.items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen activation-dual source changed: {label}")


def metric_close(observed: dict[str, float], expected: dict[str, float]) -> bool:
    return observed.keys() == expected.keys() and all(
        math.isclose(
            float(observed[key]),
            float(expected[key]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        for key in observed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--weight-parent-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json"),
    )
    parser.add_argument(
        "--frozen-right-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.json"
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
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-activation-dual-binary-mlp-v1.json"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-activation-dual-binary-mlp-v1.pt"
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    method_path = (
        runtime
        / "qwen38_27b_activation_dual_binary_method_preregistration_v1.json"
    )
    execution_path = (
        runtime
        / "qwen38_27b_activation_dual_binary_execution_amendment_v1.json"
    )
    method = json.loads(method_path.read_text())
    execution = json.loads(execution_path.read_text())
    if (
        method.get("schema")
        != "qwen38_27b_activation_dual_binary_method_preregistration_v1"
        or method.get("model") != MODEL_ID
        or method.get("revision") != REVISION
        or method.get("execution_authorized") is not False
        or execution.get("schema")
        != "qwen38_27b_activation_dual_binary_execution_amendment_v1"
        or execution.get("model") != MODEL_ID
        or execution.get("revision") != REVISION
        or execution.get("execution_authorized") is not True
        or execution.get("method_preregistration_sha256")
        != sha256_file(method_path)
    ):
        raise RuntimeError("activation-dual execution contract is invalid")
    validate_frozen_inputs(method["frozen_inputs"])
    validate_frozen_inputs(execution["frozen_inputs"])

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

    weight_parent_content = args.weight_parent_receipt.read_bytes()
    weight_parent = json.loads(weight_parent_content)
    if (
        weight_parent.get("schema")
        != "qwen38_27b_binary_low_rank_residual_receipt_v1"
        or weight_parent.get("model") != MODEL_ID
        or weight_parent.get("revision") != REVISION
        or weight_parent.get("complete") is not True
        or weight_parent.get("terminal") is not True
        or weight_parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("activation-dual weight parent did not pass")

    frozen_content = args.frozen_right_receipt.read_bytes()
    frozen_receipt = json.loads(frozen_content)
    frozen_artifact_path = Path(
        frozen_receipt.get("factor_artifact", {}).get("path", "")
    )
    if (
        frozen_receipt.get("schema")
        != "qwen38_27b_activation_binary_mlp_receipt_v1"
        or frozen_receipt.get("model") != MODEL_ID
        or frozen_receipt.get("revision") != REVISION
        or frozen_receipt.get("complete") is not True
        or frozen_receipt.get("terminal") is not True
        or frozen_receipt.get("parents", {}).get("binary_low_rank_receipt_sha256")
        != hashlib.sha256(weight_parent_content).hexdigest()
        or not frozen_artifact_path.is_file()
        or sha256_file(frozen_artifact_path)
        != frozen_receipt.get("factor_artifact", {}).get("sha256")
    ):
        raise RuntimeError("frozen-right activation receipt is invalid")
    frozen_artifact = torch.load(
        frozen_artifact_path, map_location="cpu", weights_only=False
    )
    if (
        frozen_artifact.get("schema")
        != "qwen38_27b_activation_binary_mlp_factor_artifact_v1"
        or frozen_artifact.get("model") != MODEL_ID
        or frozen_artifact.get("revision") != REVISION
        or frozen_artifact.get("rank") != method["candidate"]["rank"]
    ):
        raise RuntimeError("frozen-right factor artifact changed")

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
        or frozen_artifact.get("activation_cache_receipt_sha256")
        != hashlib.sha256(cache_receipt_content).hexdigest()
    ):
        raise RuntimeError("Qwen3.8 activation cache is invalid")
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
        or tuple(calibration.shape) != (calibration_rows, SHAPES[GATE][1])
        or tuple(validation.shape) != (validation_rows, SHAPES[GATE][1])
        or not bool(torch.isfinite(calibration).all())
        or not bool(torch.isfinite(validation).all())
    ):
        raise RuntimeError("Qwen3.8 activation cache dimensions changed")

    candidate = method["candidate"]
    rank = int(candidate["rank"])
    group = int(candidate["group_size"])
    support = int(candidate["nonzero_per_group"])
    fit_args = {
        "rank": rank,
        "iterations": int(candidate["iterations"]),
        "pca_iterations": int(candidate["randomized_pca_iterations"]),
        "seed": int(candidate["seed"]),
        "relative_ridge": float(candidate["relative_ridge"]),
    }

    frozen_gate_up = frozen_artifact["gate_up"]
    if (
        frozen_gate_up.get("tensor_names") != [GATE, UP]
        or frozen_gate_up.get("split_rows") != [SHAPES[GATE][0], SHAPES[UP][0]]
    ):
        raise RuntimeError("frozen gate/up factor scope changed")
    frozen_gate_up_right = BinaryRightFactor(
        signs=frozen_gate_up["right_signs"],
        scales=frozen_gate_up["right_scales"],
    )
    frozen_gate_up_left = exported_int4_left(
        frozen_gate_up["left_codes"], frozen_gate_up["left_scales"]
    ).split([SHAPES[GATE][0], SHAPES[UP][0]], dim=0)

    sources: dict[str, torch.Tensor] = {}
    source_records: dict[str, dict[str, Any]] = {}
    ternary = {}
    reference_cal = {}
    reference_val = {}
    direct_cal = {}
    direct_val = {}
    for name in (GATE, UP):
        sources[name], source_records[name] = load_weight(
            args.model_dir, weight_map, name
        )
        ternary[name] = exported_row_normalized_ternary(
            sources[name], group=group, nonzero_per_group=support
        )
        reference_cal[name] = functional.linear(calibration, sources[name])
        reference_val[name] = functional.linear(validation, sources[name])
        direct_cal[name] = functional.linear(calibration, ternary[name].weights)
        direct_val[name] = functional.linear(validation, ternary[name].weights)

    gate_up_targets = tuple(
        reference_cal[name] - direct_cal[name] for name in (GATE, UP)
    )
    dual_gate_up = fit_activation_dual_binary_residual(
        calibration, gate_up_targets, **fit_args
    )
    dual_gate_up_cal = activation_dual_outputs(
        calibration, dual_gate_up.left_coefficients, dual_gate_up.right
    )
    dual_gate_up_val = activation_dual_outputs(
        validation, dual_gate_up.left_coefficients, dual_gate_up.right
    )
    frozen_gate_up_cal = activation_dual_outputs(
        calibration, frozen_gate_up_left, frozen_gate_up_right
    )
    frozen_gate_up_val = activation_dual_outputs(
        validation, frozen_gate_up_left, frozen_gate_up_right
    )

    projection_records = {}
    identity_errors = []
    dual_outputs_cal = {}
    dual_outputs_val = {}
    frozen_outputs_cal = {}
    frozen_outputs_val = {}
    for index, name in enumerate((GATE, UP)):
        dual_outputs_cal[name] = direct_cal[name] + dual_gate_up_cal[index]
        dual_outputs_val[name] = direct_val[name] + dual_gate_up_val[index]
        frozen_outputs_cal[name] = direct_cal[name] + frozen_gate_up_cal[index]
        frozen_outputs_val[name] = direct_val[name] + frozen_gate_up_val[index]
        direct_metrics = error_metrics(reference_val[name], direct_val[name])
        frozen_metrics = error_metrics(
            reference_val[name], frozen_outputs_val[name]
        )
        dual_metrics = error_metrics(reference_val[name], dual_outputs_val[name])
        frozen_receipt_metrics = frozen_receipt["projections"][name]["validation"][
            "activation_calibrated_binary"
        ]
        frozen_direct_metrics = frozen_receipt["projections"][name]["validation"][
            "direct_ternary"
        ]
        if not metric_close(direct_metrics, frozen_direct_metrics):
            raise RuntimeError(f"direct-ternary replay changed for {name}")
        if not metric_close(frozen_metrics, frozen_receipt_metrics):
            raise RuntimeError(f"frozen-right replay changed for {name}")
        projection_records[name] = {
            "source": source_records[name],
            "scale_export": ternary[name].scale_record,
            "validation": {
                "direct_ternary": direct_metrics,
                "frozen_weight_right": frozen_metrics,
                "activation_dual_right": dual_metrics,
                "dual_error_reduction_vs_direct": reduction(
                    dual_metrics["squared_error"], direct_metrics["squared_error"]
                ),
                "dual_error_reduction_vs_frozen_right": reduction(
                    dual_metrics["squared_error"], frozen_metrics["squared_error"]
                ),
                "dual_relative_regression_vs_frozen_right": (
                    dual_metrics["squared_error"]
                    / max(frozen_metrics["squared_error"], 1e-30)
                    - 1.0
                ),
            },
        }
        identity_errors.append(
            factor_identity_error(
                validation,
                dual_gate_up.left_coefficients[index],
                dual_gate_up.right,
            )
        )

    def swiglu(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional.silu(values[GATE]) * values[UP]

    reference_intermediate_cal = swiglu(reference_cal)
    reference_intermediate_val = swiglu(reference_val)
    direct_intermediate_cal = swiglu(direct_cal)
    direct_intermediate_val = swiglu(direct_val)
    frozen_intermediate_cal = swiglu(frozen_outputs_cal)
    frozen_intermediate_val = swiglu(frozen_outputs_val)
    dual_intermediate_cal = swiglu(dual_outputs_cal)
    dual_intermediate_val = swiglu(dual_outputs_val)

    sources[DOWN], source_records[DOWN] = load_weight(
        args.model_dir, weight_map, DOWN
    )
    ternary[DOWN] = exported_row_normalized_ternary(
        sources[DOWN], group=group, nonzero_per_group=support
    )
    reference_output_cal = functional.linear(
        reference_intermediate_cal, sources[DOWN]
    )
    reference_output_val = functional.linear(
        reference_intermediate_val, sources[DOWN]
    )
    direct_output_val = functional.linear(
        direct_intermediate_val, ternary[DOWN].weights
    )

    frozen_down = frozen_artifact["down"]
    if frozen_down.get("tensor_name") != DOWN:
        raise RuntimeError("frozen down factor scope changed")
    frozen_down_right = BinaryRightFactor(
        signs=frozen_down["right_signs"], scales=frozen_down["right_scales"]
    )
    frozen_down_left = exported_int4_left(
        frozen_down["left_codes"], frozen_down["left_scales"]
    )
    frozen_down_base_cal = functional.linear(
        frozen_intermediate_cal, ternary[DOWN].weights
    )
    frozen_down_base_val = functional.linear(
        frozen_intermediate_val, ternary[DOWN].weights
    )
    frozen_output_cal = frozen_down_base_cal + activation_dual_outputs(
        frozen_intermediate_cal, (frozen_down_left,), frozen_down_right
    )[0]
    frozen_output_val = frozen_down_base_val + activation_dual_outputs(
        frozen_intermediate_val, (frozen_down_left,), frozen_down_right
    )[0]

    dual_down_base_cal = functional.linear(
        dual_intermediate_cal, ternary[DOWN].weights
    )
    dual_down_base_val = functional.linear(
        dual_intermediate_val, ternary[DOWN].weights
    )
    dual_down = fit_activation_dual_binary_residual(
        dual_intermediate_cal,
        (reference_output_cal - dual_down_base_cal,),
        **fit_args,
    )
    dual_output_cal = dual_down_base_cal + activation_dual_outputs(
        dual_intermediate_cal, dual_down.left_coefficients, dual_down.right
    )[0]
    dual_output_val = dual_down_base_val + activation_dual_outputs(
        dual_intermediate_val, dual_down.left_coefficients, dual_down.right
    )[0]
    identity_errors.append(
        factor_identity_error(
            dual_intermediate_val,
            dual_down.left_coefficients[0],
            dual_down.right,
        )
    )

    direct_metrics = error_metrics(reference_output_val, direct_output_val)
    frozen_metrics = error_metrics(reference_output_val, frozen_output_val)
    dual_metrics = error_metrics(reference_output_val, dual_output_val)
    dual_calibration_metrics = error_metrics(reference_output_cal, dual_output_cal)
    frozen_receipt_metrics = frozen_receipt["complete_mlp"]["validation"][
        "activation_calibrated_binary"
    ]
    frozen_direct_metrics = frozen_receipt["complete_mlp"]["validation"][
        "direct_complete_ternary"
    ]
    if not metric_close(direct_metrics, frozen_direct_metrics):
        raise RuntimeError("complete direct-ternary replay changed")
    if not metric_close(frozen_metrics, frozen_receipt_metrics):
        raise RuntimeError("complete frozen-right replay changed")

    storage = complete_mlp_storage_projection(
        hidden=SHAPES[GATE][1],
        intermediate=SHAPES[GATE][0],
        rank=rank,
        group=group,
        nonzero_per_group=support,
    )
    gate_spec = method["gate"]
    full_vs_frozen = reduction(
        dual_metrics["squared_error"], frozen_metrics["squared_error"]
    )
    full_vs_direct = reduction(
        dual_metrics["squared_error"], direct_metrics["squared_error"]
    )
    maximum_projection_regression = max(
        projection_records[name]["validation"][
            "dual_relative_regression_vs_frozen_right"
        ]
        for name in (GATE, UP)
    )
    maximum_identity_error = max(identity_errors)
    gate = {
        **gate_spec,
        "observed_full_tensors": 3,
        "observed_full_mlp_validation_error_reduction_vs_frozen_right": full_vs_frozen,
        "observed_full_mlp_validation_error_reduction_vs_direct": full_vs_direct,
        "observed_maximum_per_projection_validation_regression_vs_frozen_right": maximum_projection_regression,
        "observed_full_mlp_validation_relative_rmse": dual_metrics["relative_rmse"],
        "observed_complete_mlp_bits_per_weight": storage["bits_per_weight"],
        "observed_kernel_identity_max_abs_error": maximum_identity_error,
    }
    gate["passed"] = bool(
        gate["observed_full_tensors"] == int(gate_spec["required_full_tensors"])
        and full_vs_frozen
        >= float(
            gate_spec[
                "minimum_full_mlp_validation_error_reduction_vs_frozen_right"
            ]
        )
        and full_vs_direct
        >= float(gate_spec["minimum_full_mlp_validation_error_reduction_vs_direct"])
        and maximum_projection_regression
        <= float(
            gate_spec[
                "maximum_per_projection_validation_regression_vs_frozen_right"
            ]
        )
        and dual_metrics["relative_rmse"]
        <= float(gate_spec["maximum_full_mlp_validation_relative_rmse"])
        and storage["bits_per_weight"]
        <= float(gate_spec["maximum_complete_mlp_bits_per_weight"])
        and maximum_identity_error
        <= float(gate_spec["maximum_kernel_identity_error"])
    )

    artifact = {
        "schema": "qwen38_27b_activation_dual_binary_mlp_factor_artifact_v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "weight_parent_receipt_sha256": hashlib.sha256(
            weight_parent_content
        ).hexdigest(),
        "frozen_right_receipt_sha256": hashlib.sha256(frozen_content).hexdigest(),
        "activation_cache_receipt_sha256": hashlib.sha256(
            cache_receipt_content
        ).hexdigest(),
        "rank": rank,
        "group_size": group,
        "nonzero_per_group": support,
        "iterations": int(candidate["iterations"]),
        "randomized_pca_iterations": int(candidate["randomized_pca_iterations"]),
        "relative_ridge": float(candidate["relative_ridge"]),
        "gate_up": {
            "tensor_names": [GATE, UP],
            "right_signs": dual_gate_up.right.signs.cpu(),
            "right_scales": dual_gate_up.right.scales.cpu(),
            "left_codes": torch.cat(dual_gate_up.left_codes, dim=0).cpu(),
            "left_scales": torch.cat(dual_gate_up.left_scales, dim=0).cpu(),
            "split_rows": [SHAPES[GATE][0], SHAPES[UP][0]],
        },
        "down": {
            "tensor_name": DOWN,
            "right_signs": dual_down.right.signs.cpu(),
            "right_scales": dual_down.right.scales.cpu(),
            "left_codes": dual_down.left_codes[0].cpu(),
            "left_scales": dual_down.left_scales[0].cpu(),
        },
    }
    atomic_torch_save(artifact, args.artifact)
    receipt = {
        "schema": "qwen38_27b_activation_dual_binary_mlp_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "model": MODEL_ID,
        "revision": REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "method_preregistration_sha256": sha256_file(method_path),
        "execution_amendment_sha256": sha256_file(execution_path),
        "parents": {
            "weight_binary_receipt": str(args.weight_parent_receipt),
            "weight_binary_receipt_sha256": hashlib.sha256(
                weight_parent_content
            ).hexdigest(),
            "frozen_right_receipt": str(args.frozen_right_receipt),
            "frozen_right_receipt_sha256": hashlib.sha256(frozen_content).hexdigest(),
            "frozen_right_gate_passed": frozen_receipt.get("gate", {}).get("passed")
            is True,
            "activation_cache_receipt": str(args.activation_cache_receipt),
            "activation_cache_receipt_sha256": hashlib.sha256(
                cache_receipt_content
            ).hexdigest(),
        },
        "activation_rows": {
            "calibration": int(calibration.shape[0]),
            "validation": int(validation.shape[0]),
        },
        "gate_up": {
            "objective_history": list(dual_gate_up.objective_history),
            "best_iteration": dual_gate_up.best_iteration,
            "projections": projection_records,
        },
        "down": {
            "source": source_records[DOWN],
            "scale_export": ternary[DOWN].scale_record,
            "objective_history": list(dual_down.objective_history),
            "best_iteration": dual_down.best_iteration,
        },
        "complete_mlp": {
            "calibration": dual_calibration_metrics,
            "validation": {
                "direct_complete_ternary": direct_metrics,
                "frozen_weight_right": frozen_metrics,
                "activation_dual_right": dual_metrics,
                "dual_error_reduction_vs_frozen_right": full_vs_frozen,
                "dual_error_reduction_vs_direct": full_vs_direct,
            },
        },
        "storage": storage,
        "factor_artifact": {
            "path": str(args.artifact),
            "bytes": args.artifact.stat().st_size,
            "sha256": sha256_file(args.artifact),
            "contains_ternary_backbone": False,
        },
        "gate": gate,
        "claim_boundary": {
            "full_real_layer0_gate_up_down_weights": True,
            "real_disjoint_layer0_activations": True,
            "identical_rate_comparison_to_frozen_right": True,
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
