#!/usr/bin/env python3
# ruff: noqa: E402
"""Expand the frozen rank-16 sensitivity residual across all Qwen3.8 layers."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUNTIME))
import collect_qwen38_27b_affine_butterfly_block_cache_v1 as cache_v1
import collect_qwen38_27b_full_context_block_cache_v42 as cache_v42
import run_qwen38_27b_affine_butterfly_block_gate_v1 as parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_all_layer_mixed_direct_v43 as direct_v43
import run_qwen38_27b_all_layer_mixed_train_v44 as train_v44
import run_qwen38_27b_mixed_direct_screen_v1 as pilot_direct

from clark_quantize.qwen38_affine_artifact import sha256_file
from clark_quantize.qwen38_calibration_span_residual import (
    Int8ResidualAdapter,
    fit_float_residual_adapter,
    int8_adapter_bytes,
    load_int8_adapter,
    normalized_mse,
    quantize_residual_adapter,
    write_int8_adapter,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
FULL_CONTEXT_RECEIPT_SHA256 = train_v44.FULL_CONTEXT_RECEIPT_SHA256
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
ADAPTER_BYTES = 174_144
SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_receipt_v45"
LAYER_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_layer_v45"
PAYLOAD_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_payload_v45"


def trim_allocator() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
    if len(Path("/proc/swaps").read_text().splitlines()) != 1:
        raise SystemExit("infrastructure-invalid: swap is active")
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    if fields.get("MemAvailable", 0) < 18 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 18 GiB memory available")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: v45 disk floor failed")


def selected_rows(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1]).index_select(0, rows).float()


def classify_sensitivity(base_validation_nmse: float) -> str:
    if not math.isfinite(base_validation_nmse) or base_validation_nmse < 0:
        raise ValueError("invalid base internal-validation NMSE")
    return (
        "rank16_int8_residual"
        if base_validation_nmse >= SENSITIVITY_THRESHOLD
        else "exact_bypass"
    )


def payload_descriptor(
    *,
    path: Path,
    layer_index: int,
    preregistration_sha256: str,
    training_receipt_sha256: str,
    full_context_shard_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PAYLOAD_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "layer": layer_index,
        "rank": RANK,
        "ridge_fraction": RIDGE_FRACTION,
        "input_features": 5120,
        "output_features": 5120,
        "format": (
            "left_INT8_per_column_scale_FP16_then_right_INT8_per_row_"
            "scale_FP16_then_bias_FP16"
        ),
        "payload": {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "preregistration_sha256": preregistration_sha256,
        "training_receipt_sha256": training_receipt_sha256,
        "full_context_shard_sha256": full_context_shard_sha256,
        "complete": True,
        "terminal": True,
    }


def validate_payload_descriptor(
    descriptor: dict[str, Any],
    *,
    payload_path: Path,
    layer_index: int,
    preregistration_sha256: str,
    training_receipt_sha256: str,
    full_context_shard_sha256: str,
) -> Int8ResidualAdapter:
    if (
        descriptor.get("schema") != PAYLOAD_SCHEMA
        or descriptor.get("model") != MODEL_ID
        or descriptor.get("revision") != MODEL_REVISION
        or descriptor.get("layer") != layer_index
        or descriptor.get("rank") != RANK
        or descriptor.get("ridge_fraction") != RIDGE_FRACTION
        or descriptor.get("input_features") != 5120
        or descriptor.get("output_features") != 5120
        or descriptor.get("preregistration_sha256")
        != preregistration_sha256
        or descriptor.get("training_receipt_sha256")
        != training_receipt_sha256
        or descriptor.get("full_context_shard_sha256")
        != full_context_shard_sha256
        or descriptor.get("payload", {}).get("bytes") != payload_path.stat().st_size
        or descriptor.get("payload", {}).get("sha256") != sha256_file(payload_path)
        or descriptor.get("complete") is not True
        or descriptor.get("terminal") is not True
    ):
        raise ValueError("v45 residual payload identity changed")
    adapter = load_int8_adapter(
        payload_path,
        input_features=5120,
        output_features=5120,
        rank=RANK,
        ridge_fraction=RIDGE_FRACTION,
    )
    if adapter.persistent_bytes != ADAPTER_BYTES:
        raise ValueError("v45 residual payload byte count changed")
    return adapter


def write_or_validate_adapter(
    adapter: Int8ResidualAdapter,
    *,
    payload_path: Path,
    descriptor_path: Path,
    layer_index: int,
    preregistration_sha256: str,
    training_receipt_sha256: str,
    full_context_shard_sha256: str,
) -> dict[str, Any]:
    expected = int8_adapter_bytes(adapter)
    if len(expected) != ADAPTER_BYTES:
        raise ValueError("v45 fitted residual byte count changed")
    if payload_path.exists() and payload_path.read_bytes() != expected:
        raise ValueError("existing v45 residual payload changed")
    if not payload_path.exists():
        write_int8_adapter(payload_path, adapter)
    expected_descriptor = payload_descriptor(
        path=payload_path,
        layer_index=layer_index,
        preregistration_sha256=preregistration_sha256,
        training_receipt_sha256=training_receipt_sha256,
        full_context_shard_sha256=full_context_shard_sha256,
    )
    if descriptor_path.exists():
        descriptor = json.loads(descriptor_path.read_text())
        if descriptor != expected_descriptor:
            raise ValueError("existing v45 residual descriptor changed")
    else:
        cache_v1.atomic_json(expected_descriptor, descriptor_path)
        descriptor = expected_descriptor
    validate_payload_descriptor(
        descriptor,
        payload_path=payload_path,
        layer_index=layer_index,
        preregistration_sha256=preregistration_sha256,
        training_receipt_sha256=training_receipt_sha256,
        full_context_shard_sha256=full_context_shard_sha256,
    )
    return descriptor


def validate_terminal_layer(
    result: dict[str, Any],
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    training_receipt_sha256: str,
) -> None:
    layer_index = result.get("layer")
    if (
        result.get("schema") != LAYER_SCHEMA
        or result.get("model") != MODEL_ID
        or result.get("revision") != MODEL_REVISION
        or result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("training_receipt_sha256") != training_receipt_sha256
        or not isinstance(layer_index, int)
        or result.get("complete") is not True
        or result.get("terminal") is not True
    ):
        raise ValueError("v45 terminal layer identity changed")
    mode = result.get("mode")
    frozen_classification = classify_sensitivity(
        float(result.get("base_validation_nmse", float("nan")))
    )
    if frozen_classification != mode:
        raise ValueError("v45 terminal sensitivity classification changed")
    if mode == "rank16_int8_residual":
        if result.get("persistent_payload_bytes") != ADAPTER_BYTES:
            raise ValueError("v45 terminal residual byte count changed")
        descriptor_path = artifact_root / result["descriptor"]
        payload_path = artifact_root / result["payload"]
        descriptor = json.loads(descriptor_path.read_text())
        validate_payload_descriptor(
            descriptor,
            payload_path=payload_path,
            layer_index=layer_index,
            preregistration_sha256=preregistration_sha256,
            training_receipt_sha256=training_receipt_sha256,
            full_context_shard_sha256=result["full_context_shard_sha256"],
        )
    elif mode == "exact_bypass":
        if (
            result.get("descriptor") is not None
            or result.get("payload") is not None
            or result.get("persistent_payload_bytes") != 0
            or result.get("bypass_tensor_exact") is not True
        ):
            raise ValueError("v45 exact bypass unexpectedly has a payload")
    else:
        raise ValueError("v45 terminal layer mode changed")


def gate_from_layers(
    layers: list[dict[str, Any]],
    *,
    minimum_mean_active_validation_reduction: float,
    maximum_final_validation_nmse: float,
    maximum_residual_bytes: int,
) -> dict[str, Any]:
    if len(layers) != 64 or [item.get("layer") for item in layers] != list(range(64)):
        raise ValueError("v45 layer denominator changed")
    active = [item for item in layers if item["mode"] == "rank16_int8_residual"]
    bypass = [item for item in layers if item["mode"] == "exact_bypass"]
    mean_active = (
        sum(item["relative_validation_nmse_reduction"] for item in active)
        / len(active)
        if active
        else 0.0
    )
    total_bytes = sum(item["persistent_payload_bytes"] for item in layers)
    gate = {
        "required_layers": 64,
        "observed_layers": len(layers),
        "active_residual_layers": len(active),
        "exact_bypass_layers": len(bypass),
        "all_layers_terminal": all(item["terminal"] for item in layers),
        "all_values_finite": all(
            math.isfinite(item[key])
            for item in layers
            for key in (
                "base_calibration_nmse",
                "base_validation_nmse",
                "final_calibration_nmse",
                "final_validation_nmse",
            )
        ),
        "all_active_layers_calibration_not_regressed": all(
            item["final_calibration_nmse"] <= item["base_calibration_nmse"]
            for item in active
        ),
        "all_active_layers_validation_not_regressed": all(
            item["final_validation_nmse"] <= item["base_validation_nmse"]
            for item in active
        ),
        "all_bypass_layers_exact": all(
            item["bypass_tensor_exact"] is True for item in bypass
        ),
        "mean_active_relative_validation_nmse_reduction": mean_active,
        "minimum_mean_active_relative_validation_nmse_reduction": (
            minimum_mean_active_validation_reduction
        ),
        "maximum_final_validation_nmse_per_layer": maximum_final_validation_nmse,
        "all_final_validation_nmse_within_maximum": all(
            item["final_validation_nmse"] <= maximum_final_validation_nmse
            for item in layers
        ),
        "all_pilot_modes_exact": all(
            layers[index]["pilot_mode_exact"] is True for index in PILOT_LAYERS
        ),
        "all_active_pilot_payloads_exact": all(
            layers[index]["pilot_payload_exact"] is True
            for index in (0, 63)
        ),
        "pilot_bypass_layer_has_no_payload": (
            layers[32]["mode"] == "exact_bypass"
            and layers[32]["payload"] is None
        ),
        "total_residual_payload_bytes": total_bytes,
        "maximum_residual_payload_bytes": maximum_residual_bytes,
        "residual_payload_within_maximum": total_bytes <= maximum_residual_bytes,
    }
    gate["passed"] = bool(
        gate["observed_layers"] == gate["required_layers"]
        and gate["active_residual_layers"] > 0
        and gate["all_layers_terminal"]
        and gate["all_values_finite"]
        and gate["all_active_layers_calibration_not_regressed"]
        and gate["all_active_layers_validation_not_regressed"]
        and gate["all_bypass_layers_exact"]
        and gate["mean_active_relative_validation_nmse_reduction"]
        >= gate["minimum_mean_active_relative_validation_nmse_reduction"]
        and gate["all_final_validation_nmse_within_maximum"]
        and gate["all_pilot_modes_exact"]
        and gate["all_active_pilot_payloads_exact"]
        and gate["pilot_bypass_layer_has_no_payload"]
        and gate["residual_payload_within_maximum"]
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_sensitivity_residual_v45.json",
    )
    parser.add_argument(
        "--full-context-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-context-block-cache-v42"),
    )
    parser.add_argument(
        "--full-context-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-full-context-block-cache-v42/receipt.json"
        ),
    )
    parser.add_argument(
        "--direct-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-direct-v43"),
    )
    parser.add_argument(
        "--direct-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/receipt.json"
        ),
    )
    parser.add_argument(
        "--trained-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json"
        ),
    )
    parser.add_argument(
        "--pilot-residual-preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_sensitivity_gated_residual_v37.json",
    )
    parser.add_argument(
        "--pilot-residual-calibration",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_sensitivity_gated_residual_v37/calibration.json",
    )
    parser.add_argument(
        "--pilot-residual-artifact-root",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_sensitivity_gated_residual_v37/artifacts",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-sensitivity-residual-v45"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-sensitivity-residual-v45/receipt.json"
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-sensitivity-residual-v45/complete.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen v45 all-layer residual uses four CPU threads")
    torch.set_num_threads(args.threads)

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = sha256_file(args.preregistration)
    direct_receipt_sha = sha256_file(args.direct_receipt)
    training_receipt_sha = sha256_file(args.training_receipt)
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_all_layer_sensitivity_residual_preregistration_v45"
        or prereg.get("status") != "frozen_after_v44_pass_before_residual_fit"
        or prereg.get("scope", {}).get("active_model_only") != MODEL_ID
        or prereg.get("scope", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("target_layers") != list(range(64))
        or prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True
        or prereg.get("parent_training", {}).get("receipt_sha256")
        != training_receipt_sha
        or prereg.get("frozen_inputs", {}).get("runner_sha256")
        != sha256_file(Path(__file__))
        or prereg.get("sensitivity_rule", {}).get("minimum_base_validation_nmse")
        != SENSITIVITY_THRESHOLD
        or prereg.get("adapter", {}).get("rank") != RANK
        or prereg.get("adapter", {}).get("ridge_fraction") != RIDGE_FRACTION
    ):
        raise ValueError("v45 preregistration changed")
    resource_guard(
        Path("/workspace"), int(prereg["resources"]["minimum_free_disk_bytes"])
    )

    full_context_receipt = json.loads(args.full_context_receipt.read_text())
    direct_receipt = json.loads(args.direct_receipt.read_text())
    training_receipt = json.loads(args.training_receipt.read_text())
    pilot_prereg = json.loads(args.pilot_residual_preregistration.read_text())
    pilot_calibration = json.loads(args.pilot_residual_calibration.read_text())
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        sha256_file(index_path) != INDEX_SHA256
        or sha256_file(args.full_context_receipt) != FULL_CONTEXT_RECEIPT_SHA256
        or full_context_receipt.get("gate", {}).get("passed") is not True
        or direct_receipt.get("schema") != direct_v43.SCHEMA
        or direct_receipt.get("gate", {}).get("passed") is not True
        or training_receipt.get("schema") != train_v44.SCHEMA
        or training_receipt.get("direct_receipt_sha256") != direct_receipt_sha
        or training_receipt.get("gate", {}).get("passed") is not True
        or len(training_receipt.get("layers", [])) != 64
        or [record.get("layer") for record in training_receipt.get("layers", [])]
        != list(range(64))
        or not all(
            record.get("passed") is True
            for record in training_receipt.get("layers", [])
        )
        or training_receipt.get("complete") is not True
        or training_receipt.get("terminal") is not True
        or sha256_file(args.pilot_residual_preregistration)
        != PILOT_RESIDUAL_PREREGISTRATION_SHA256
        or sha256_file(args.pilot_residual_calibration)
        != PILOT_RESIDUAL_CALIBRATION_SHA256
        or pilot_prereg["sensitivity_rule"]["minimum_base_nmse_for_residual"]
        != SENSITIVITY_THRESHOLD
        or pilot_prereg["fixed_adapter"]["rank"] != RANK
        or pilot_prereg["fixed_adapter"]["ridge_fraction"] != RIDGE_FRACTION
        or pilot_calibration.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("v45 source evidence changed")
    train_v44.validate_direct_parent_artifacts(
        direct_receipt, direct_artifact_root=args.direct_artifact_root
    )
    train_v44.validate_full_context_parent_artifacts(
        full_context_receipt, full_context_root=args.full_context_root
    )
    for result in training_receipt["layers"]:
        train_v44.validate_terminal_layer(
            result,
            artifact_root=args.trained_artifact_root,
            repository_commit=training_receipt["repository_commit"],
            preregistration_sha256=training_receipt["preregistration_sha256"],
            direct_receipt_sha256=direct_receipt_sha,
        )

    if args.receipt.exists() or args.completion_marker.exists():
        if not args.receipt.is_file() or not args.completion_marker.is_file():
            raise RuntimeError("partial v45 terminal receipt")
        existing = json.loads(args.receipt.read_text())
        marker = json.loads(args.completion_marker.read_text())
        if (
            existing.get("schema") != SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != prereg_sha
            or existing.get("training_receipt_sha256") != training_receipt_sha
            or len(existing.get("layers", [])) != 64
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
            or marker.get("receipt_sha256") != sha256_file(args.receipt)
            or marker.get("passed") != existing.get("gate", {}).get("passed")
            or marker.get("layer_count") != 64
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v45 terminal receipt changed")
        for result in existing["layers"]:
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                training_receipt_sha256=training_receipt_sha,
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    context = torch.load(
        args.full_context_root / "context.pt", map_location="cpu", weights_only=False
    )
    calibration_rows = context["calibration_flat_rows"].long()
    validation_rows = context["validation_flat_rows"].long()
    selected = torch.cat((calibration_rows, validation_rows)).long()
    attention_mask = context["attention_mask"].long()
    position_ids = context["text_position_ids"].long()
    if (
        calibration_rows.numel() != 64
        or validation_rows.numel() != 32
        or torch.unique(selected).numel() != 96
        or int(calibration_rows.max()) >= 8 * 64
        or int(validation_rows.min()) < 8 * 64
        or int(validation_rows.max()) >= 12 * 64
        or attention_mask.shape != (12, 64)
        or position_ids.shape != (12, 64)
    ):
        raise ValueError("v45 calibration/validation identity changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if config.num_hidden_layers != 64 or config.hidden_size != 5120:
        raise ValueError("v45 Qwen3.8 architecture changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    full_by_layer = {
        int(record["layer"]): record for record in full_context_receipt["layers"]
    }
    trained_by_layer = {
        int(record["layer"]): record for record in training_receipt["layers"]
    }
    pilot_by_layer = {
        int(record["layer"]): record for record in pilot_calibration["layers"]
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.monotonic()

    for layer_index in range(64):
        layer_root = args.artifact_root / "layers" / f"{layer_index:02d}"
        terminal_path = layer_root / "terminal-result.json"
        if terminal_path.exists():
            result = json.loads(terminal_path.read_text())
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                training_receipt_sha256=training_receipt_sha,
            )
            results.append(result)
            print(json.dumps({"event": "v45_layer_reused", "layer": layer_index}), flush=True)
            continue
        layer_started = time.monotonic()
        print(json.dumps({"event": "v45_layer_started", "layer": layer_index}), flush=True)
        shard_record = full_by_layer[layer_index]
        shard_path = args.full_context_root / "layers" / f"{layer_index:02d}.pt"
        if (
            shard_path.stat().st_size != shard_record["shard"]["bytes"]
            or sha256_file(shard_path) != shard_record["shard"]["sha256"]
        ):
            raise ValueError("v45 full-context shard changed")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        cache_v42.validate_shard_tensors(
            shard,
            layer_index,
            preregistration_sha256=full_context_receipt["preregistration_sha256"],
            producer_commit=full_context_receipt["repository_commit"],
        )
        hidden = shard["input"].contiguous()
        teacher = shard["output"].contiguous()
        state, _ = cache_v1.layer_state(args.model_dir, weight_map, layer_index)
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        layer_type = config.layer_types[layer_index]
        exact_layer = parent.instantiate_layer(config, layer_index, state)
        with torch.no_grad():
            exact_output = qat.forward_with_grad(
                exact_layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )
        if not torch.equal(exact_output, teacher):
            raise ValueError("v45 exact BF16 shard replay changed")
        del exact_layer, exact_output
        gc.collect()

        block = train_v44.build_frozen_trained_block(
            config=config,
            layer_index=layer_index,
            state=state,
            selected_rows=calibration_rows,
            direct_aff3_root=args.direct_artifact_root / "aff3-base",
            direct_repository_commit=direct_receipt["repository_commit"],
            direct_preregistration_sha256=direct_receipt[
                "preregistration_sha256"
            ],
            trained_artifact_root=args.trained_artifact_root,
            trained_result=trained_by_layer[layer_index],
            trained_repository_commit=training_receipt["repository_commit"],
            trained_preregistration_sha256=training_receipt[
                "preregistration_sha256"
            ],
            direct_receipt_sha256=direct_receipt_sha,
        )
        for target in pilot_direct.mlp_targets(layer_index):
            state.pop(target["state_name"])
        for name in parent.site_layout(layer_type)["attention_output"]:
            state.pop(name)
        with torch.no_grad():
            calibration_output = qat.forward_with_grad(
                block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )
        candidate_mlp = block.layer.get_submodule("mlp")
        candidate_mlp.selected_rows = validation_rows.long().contiguous()
        with torch.no_grad():
            validation_output = qat.forward_with_grad(
                block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )
        calibration_inputs = selected_rows(hidden, calibration_rows)
        calibration_candidate = selected_rows(calibration_output, calibration_rows)
        calibration_teacher = selected_rows(teacher, calibration_rows)
        validation_inputs = selected_rows(hidden, validation_rows)
        validation_candidate = selected_rows(validation_output, validation_rows)
        validation_teacher = selected_rows(teacher, validation_rows)
        base_calibration_nmse = normalized_mse(
            calibration_candidate, calibration_teacher
        )
        base_validation_nmse = normalized_mse(validation_candidate, validation_teacher)
        mode = classify_sensitivity(base_validation_nmse)
        descriptor_relative: str | None = None
        payload_relative: str | None = None
        persistent_bytes = 0
        bypass_exact: bool | None = None
        pilot_payload_exact: bool | None = None
        adapter: Int8ResidualAdapter | None = None
        if mode == "rank16_int8_residual":
            adapter = quantize_residual_adapter(
                fit_float_residual_adapter(
                    calibration_inputs,
                    calibration_candidate,
                    calibration_teacher,
                    rank=RANK,
                    ridge_fraction=RIDGE_FRACTION,
                )
            )
            payload_path = layer_root / f"layer-{layer_index}.csr"
            descriptor_path = layer_root / f"layer-{layer_index}.json"
            write_or_validate_adapter(
                adapter,
                payload_path=payload_path,
                descriptor_path=descriptor_path,
                layer_index=layer_index,
                preregistration_sha256=prereg_sha,
                training_receipt_sha256=training_receipt_sha,
                full_context_shard_sha256=shard_record["shard"]["sha256"],
            )
            descriptor_relative = str(descriptor_path.relative_to(args.artifact_root))
            payload_relative = str(payload_path.relative_to(args.artifact_root))
            persistent_bytes = adapter.persistent_bytes
            corrected_calibration = calibration_candidate + adapter.correction(
                calibration_inputs
            )
            corrected_validation = validation_candidate + adapter.correction(
                validation_inputs
            )
            if layer_index in (0, 63):
                pilot_payload_path = (
                    args.pilot_residual_artifact_root / f"layer-{layer_index}.csr"
                )
                expected_pilot_sha = pilot_by_layer[layer_index]["payload"]["sha256"]
                if sha256_file(pilot_payload_path) != expected_pilot_sha:
                    raise ValueError("v45 pilot residual source payload changed")
                pilot_payload_exact = sha256_file(payload_path) == expected_pilot_sha
                if not pilot_payload_exact:
                    raise ValueError("v45 pilot residual payload reproduction changed")
        else:
            corrected_calibration = calibration_candidate
            corrected_validation = validation_candidate
            bypass_exact = bool(
                torch.equal(corrected_calibration, calibration_candidate)
                and torch.equal(corrected_validation, validation_candidate)
            )
        final_calibration_nmse = normalized_mse(
            corrected_calibration, calibration_teacher
        )
        final_validation_nmse = normalized_mse(
            corrected_validation, validation_teacher
        )
        pilot_mode_exact: bool | None = None
        if layer_index in PILOT_LAYERS:
            pilot_mode_exact = mode == pilot_by_layer[layer_index]["mode"]
            if not pilot_mode_exact:
                raise ValueError("v45 pilot sensitivity mode reproduction changed")
            if abs(
                base_calibration_nmse
                - float(pilot_by_layer[layer_index]["base_calibration_nmse"])
            ) > 1e-8:
                raise ValueError("v45 pilot base calibration score changed")
            if abs(
                final_calibration_nmse
                - float(pilot_by_layer[layer_index]["corrected_calibration_nmse"])
            ) > 1e-8:
                raise ValueError("v45 pilot corrected calibration score changed")
        result = {
            "schema": LAYER_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "repository_commit": repository_commit,
            "preregistration_sha256": prereg_sha,
            "training_receipt_sha256": training_receipt_sha,
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "full_context_shard_sha256": shard_record["shard"]["sha256"],
            "layer": layer_index,
            "type": layer_type,
            "calibration_rows": calibration_rows.numel(),
            "internal_validation_rows": validation_rows.numel(),
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "mode": mode,
            "base_calibration_nmse": base_calibration_nmse,
            "base_validation_nmse": base_validation_nmse,
            "final_calibration_nmse": final_calibration_nmse,
            "final_validation_nmse": final_validation_nmse,
            "relative_calibration_nmse_reduction": (
                base_calibration_nmse - final_calibration_nmse
            ) / max(base_calibration_nmse, 1e-30),
            "relative_validation_nmse_reduction": (
                base_validation_nmse - final_validation_nmse
            ) / max(base_validation_nmse, 1e-30),
            "descriptor": descriptor_relative,
            "payload": payload_relative,
            "persistent_payload_bytes": persistent_bytes,
            "bypass_tensor_exact": bypass_exact,
            "pilot_mode_exact": pilot_mode_exact,
            "pilot_payload_exact": pilot_payload_exact,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        cache_v1.atomic_json(result, terminal_path)
        validate_terminal_layer(
            result,
            artifact_root=args.artifact_root,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            training_receipt_sha256=training_receipt_sha,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "v45_layer_terminal",
                    "layer": layer_index,
                    "mode": mode,
                    "base_validation_nmse": base_validation_nmse,
                    "final_validation_nmse": final_validation_nmse,
                    "passed": final_validation_nmse <= base_validation_nmse,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            shard,
            hidden,
            teacher,
            state,
            block,
            calibration_output,
            validation_output,
            candidate_mlp,
            calibration_inputs,
            calibration_candidate,
            calibration_teacher,
            validation_inputs,
            validation_candidate,
            validation_teacher,
            corrected_calibration,
            corrected_validation,
            position_embeddings,
            masks,
            adapter,
        )
        gc.collect()
        trim_allocator()

    gate = gate_from_layers(
        results,
        minimum_mean_active_validation_reduction=float(
            prereg["gate"][
                "minimum_mean_active_relative_validation_nmse_reduction"
            ]
        ),
        maximum_final_validation_nmse=float(
            prereg["gate"]["maximum_final_validation_nmse_per_layer"]
        ),
        maximum_residual_bytes=int(
            prereg["gate"]["maximum_residual_payload_bytes"]
        ),
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_receipt_sha,
        "direct_receipt_sha256": direct_receipt_sha,
        "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
        "pilot_residual_preregistration_sha256": (
            PILOT_RESIDUAL_PREREGISTRATION_SHA256
        ),
        "pilot_residual_calibration_sha256": PILOT_RESIDUAL_CALIBRATION_SHA256,
        "layers": results,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "claim_boundary": prereg["claim_boundary"],
    }
    cache_v1.atomic_json(receipt, args.receipt)
    cache_v1.atomic_json(
        {
            "terminal": True,
            "passed": gate["passed"],
            "receipt_sha256": sha256_file(args.receipt),
            "layer_count": len(results),
            "active_residual_layers": gate["active_residual_layers"],
            "total_residual_payload_bytes": gate[
                "total_residual_payload_bytes"
            ],
        },
        args.completion_marker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
