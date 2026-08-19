#!/usr/bin/env python3
"""Fit a rank-16 INT8 subspace correction after the frozen v75 cell path."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.nn import functional

REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUNTIME))

import collect_qwen38_27b_affine_butterfly_block_cache_v1 as cache_v1
import collect_qwen38_27b_full_context_block_cache_v42 as cache_v42
import run_qwen38_27b_affine_butterfly_block_gate_v1 as block_parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_all_layer_mixed_direct_v43 as direct_v43
import run_qwen38_27b_cell_scale_covariate_shift_diagnostic_v77 as v77
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_optimal_cell_scale_fresh_replication_v76 as v76
import run_qwen38_27b_prompt_robust_cell_support_v78 as v78
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

from clark_quantize.qwen38_calibration_span_residual import (
    expected_int8_adapter_bytes,
    fit_float_residual_adapter,
    load_int8_adapter,
    quantize_residual_adapter,
    write_int8_adapter,
)
from clark_quantize.qwen38_mixed_cq3_runtime import tensor_sha256

MODEL_ID = v71.MODEL_ID
MODEL_REVISION = v71.MODEL_REVISION
LAYER = v71.LAYER
HIDDEN = v71.HIDDEN
INTERMEDIATE = v71.INTERMEDIATE
CQ3_PAYLOAD_BYTES = v71.PAYLOAD_BYTES
OUTPUT_TILES = v76.OUTPUT_TILES
TILE_ROWS = v76.TILE_ROWS
INPUT_GROUPS = v76.INPUT_GROUPS
GROUP_SIZE = v76.GROUP_SIZE
RANK = 16
RIDGE_FRACTION = 0.01
ADAPTER_BYTES = expected_int8_adapter_bytes(INTERMEDIATE, HIDDEN, RANK)
SCHEMA = "qwen38_27b_hierarchical_cell_subspace_residual_receipt_v79"


def relative_improvement(
    baseline: torch.Tensor, candidate: torch.Tensor, exact: torch.Tensor
) -> tuple[float, float, float]:
    base_nmse = v71.tensor_nmse(baseline, exact)
    candidate_nmse = v71.tensor_nmse(candidate, exact)
    return (
        base_nmse,
        candidate_nmse,
        (base_nmse - candidate_nmse) / max(base_nmse, 1e-30),
    )


def prompt_metrics(
    exact: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    owners: torch.Tensor,
) -> list[dict[str, Any]]:
    records = []
    for environment in owners.unique(sorted=True):
        rows = owners == environment
        base, corrected, improvement = relative_improvement(
            baseline[rows], candidate[rows], exact[rows]
        )
        residual = (exact[rows] - baseline[rows]).reshape(
            -1, OUTPUT_TILES, TILE_ROWS
        )
        correction = (candidate[rows] - baseline[rows]).reshape(
            -1, OUTPUT_TILES, TILE_ROWS
        )
        records.append(
            {
                "environment": int(environment),
                "rows": int(rows.sum()),
                "baseline_complete_mlp_nmse": base,
                "candidate_complete_mlp_nmse": corrected,
                "relative_improvement": improvement,
                "directional_capacity": v77.vector_geometry(
                    residual, correction
                )["nonnegative_scalar_directional_capacity"],
            }
        )
    return records


def load_v75_correction(
    artifact: Path,
    down: Any,
    down_input: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, bool]]:
    payload = artifact.read_bytes()
    mask, packed_cq3_codes, packed_cq3_scales, int4_codes, int4_scales = (
        v76.decode_frozen_hybrid(payload)
    )
    baseline_codes = down.codes.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE
    ).contiguous()
    baseline_scales = down.scales.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS
    ).contiguous()
    base_cells = baseline_codes.permute(0, 2, 1, 3)[~mask].contiguous()
    base_scales = baseline_scales.permute(0, 2, 1)[~mask].contiguous()
    checks = {
        "packed_unselected_cq3_codes_exact": torch.equal(
            packed_cq3_codes, base_cells.reshape(-1, 1, GROUP_SIZE)
        ),
        "packed_unselected_cq3_scales_exact": torch.equal(
            packed_cq3_scales, base_scales.reshape(-1, 1)
        ),
    }
    correction = v76.frozen_hybrid_correction(
        down_input.float().reshape(-1, INPUT_GROUPS, GROUP_SIZE),
        baseline_codes,
        baseline_scales,
        mask,
        int4_codes,
        int4_scales,
    ).reshape(down_input.shape[0], HIDDEN)
    return correction, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--full-context-receipt", required=True, type=Path)
    parser.add_argument(
        "--full-context-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-context-block-cache-v42"),
    )
    parser.add_argument("--v44-receipt", required=True, type=Path)
    parser.add_argument("--v75-receipt", required=True, type=Path)
    parser.add_argument("--v75-artifact", required=True, type=Path)
    parser.add_argument("--v76-preregistration", required=True, type=Path)
    parser.add_argument("--v76-receipt", required=True, type=Path)
    parser.add_argument("--v76-checkpoint", required=True, type=Path)
    parser.add_argument("--v77-receipt", required=True, type=Path)
    parser.add_argument("--v78-receipt", required=True, type=Path)
    parser.add_argument("--v37-terminal", required=True, type=Path)
    parser.add_argument("--v39-receipt", required=True, type=Path)
    parser.add_argument("--v40-receipt", required=True, type=Path)
    parser.add_argument(
        "--v44-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads != 4:
        raise ValueError("v79 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    cache_v1.resource_guard(Path("/workspace"))
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v79 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v79 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg = json.loads(args.preregistration.read_text())
    auth = prereg["authenticated_inputs"]
    parents = {
        "full_context_receipt_sha256": args.full_context_receipt,
        "v44_training_receipt_sha256": args.v44_receipt,
        "v75_receipt_sha256": args.v75_receipt,
        "v76_receipt_sha256": args.v76_receipt,
        "v77_receipt_sha256": args.v77_receipt,
        "v78_receipt_sha256": args.v78_receipt,
    }
    parent_json = {key: json.loads(path.read_text()) for key, path in parents.items()}
    kernel_paths = {
        "v37_terminal_decision_sha256": args.v37_terminal,
        "v39_stability_receipt_sha256": args.v39_receipt,
        "v40_ggml_receipt_sha256": args.v40_receipt,
    }
    kernel_json = {
        key: json.loads(path.read_text()) for key, path in kernel_paths.items()
    }
    if (
        prereg.get("schema")
        != "qwen38_27b_hierarchical_cell_subspace_residual_preregistration_v79"
        or prereg.get("status") != "frozen_development_screen_before_execution"
        or prereg.get("adapter", {}).get("rank_or_ridge_sweep") is not False
        or any(v71.file_sha256(path) != auth[key] for key, path in parents.items())
        or v71.file_sha256(args.v76_checkpoint) != auth["v76_checkpoint_sha256"]
        or v71.file_sha256(args.v75_artifact) != auth["v75_hybrid_payload_sha256"]
        or any(
            v71.file_sha256(path) != prereg["prior_exact_kernel_evidence"][key]
            for key, path in kernel_paths.items()
        )
        or parent_json["full_context_receipt_sha256"].get("gate", {}).get("passed")
        is not True
        or parent_json["v44_training_receipt_sha256"].get("gate", {}).get("passed")
        is not True
        or parent_json["v75_receipt_sha256"].get("gate", {}).get("passed") is not True
        or parent_json["v76_receipt_sha256"].get("gate", {}).get("passed") is not False
        or parent_json["v77_receipt_sha256"].get("summary", {}).get("classification")
        != "cell_support_or_direction_shift"
        or parent_json["v78_receipt_sha256"].get("gate", {}).get("passed") is not False
        or kernel_json["v37_terminal_decision_sha256"].get("scientific_gate_passed")
        is not True
        or kernel_json["v39_stability_receipt_sha256"].get("gate", {}).get("passed")
        is not True
        or kernel_json["v40_ggml_receipt_sha256"].get("gate", {}).get("passed")
        is not True
    ):
        raise ValueError("v79 frozen protocol or authenticated parents changed")
    if ADAPTER_BYTES != int(prereg["adapter"]["persistent_bytes"]):
        raise ValueError("v79 adapter byte formula changed")

    from transformers import AutoConfig, AutoTokenizer

    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("v79 Qwen3.8-27B model index changed")
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in auth["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v79 source tensor changed: {name}")

    context_receipt = parent_json["full_context_receipt_sha256"]
    context_path = args.full_context_root / "context.pt"
    shard_path = args.full_context_root / "layers" / f"{LAYER}.pt"
    if (
        v71.file_sha256(context_path) != context_receipt["context"]["sha256"]
        or v71.file_sha256(shard_path)
        != context_receipt["layers"][LAYER]["shard"]["sha256"]
    ):
        raise ValueError("v79 full-context artifacts changed")
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    fit_rows = context["calibration_flat_rows"].long()
    validation_rows = context["validation_flat_rows"].long()
    selected_rows = torch.cat((fit_rows, validation_rows))
    validation_owners = v78.environment_owners(
        torch.div(validation_rows, 64, rounding_mode="floor")
    )
    hidden = shard["input"].contiguous()
    positions, masks = qat.make_context(
        config,
        hidden,
        context["attention_mask"].long(),
        context["text_position_ids"].long(),
    )
    exact_layer = block_parent.instantiate_layer(config, LAYER, state)

    def v42_exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=positions,
            attention_mask=masks[config.layer_types[LAYER]],
            position_ids=context["text_position_ids"].long(),
        )

    exact_output, v42_mlp_input, v42_teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, v42_exact_forward, selected_rows
        )
    )
    if not torch.equal(exact_output, shard["output"]):
        raise ValueError("v79 v42 exact layer replay changed")
    del exact_output, hidden, shard, positions, masks
    gc.collect()

    payload_root = args.v44_artifact_root / "layers" / str(LAYER) / "cq3"
    payloads = {
        role: payload_root / f"blk.{LAYER}.ffn_{role}.weight.cq3"
        for role in ("gate", "up", "down")
    }
    for role, path in payloads.items():
        if (
            path.stat().st_size != CQ3_PAYLOAD_BYTES
            or v71.file_sha256(path) != auth["v44_payload_sha256"][role]
        ):
            raise ValueError(f"v79 CQ3 payload changed: {role}")
    gate = v71.load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = v71.load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = v71.load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
    source_down = state["mlp.down_proj.weight"]
    with torch.no_grad():
        v42_exact_mlp = functional.linear(v42_teacher_down_input.float(), source_down.float())
        v42_down_input = (
            functional.silu(gate(v42_mlp_input).float()) * up(v42_mlp_input).float()
        ).to(torch.bfloat16)
        v42_baseline_mlp = down(v42_down_input).float()
    v42_cell_correction, packed_checks = load_v75_correction(
        args.v75_artifact, down, v42_down_input
    )
    v42_cell_candidate = v42_baseline_mlp + v42_cell_correction
    fit_count = fit_rows.numel()
    v42_cell_fit_metric = relative_improvement(
        v42_baseline_mlp[:fit_count],
        v42_cell_candidate[:fit_count],
        v42_exact_mlp[:fit_count],
    )
    v42_cell_validation_metric = relative_improvement(
        v42_baseline_mlp[fit_count:],
        v42_cell_candidate[fit_count:],
        v42_exact_mlp[fit_count:],
    )
    adapter = quantize_residual_adapter(
        fit_float_residual_adapter(
            v42_down_input[:fit_count],
            v42_cell_candidate[:fit_count],
            v42_exact_mlp[:fit_count],
            rank=RANK,
            ridge_fraction=RIDGE_FRACTION,
        )
    )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    adapter_path = args.artifact_root / "blk.59.ffn_down.v75-residual-rank16-int8"
    if not adapter_path.exists():
        write_int8_adapter(adapter_path, adapter)
    restored = load_int8_adapter(
        adapter_path,
        input_features=INTERMEDIATE,
        output_features=HIDDEN,
        rank=RANK,
        ridge_fraction=RIDGE_FRACTION,
    )
    payload_roundtrip_exact = all(
        torch.equal(getattr(adapter, field), getattr(restored, field))
        for field in (
            "left_codes",
            "left_scales",
            "right_codes",
            "right_scales",
            "bias",
        )
    )
    v42_composed = v42_cell_candidate + restored.correction(v42_down_input)
    fit_metric = relative_improvement(
        v42_baseline_mlp[:fit_count],
        v42_composed[:fit_count],
        v42_exact_mlp[:fit_count],
    )
    validation_metric = relative_improvement(
        v42_baseline_mlp[fit_count:],
        v42_composed[fit_count:],
        v42_exact_mlp[fit_count:],
    )
    validation_prompt_records = prompt_metrics(
        v42_exact_mlp[fit_count:],
        v42_baseline_mlp[fit_count:],
        v42_composed[fit_count:],
        validation_owners,
    )

    v76_spec = json.loads(args.v76_preregistration.read_text())
    encoded = tokenizer(
        list(v76_spec["fresh_prompt_protocol"]["prompts"]),
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )
    fresh_ids = encoded["input_ids"]
    fresh_mask = encoded["attention_mask"]
    fresh_rows = cache_v1.selected_flat_rows(fresh_mask, range(8), 64)
    fresh_owners = v78.environment_owners(torch.div(fresh_rows, 64, rounding_mode="floor"))
    v76_parent = parent_json["v76_receipt_sha256"]
    if (
        tensor_sha256(fresh_ids) != v76_parent["fresh_data"]["input_ids_sha256"]
        or tensor_sha256(fresh_rows)
        != v76_parent["fresh_data"]["selected_rows_sha256"]
    ):
        raise ValueError("v79 v76 data replay changed")
    checkpoint = torch.load(args.v76_checkpoint, map_location="cpu", weights_only=False)
    fresh_hidden = checkpoint["hidden"].contiguous()
    if (
        checkpoint.get("next_layer") != LAYER
        or tensor_sha256(fresh_hidden)
        != v76_parent["exact_propagation"]["layer59_input_sha256"]
    ):
        raise ValueError("v79 v76 checkpoint replay changed")
    fresh_position_ids = torch.arange(64, dtype=torch.long).expand(8, -1)
    fresh_positions, fresh_masks = qat.make_context(
        config, fresh_hidden, fresh_mask, fresh_position_ids
    )

    def fresh_exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            fresh_hidden,
            position_embeddings=fresh_positions,
            attention_mask=fresh_masks[config.layer_types[LAYER]],
            position_ids=fresh_position_ids,
        )

    fresh_exact_output, fresh_mlp_input, fresh_teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, fresh_exact_forward, fresh_rows
        )
    )
    fresh_exact_replay = (
        tensor_sha256(fresh_exact_output)
        == v76_parent["exact_propagation"]["layer59_output_sha256"]
    )
    with torch.no_grad():
        fresh_exact_mlp = functional.linear(
            fresh_teacher_down_input.float(), source_down.float()
        )
        fresh_down_input = (
            functional.silu(gate(fresh_mlp_input).float()) * up(fresh_mlp_input).float()
        ).to(torch.bfloat16)
        fresh_baseline_mlp = down(fresh_down_input).float()
    fresh_cell_correction, fresh_packed_checks = load_v75_correction(
        args.v75_artifact, down, fresh_down_input
    )
    fresh_cell_candidate = fresh_baseline_mlp + fresh_cell_correction
    fresh_cell_metric = relative_improvement(
        fresh_baseline_mlp, fresh_cell_candidate, fresh_exact_mlp
    )
    fresh_composed = fresh_cell_candidate + restored.correction(fresh_down_input)
    fresh_metric = relative_improvement(
        fresh_baseline_mlp, fresh_composed, fresh_exact_mlp
    )
    fresh_prompt_records = prompt_metrics(
        fresh_exact_mlp, fresh_baseline_mlp, fresh_composed, fresh_owners
    )
    fresh_exact_tiled = fresh_exact_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    fresh_base_tiled = fresh_baseline_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    fresh_composed_tiled = fresh_composed.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    base_tile_error = (fresh_exact_tiled - fresh_base_tiled).square().sum(dim=(0, 2))
    composed_tile_error = (
        fresh_exact_tiled - fresh_composed_tiled
    ).square().sum(dim=(0, 2))
    positive_tile_fraction = float((composed_tile_error < base_tile_error).float().mean())
    improved_prompt_fraction = sum(
        record["relative_improvement"] > 0 for record in fresh_prompt_records
    ) / len(fresh_prompt_records)
    median_directional_capacity = float(
        torch.quantile(
            torch.tensor(
                [record["directional_capacity"] for record in fresh_prompt_records],
                dtype=torch.float64,
            ),
            0.5,
        )
    )

    composed_payload_bytes = (
        parent_json["v75_receipt_sha256"]["storage"][
            "hybrid_three_projection_mlp_payload_bytes"
        ]
        + adapter_path.stat().st_size
    )
    baseline_payload_bytes = 3 * CQ3_PAYLOAD_BYTES
    storage_increase = (composed_payload_bytes - baseline_payload_bytes) / baseline_payload_bytes
    gate_spec = prereg["development_gate"]
    v75_v42_reference = parent_json["v75_receipt_sha256"]["metrics"][
        "relative_development_complete_mlp_nmse_improvement"
    ]
    v75_v76_reference = v76_parent["metrics"][
        "relative_fresh_complete_mlp_nmse_improvement"
    ]
    checks = {
        "all_parent_artifacts_authenticated": True,
        "v42_exact_layer_replay": True,
        "v76_exact_layer_replay": fresh_exact_replay,
        "v42_validation_used_for_refit": False,
        "v76_used_for_refit": False,
        "packed_v75_roundtrip_exact": all(packed_checks.values())
        and all(fresh_packed_checks.values()),
        "v75_v42_fit_metric_reproduced": abs(
            v42_cell_fit_metric[2]
            - parent_json["v75_receipt_sha256"]["metrics"][
                "relative_calibration_complete_mlp_nmse_improvement"
            ]
        )
        <= 1e-7,
        "v75_v42_validation_metric_reproduced": abs(
            v42_cell_validation_metric[2] - v75_v42_reference
        )
        <= 1e-7,
        "v75_v76_metric_reproduced": abs(fresh_cell_metric[2] - v75_v76_reference)
        <= 1e-7,
        "adapter_payload_roundtrip_exact": payload_roundtrip_exact,
        "adapter_payload_bytes_exact": adapter_path.stat().st_size
        == int(gate_spec["required_adapter_bytes"]),
        "all_values_finite": bool(
            torch.isfinite(v42_composed).all()
            and torch.isfinite(fresh_composed).all()
        ),
        "minimum_fit_improvement_met": fit_metric[2]
        >= float(gate_spec["minimum_relative_fit_complete_mlp_nmse_improvement"]),
        "minimum_v42_validation_improvement_met": validation_metric[2]
        >= float(
            gate_spec["minimum_relative_v42_validation_complete_mlp_nmse_improvement"]
        ),
        "minimum_v42_gain_over_v75_met": validation_metric[2] - v75_v42_reference
        >= float(gate_spec["minimum_absolute_v42_validation_improvement_over_v75"]),
        "minimum_v76_development_improvement_met": fresh_metric[2]
        >= float(
            gate_spec["minimum_relative_v76_development_complete_mlp_nmse_improvement"]
        ),
        "minimum_v76_gain_over_v75_met": fresh_metric[2] - v75_v76_reference
        >= float(gate_spec["minimum_absolute_v76_improvement_over_v75"]),
        "minimum_v76_positive_tile_fraction_met": positive_tile_fraction
        >= float(gate_spec["minimum_v76_positive_output_tile_fraction"]),
        "minimum_v76_improved_prompt_fraction_met": improved_prompt_fraction
        >= float(gate_spec["minimum_v76_improved_prompt_fraction"]),
        "minimum_v76_median_directional_capacity_met": median_directional_capacity
        >= float(gate_spec["minimum_v76_median_prompt_directional_capacity"]),
        "maximum_payload_increase_met": storage_increase
        <= float(gate_spec["maximum_relative_full_mlp_payload_increase"]),
    }
    passed = all(checks.values())
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": v71.file_sha256(args.preregistration),
        "authenticated_inputs": {
            key: v71.file_sha256(path) for key, path in parents.items()
        }
        | {
            "v75_hybrid_payload_sha256": v71.file_sha256(args.v75_artifact),
            "v76_checkpoint_sha256": v71.file_sha256(args.v76_checkpoint),
            "source_tensor_sha256": auth["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
            "prior_exact_kernel_evidence": {
                key: v71.file_sha256(path) for key, path in kernel_paths.items()
            },
        },
        "adapter": {
            "rank": RANK,
            "ridge_fraction": RIDGE_FRACTION,
            "input_features": INTERMEDIATE,
            "output_features": HIDDEN,
            "fit_rows": fit_count,
            "payload": {
                "path": str(adapter_path),
                "bytes": adapter_path.stat().st_size,
                "sha256": v71.file_sha256(adapter_path),
            },
            "left_codes_sha256": tensor_sha256(restored.left_codes),
            "left_scales_sha256": tensor_sha256(restored.left_scales),
            "right_codes_sha256": tensor_sha256(restored.right_codes),
            "right_scales_sha256": tensor_sha256(restored.right_scales),
            "bias_sha256": tensor_sha256(restored.bias),
            "rank_or_ridge_sweep": False,
            "macs_per_token": RANK * (INTERMEDIATE + HIDDEN),
            "rank_scratch_bytes_per_task": 4 * RANK,
            "high_precision_factor_materialization_at_runtime": False,
        },
        "metrics": {
            "fit": {
                "baseline_complete_mlp_nmse": fit_metric[0],
                "candidate_complete_mlp_nmse": fit_metric[1],
                "relative_improvement": fit_metric[2],
                "v75_cell_only_relative_improvement": v42_cell_fit_metric[2],
            },
            "v42_validation_development": {
                "baseline_complete_mlp_nmse": validation_metric[0],
                "candidate_complete_mlp_nmse": validation_metric[1],
                "relative_improvement": validation_metric[2],
                "v75_reference": v75_v42_reference,
                "reproduced_v75_cell_only_relative_improvement": v42_cell_validation_metric[2],
                "absolute_improvement_over_v75": validation_metric[2]
                - v75_v42_reference,
                "per_prompt": validation_prompt_records,
            },
            "v76_development": {
                "baseline_complete_mlp_nmse": fresh_metric[0],
                "candidate_complete_mlp_nmse": fresh_metric[1],
                "relative_improvement": fresh_metric[2],
                "v75_reference": v75_v76_reference,
                "reproduced_v75_cell_only_relative_improvement": fresh_cell_metric[2],
                "absolute_improvement_over_v75": fresh_metric[2]
                - v75_v76_reference,
                "positive_output_tile_fraction": positive_tile_fraction,
                "improved_prompt_fraction": improved_prompt_fraction,
                "median_prompt_directional_capacity": median_directional_capacity,
                "per_prompt": fresh_prompt_records,
            },
        },
        "storage_and_compute": {
            "baseline_three_cq3_mlp_payload_bytes": baseline_payload_bytes,
            "v75_three_projection_mlp_payload_bytes": parent_json[
                "v75_receipt_sha256"
            ]["storage"]["hybrid_three_projection_mlp_payload_bytes"],
            "adapter_payload_bytes": adapter_path.stat().st_size,
            "composed_three_projection_mlp_payload_bytes": composed_payload_bytes,
            "relative_full_mlp_payload_increase": storage_increase,
            "additional_macs_per_token_vs_v75": RANK * (INTERMEDIATE + HIDDEN),
            "additional_runtime_passes_vs_v75": 2,
        },
        "data_status": prereg["data_status"],
        "checks": checks,
        "gate": {
            "passed": passed,
            "decision": (
                "freeze_hierarchical_cell_subspace_composition_for_second_new_prompt_replication"
                if passed
                else "reject_rank16_residual_after_v75_cell_path"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    v76.atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_hierarchical_cell_subspace_residual_marker_v79",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": passed,
        "terminal": True,
    }
    v76.atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
