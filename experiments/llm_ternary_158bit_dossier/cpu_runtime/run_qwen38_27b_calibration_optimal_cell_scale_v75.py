#!/usr/bin/env python3
"""Fit deployment-native INT4 cell scales by conditional least squares."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
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
import run_qwen38_27b_exact_marginal_tile_mixed_down_v73 as v73
import run_qwen38_27b_greedy_input_cell_mixed_down_v74 as v74
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_mixed_cq3_runtime import tensor_sha256

MODEL_ID = v71.MODEL_ID
MODEL_REVISION = v71.MODEL_REVISION
LAYER = v71.LAYER
HIDDEN = v71.HIDDEN
INTERMEDIATE = v71.INTERMEDIATE
CQ3_PAYLOAD_BYTES = v71.PAYLOAD_BYTES
TILE_ROWS = v74.TILE_ROWS
OUTPUT_TILES = v74.OUTPUT_TILES
GROUP_SIZE = v74.GROUP_SIZE
INPUT_GROUPS = v74.INPUT_GROUPS
SCHEMA = "qwen38_27b_calibration_optimal_cell_scale_receipt_v75"


def greedy_select_optimal_scale_cells(
    code_response: torch.Tensor,
    base_response: torch.Tensor,
    residual: torch.Tensor,
    *,
    steps: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, float | int]],
]:
    if (
        code_response.ndim != 4
        or code_response.shape != base_response.shape
        or residual.shape
        != (code_response.shape[0], code_response.shape[1], code_response.shape[3])
        or not 0 < steps < code_response.shape[2]
    ):
        raise ValueError("invalid optimal-scale cell-selection inputs")
    code = code_response.float()
    base = base_response.float()
    working = residual.float().clone()
    code_energy = code.square().sum(dim=0)
    code_dot_base = (code * base).sum(dim=0)
    base_energy = base.square().sum(dim=0)
    selected_mask = torch.zeros(
        code.shape[1], code.shape[2], dtype=torch.bool, device=code.device
    )
    selected = torch.empty(code.shape[1], steps, dtype=torch.long)
    selected_scales = torch.empty(
        code.shape[1], steps, code.shape[3], dtype=torch.float16
    )
    tile_index = torch.arange(code.shape[1], device=code.device)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        working_dot_code = torch.einsum("ntr,ntgr->tgr", working, code)
        working_dot_base = torch.einsum("ntr,ntgr->tgr", working, base)
        numerator = working_dot_code + code_dot_base
        candidate_scale = torch.where(
            code_energy > 0,
            numerator / code_energy.clamp_min(1e-30),
            torch.zeros_like(numerator),
        ).clamp_min(0)
        stored_scale = candidate_scale.to(torch.float16)
        deployed_scale = stored_scale.float()
        correlation = (
            deployed_scale * working_dot_code - working_dot_base
        ).sum(dim=-1)
        delta_energy = (
            deployed_scale.square() * code_energy
            - 2 * deployed_scale * code_dot_base
            + base_energy
        ).sum(dim=-1)
        benefit = 2 * correlation - delta_energy
        benefit.masked_fill_(selected_mask, -torch.inf)
        best_group = benefit.argmax(dim=1)
        best_benefit = benefit[tile_index, best_group]
        best_scale = stored_scale[tile_index, best_group]
        selected[:, step] = best_group.cpu()
        selected_scales[:, step] = best_scale.cpu()
        selected_mask[tile_index, best_group] = True
        chosen_delta = (
            best_scale.float().unsqueeze(0)
            * code[:, tile_index, best_group, :]
            - base[:, tile_index, best_group, :]
        )
        working.sub_(chosen_delta)
        history.append(
            {
                "step": step + 1,
                "minimum_selected_marginal_benefit": float(best_benefit.min()),
                "median_selected_marginal_benefit": float(best_benefit.median()),
                "maximum_selected_marginal_benefit": float(best_benefit.max()),
                "minimum_selected_scale": float(best_scale.float().min()),
                "maximum_selected_scale": float(best_scale.float().max()),
            }
        )
    return selected, selected_scales, working, history


def selected_optimal_scale_correction(
    code_response: torch.Tensor,
    base_response: torch.Tensor,
    selected: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if (
        selected.shape != scales.shape[:2]
        or scales.shape[2] != code_response.shape[3]
        or selected.shape[0] != code_response.shape[1]
        or code_response.shape != base_response.shape
    ):
        raise ValueError("selected optimal-scale cells changed")
    result = torch.zeros(
        code_response.shape[0],
        code_response.shape[1],
        code_response.shape[3],
        dtype=torch.float32,
    )
    tile_index = torch.arange(code_response.shape[1])
    for step in range(selected.shape[1]):
        result.add_(
            scales[:, step].float().unsqueeze(0)
            * code_response[:, tile_index, selected[:, step], :].float()
            - base_response[:, tile_index, selected[:, step], :].float()
        )
    return result


def install_selected_scales(
    parent_scales: torch.Tensor,
    selected: torch.Tensor,
    selected_scales: torch.Tensor,
) -> torch.Tensor:
    if (
        parent_scales.shape != (HIDDEN, INPUT_GROUPS)
        or selected.shape != selected_scales.shape[:2]
        or selected_scales.shape[2] != TILE_ROWS
    ):
        raise ValueError("optimal-scale installation shape changed")
    result = parent_scales.clone().reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS
    )
    tile_index = torch.arange(OUTPUT_TILES)
    for step in range(selected.shape[1]):
        result[tile_index, :, selected[:, step]] = selected_scales[:, step]
    return result.reshape(HIDDEN, INPUT_GROUPS).contiguous()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


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
    parser.add_argument("--v74-receipt", required=True, type=Path)
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
        raise ValueError("v75 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v75 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v75 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = v71.file_sha256(args.preregistration)
    full_context_sha = v71.file_sha256(args.full_context_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v74_sha = v71.file_sha256(args.v74_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    prior = json.loads(args.v74_receipt.read_text())
    authenticated = prereg["authenticated_inputs"]
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_calibration_optimal_cell_scale_preregistration_v75"
        or prereg.get("status") != "frozen_development_screen_before_execution"
        or prereg.get("data_status", {}).get(
            "development_confirmation_is_architecture_contaminated"
        )
        is not True
        or authenticated.get("full_context_receipt_sha256") != full_context_sha
        or authenticated.get("v44_training_receipt_sha256") != v44_sha
        or authenticated.get("v74_negative_receipt_sha256") != v74_sha
        or context_receipt.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
        or prior.get("gate", {}).get("passed") is not False
        or prior.get("gate", {}).get("decision")
        != "reject_source_int4_input_cell_substitution"
        or prior.get("metrics", {}).get(
            "relative_confirmation_complete_mlp_nmse_improvement", 0
        )
        <= 0.08
    ):
        raise ValueError("v75 frozen protocol or authenticated parents changed")

    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("Qwen3.8-27B model index changed")
    layer_record = context_receipt["layers"][LAYER]
    shard_path = args.full_context_root / "layers" / f"{LAYER}.pt"
    context_path = args.full_context_root / "context.pt"
    if (
        v71.file_sha256(shard_path) != layer_record["shard"]["sha256"]
        or v71.file_sha256(context_path) != context_receipt["context"]["sha256"]
    ):
        raise ValueError("v75 full-context artifacts changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in authenticated["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v75 source tensor changed: {name}")

    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    calibration_rows = context["calibration_flat_rows"].long()
    validation_rows = context["validation_flat_rows"].long()
    selected_rows = torch.cat((calibration_rows, validation_rows))
    hidden = shard["input"].contiguous()
    teacher_layer_output = shard["output"].contiguous()
    attention_mask = context["attention_mask"].long()
    position_ids = context["text_position_ids"].long()
    if (
        hidden.shape != (12, 64, HIDDEN)
        or calibration_rows.numel() != 64
        or validation_rows.numel() != 32
        or int(calibration_rows.max()) >= 8 * 64
        or int(validation_rows.min()) < 8 * 64
    ):
        raise ValueError("v75 development data split changed")

    exact_layer = block_parent.instantiate_layer(config, LAYER, state)
    position_embeddings, masks = qat.make_context(
        config, hidden, attention_mask, position_ids
    )
    layer_type = config.layer_types[LAYER]

    def exact_forward(
        layer: torch.nn.Module = exact_layer,
        layer_input: torch.Tensor = hidden,
    ) -> torch.Tensor:
        return qat.forward_with_grad(
            layer,
            layer_input,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
        )

    exact_output, mlp_input, teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, exact_forward, selected_rows
        )
    )
    if not torch.equal(exact_output, teacher_layer_output):
        raise ValueError("v75 exact layer replay changed")
    del exact_forward, exact_layer, exact_output, teacher_layer_output, hidden, shard
    gc.collect()

    payload_root = args.v44_artifact_root / "layers" / str(LAYER) / "cq3"
    payloads = {
        role: payload_root / f"blk.{LAYER}.ffn_{role}.weight.cq3"
        for role in ("gate", "up", "down")
    }
    for role, path in payloads.items():
        if (
            path.stat().st_size != CQ3_PAYLOAD_BYTES
            or v71.file_sha256(path)
            != authenticated["v44_payload_sha256"][role]
        ):
            raise ValueError(f"v75 trained {role} payload changed")

    gate = v71.load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = v71.load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = v71.load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
    source_down = state["mlp.down_proj.weight"]
    del state
    with torch.no_grad():
        exact_mlp_output = functional.linear(
            teacher_down_input.float(), source_down.float()
        )
        baseline_down_input = (
            functional.silu(gate(mlp_input).float()) * up(mlp_input).float()
        ).to(torch.bfloat16)
        baseline_mlp_output = down(baseline_down_input).float()
    del gate, up
    gc.collect()

    int4_codes, parent_int4_scales = v73.quantize_signed_int4_grouped(
        source_down,
        group_size=GROUP_SIZE,
        iterations=4,
    )
    cq3_codes = down.codes.reshape(HIDDEN, INPUT_GROUPS, GROUP_SIZE).contiguous()
    cq3_scales = down.scales.contiguous()
    with torch.no_grad():
        code = int4_codes.reshape(
            OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE
        ).float()
        base_weight = (
            cq3_codes.float() * cq3_scales.float().unsqueeze(-1)
        ).reshape(OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE)
        grouped_input = baseline_down_input.float().reshape(
            selected_rows.numel(), INPUT_GROUPS, GROUP_SIZE
        )
        code_response = torch.einsum("trgd,ngd->ntgr", code, grouped_input)
        base_response = torch.einsum("trgd,ngd->ntgr", base_weight, grouped_input)
    del down, code, base_weight, grouped_input, source_down
    gc.collect()

    calibration_count = calibration_rows.numel()
    baseline_tiled = baseline_mlp_output.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    exact_tiled = exact_mlp_output.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    calibration_residual = (
        exact_tiled[:calibration_count] - baseline_tiled[:calibration_count]
    )
    steps = int(prereg["int4_cell"]["greedy_steps_per_output_tile"])
    selected_groups, selected_scales, final_residual, history = (
        greedy_select_optimal_scale_cells(
            code_response[:calibration_count],
            base_response[:calibration_count],
            calibration_residual,
            steps=steps,
        )
    )
    correction = selected_optimal_scale_correction(
        code_response, base_response, selected_groups, selected_scales
    )
    candidate_tiled = baseline_tiled + correction
    candidate_mlp_output = candidate_tiled.reshape_as(baseline_mlp_output)
    if not torch.allclose(
        final_residual,
        exact_tiled[:calibration_count] - candidate_tiled[:calibration_count],
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError("v75 conditional least-squares residual changed")

    baseline_calibration_nmse = v71.tensor_nmse(
        baseline_mlp_output[:calibration_count], exact_mlp_output[:calibration_count]
    )
    candidate_calibration_nmse = v71.tensor_nmse(
        candidate_mlp_output[:calibration_count], exact_mlp_output[:calibration_count]
    )
    baseline_development_nmse = v71.tensor_nmse(
        baseline_mlp_output[calibration_count:], exact_mlp_output[calibration_count:]
    )
    candidate_development_nmse = v71.tensor_nmse(
        candidate_mlp_output[calibration_count:], exact_mlp_output[calibration_count:]
    )
    calibration_improvement = (
        baseline_calibration_nmse - candidate_calibration_nmse
    ) / max(baseline_calibration_nmse, 1e-30)
    development_improvement = (
        baseline_development_nmse - candidate_development_nmse
    ) / max(baseline_development_nmse, 1e-30)
    development_ratio = development_improvement / max(calibration_improvement, 1e-30)
    development_base_error = (
        exact_tiled[calibration_count:] - baseline_tiled[calibration_count:]
    ).square().sum(dim=(0, 2))
    development_candidate_error = (
        exact_tiled[calibration_count:] - candidate_tiled[calibration_count:]
    ).square().sum(dim=(0, 2))
    positive_development_tile_fraction = float(
        (development_base_error > development_candidate_error).float().mean()
    )

    optimized_int4_scales = install_selected_scales(
        parent_int4_scales, selected_groups, selected_scales
    )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    hybrid_path = args.artifact_root / "blk.59.ffn_down.cellmix-cq3-int4-opt-scale"
    hybrid_payload, storage, packed_sources = v74.build_hybrid_payload(
        cq3_codes,
        cq3_scales,
        int4_codes,
        optimized_int4_scales,
        selected_groups,
    )
    if not hybrid_path.exists():
        atomic_bytes(hybrid_path, hybrid_payload)
    elif hybrid_path.read_bytes() != hybrid_payload:
        raise ValueError("existing v75 hybrid payload changed")

    cq3_offset = v74.HEADER.size + storage["cell_mode_bitmap_bytes"]
    int4_offset = cq3_offset + storage["unselected_cq3_bytes"]
    checked_int4_codes, checked_int4_scales = v74.unpack_int4_group_stream(
        hybrid_payload[int4_offset:],
        rows=packed_sources["int4_codes"].shape[0] * TILE_ROWS,
        groups=1,
    )
    int4_roundtrip_exact = torch.equal(
        checked_int4_codes,
        packed_sources["int4_codes"].reshape(-1, 1, GROUP_SIZE),
    ) and torch.equal(
        checked_int4_scales,
        packed_sources["int4_scales"].reshape(-1, 1),
    )
    full_mlp_bytes = 2 * CQ3_PAYLOAD_BYTES + storage["total_bytes"]
    storage_increase = (full_mlp_bytes - 3 * CQ3_PAYLOAD_BYTES) / (
        3 * CQ3_PAYLOAD_BYTES
    )
    all_selected_positive = min(
        record["minimum_selected_marginal_benefit"] for record in history
    ) > 0

    gate_spec = prereg["development_gate"]
    checks = {
        "exact_layer_replay": True,
        "all_parent_artifacts_authenticated": True,
        "development_confirmation_marked_architecture_contaminated": True,
        "all_values_finite": bool(
            torch.isfinite(code_response).all()
            and torch.isfinite(selected_scales.float()).all()
            and torch.isfinite(candidate_mlp_output).all()
        ),
        "required_selected_cells_met": selected_groups.numel()
        == int(gate_spec["required_selected_cells"]),
        "all_selected_calibration_marginal_benefits_positive": all_selected_positive,
        "minimum_calibration_improvement_met": calibration_improvement
        >= float(gate_spec["minimum_relative_calibration_mlp_nmse_improvement"]),
        "minimum_development_improvement_met": development_improvement
        >= float(
            gate_spec["minimum_relative_development_confirmation_mlp_nmse_improvement"]
        ),
        "minimum_development_ratio_met": development_ratio
        >= float(
            gate_spec["minimum_development_confirmation_to_calibration_ratio"]
        ),
        "minimum_positive_development_tile_fraction_met": (
            positive_development_tile_fraction
            >= float(
                gate_spec[
                    "minimum_development_confirmation_positive_output_tile_fraction"
                ]
            )
        ),
        "packed_int4_roundtrip_exact": int4_roundtrip_exact,
        "canonical_payload_bytes_exact": storage["total_bytes"]
        == int(prereg["canonical_layout"]["hybrid_down_payload_bytes"]),
        "maximum_full_mlp_payload_increase_met": storage_increase
        <= float(gate_spec["maximum_relative_full_mlp_payload_increase"]),
    }
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "authenticated_inputs": {
            "full_context_receipt_sha256": full_context_sha,
            "v44_training_receipt_sha256": v44_sha,
            "v74_negative_receipt_sha256": v74_sha,
            "layer59_context_shard_sha256": v71.file_sha256(shard_path),
            "source_tensor_sha256": authenticated["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
        },
        "data_status": {
            "selection_rows": calibration_count,
            "development_confirmation_rows": validation_rows.numel(),
            "development_confirmation_is_architecture_contaminated": True,
            "fresh_holdout_claim_forbidden": True,
        },
        "cell_selection": {
            "method": "greedy exact residual reduction with conditional FP16 least-squares cell scales",
            "steps_per_output_tile": selected_groups.shape[1],
            "selected_cells": selected_groups.numel(),
            "selected_groups_by_output_tile": selected_groups.tolist(),
            "selected_groups_sha256": v74.tensor_bytes_sha256(selected_groups),
            "selected_scales_sha256": v74.tensor_bytes_sha256(selected_scales),
            "minimum_selected_scale": float(selected_scales.float().min()),
            "maximum_selected_scale": float(selected_scales.float().max()),
            "zero_selected_scales": int((selected_scales == 0).sum()),
            "greedy_history": history,
            "development_positive_output_tile_fraction": positive_development_tile_fraction,
        },
        "metrics": {
            "baseline_calibration_complete_mlp_nmse": baseline_calibration_nmse,
            "candidate_calibration_complete_mlp_nmse": candidate_calibration_nmse,
            "relative_calibration_complete_mlp_nmse_improvement": calibration_improvement,
            "baseline_development_complete_mlp_nmse": baseline_development_nmse,
            "candidate_development_complete_mlp_nmse": candidate_development_nmse,
            "relative_development_complete_mlp_nmse_improvement": development_improvement,
            "development_to_calibration_improvement_ratio": development_ratio,
        },
        "storage": {
            **storage,
            "hybrid_payload_sha256": v71.file_sha256(hybrid_path),
            "baseline_three_cq3_mlp_payload_bytes": 3 * CQ3_PAYLOAD_BYTES,
            "hybrid_three_projection_mlp_payload_bytes": full_mlp_bytes,
            "relative_full_mlp_payload_increase": storage_increase,
            "additional_runtime_bytes_vs_v74": 0,
            "additional_runtime_operations_vs_v74": 0,
            "additional_runtime_passes": 0,
            "dispatch_cell_shape": [TILE_ROWS, GROUP_SIZE],
            "high_precision_weight_materialization_at_runtime": False,
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "decision": (
                "freeze_optimal_cell_scales_for_new_prompt_replication"
                if all(checks.values())
                else "reject_fixed_code_calibration_optimal_cell_scaling"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_calibration_optimal_cell_scale_marker_v75",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": receipt["gate"]["passed"],
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
