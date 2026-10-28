#!/usr/bin/env python3
"""Diagnose prompt-conditioned geometry behind the strict v76 near miss."""

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
import run_qwen38_27b_affine_butterfly_block_gate_v1 as block_parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_all_layer_mixed_direct_v43 as direct_v43
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_optimal_cell_scale_fresh_replication_v76 as v76
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

from clark_quantize.qwen38_mixed_cq3_runtime import tensor_sha256

MODEL_ID = v76.MODEL_ID
MODEL_REVISION = v76.MODEL_REVISION
LAYER = v76.LAYER
HIDDEN = v76.HIDDEN
INTERMEDIATE = v76.INTERMEDIATE
CQ3_PAYLOAD_BYTES = v76.CQ3_PAYLOAD_BYTES
OUTPUT_TILES = v76.OUTPUT_TILES
TILE_ROWS = v76.TILE_ROWS
INPUT_GROUPS = v76.INPUT_GROUPS
GROUP_SIZE = v76.GROUP_SIZE
SCHEMA = "qwen38_27b_cell_scale_covariate_shift_diagnostic_receipt_v77"


def vector_geometry(residual: torch.Tensor, correction: torch.Tensor) -> dict[str, float]:
    residual64 = residual.double().reshape(-1)
    correction64 = correction.double().reshape(-1)
    residual_energy = float(torch.dot(residual64, residual64))
    correction_energy = float(torch.dot(correction64, correction64))
    dot = float(torch.dot(residual64, correction64))
    if residual_energy <= 0 or correction_energy <= 0:
        raise ValueError("v77 vector geometry requires nonzero finite energies")
    cosine = dot / (residual_energy * correction_energy) ** 0.5
    optimal_scalar = dot / correction_energy
    deployed_improvement = (2 * dot - correction_energy) / residual_energy
    nonnegative_directional_capacity = (
        max(dot, 0.0) ** 2 / (residual_energy * correction_energy)
    )
    optimal_improvement = dot**2 / (residual_energy * correction_energy)
    deployed_fraction = deployed_improvement / max(
        nonnegative_directional_capacity, 1e-30
    )
    return {
        "residual_energy": residual_energy,
        "correction_energy": correction_energy,
        "residual_correction_dot": dot,
        "residual_correction_cosine": cosine,
        "unconstrained_optimal_scalar": optimal_scalar,
        "deployed_relative_sse_improvement": deployed_improvement,
        "nonnegative_scalar_directional_capacity": nonnegative_directional_capacity,
        "unconstrained_optimal_scalar_relative_sse_improvement": optimal_improvement,
        "deployed_fraction_of_nonnegative_directional_capacity": deployed_fraction,
    }


def quantiles(values: torch.Tensor) -> dict[str, float]:
    finite = values.double()[torch.isfinite(values)]
    if finite.numel() == 0:
        raise ValueError("v77 diagnostic quantile input has no finite values")
    return {
        name: float(torch.quantile(finite, probability))
        for name, probability in (
            ("minimum", 0.0),
            ("p10", 0.1),
            ("median", 0.5),
            ("p90", 0.9),
            ("maximum", 1.0),
        )
    }


def per_tile_optimal_scalars(
    residual: torch.Tensor, correction: torch.Tensor
) -> torch.Tensor:
    if residual.shape != correction.shape or residual.ndim != 3:
        raise ValueError("v77 per-tile geometry shape changed")
    numerator = (residual.double() * correction.double()).sum(dim=(0, 2))
    denominator = correction.double().square().sum(dim=(0, 2))
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.full_like(denominator, torch.nan),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--v76-preregistration", required=True, type=Path)
    parser.add_argument("--v76-receipt", required=True, type=Path)
    parser.add_argument("--v76-checkpoint", required=True, type=Path)
    parser.add_argument("--v75-artifact", required=True, type=Path)
    parser.add_argument(
        "--v44-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads != 4:
        raise ValueError("v77 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    cache_v1.resource_guard(Path("/workspace"))
    if args.out.exists():
        receipt = json.loads(args.out.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
        ):
            raise ValueError("existing v77 terminal diagnostic changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    prereg = json.loads(args.preregistration.read_text())
    v76_prereg = json.loads(args.v76_preregistration.read_text())
    parent = json.loads(args.v76_receipt.read_text())
    parent_spec = prereg["parent_result"]
    rules = prereg["classification_rule"]
    if (
        prereg.get("schema")
        != "qwen38_27b_cell_scale_covariate_shift_diagnostic_preregistration_v77"
        or prereg.get("status") != "frozen_postmortem_before_vector_diagnostics"
        or prereg.get("frozen_diagnostics", {}).get("no_parameter_fit_or_artifact_write")
        is not True
        or v71.file_sha256(args.v76_receipt) != parent_spec["v76_receipt_sha256"]
        or v71.file_sha256(args.v76_checkpoint)
        != parent_spec["v76_checkpoint_sha256"]
        or parent.get("gate", {}).get("passed") is not False
        or parent.get("gate", {}).get("decision")
        != "reject_v75_as_nonreplicating_development_result"
    ):
        raise ValueError("v77 frozen protocol or rejected parent changed")

    prompts = list(v76_prereg["fresh_prompt_protocol"]["prompts"])
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    selected_rows = cache_v1.selected_flat_rows(attention_mask, range(8), 64)
    owners = torch.div(selected_rows, 64, rounding_mode="floor")
    if (
        tensor_sha256(token_ids) != parent["fresh_data"]["input_ids_sha256"]
        or tensor_sha256(attention_mask)
        != parent["fresh_data"]["attention_mask_sha256"]
        or tensor_sha256(selected_rows)
        != parent["fresh_data"]["selected_rows_sha256"]
    ):
        raise ValueError("v77 fresh data replay changed")

    checkpoint = torch.load(args.v76_checkpoint, map_location="cpu", weights_only=False)
    hidden = checkpoint["hidden"].contiguous()
    if (
        checkpoint.get("next_layer") != LAYER
        or tensor_sha256(hidden) != checkpoint.get("hidden_sha256")
        or tensor_sha256(hidden) != parent["exact_propagation"]["layer59_input_sha256"]
    ):
        raise ValueError("v77 layer-59 checkpoint replay changed")
    position_ids = torch.arange(64, dtype=torch.long).expand(8, -1)
    position_embeddings, masks = qat.make_context(
        config, hidden, attention_mask, position_ids
    )

    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("v77 Qwen3.8-27B model index changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in parent["authenticated_inputs"]["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v77 source tensor changed: {name}")
    exact_layer = block_parent.instantiate_layer(config, LAYER, state)

    def exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[config.layer_types[LAYER]],
            position_ids=position_ids,
        )

    _, mlp_input, teacher_down_input = mixed_training.capture_exact_teacher_activations(
        exact_layer, exact_forward, selected_rows
    )
    del exact_layer, hidden
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
            != parent["authenticated_inputs"]["v44_payload_sha256"][role]
        ):
            raise ValueError(f"v77 CQ3 payload changed: {role}")
    gate = v71.load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = v71.load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = v71.load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
    with torch.no_grad():
        exact_mlp_output = functional.linear(
            teacher_down_input.float(), state["mlp.down_proj.weight"].float()
        )
        baseline_down_input = (
            functional.silu(gate(mlp_input).float()) * up(mlp_input).float()
        ).to(torch.bfloat16)
        baseline_mlp_output = down(baseline_down_input).float()
    del gate, up, state
    gc.collect()

    hybrid = args.v75_artifact.read_bytes()
    if (
        v71.file_sha256(args.v75_artifact)
        != parent["authenticated_inputs"]["v75_hybrid_payload_sha256"]
    ):
        raise ValueError("v77 v75 hybrid payload changed")
    mask, _, _, int4_codes, int4_scales = v76.decode_frozen_hybrid(hybrid)
    baseline_codes = down.codes.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE
    ).contiguous()
    baseline_scales = down.scales.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS
    ).contiguous()
    correction_tiled = v76.frozen_hybrid_correction(
        baseline_down_input.float().reshape(64, INPUT_GROUPS, GROUP_SIZE),
        baseline_codes,
        baseline_scales,
        mask,
        int4_codes,
        int4_scales,
    )
    residual_tiled = (
        exact_mlp_output - baseline_mlp_output
    ).reshape(64, OUTPUT_TILES, TILE_ROWS)
    aggregate = vector_geometry(residual_tiled, correction_tiled)
    per_prompt: list[dict[str, Any]] = []
    for prompt_index in range(8):
        prompt_rows = owners == prompt_index
        prompt_residual = residual_tiled[prompt_rows]
        prompt_correction = correction_tiled[prompt_rows]
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "selected_rows": int(prompt_rows.sum()),
                **vector_geometry(prompt_residual, prompt_correction),
                "per_tile_optimal_scalar_quantiles": quantiles(
                    per_tile_optimal_scalars(prompt_residual, prompt_correction)
                ),
            }
        )

    directional = torch.tensor(
        [record["nonnegative_scalar_directional_capacity"] for record in per_prompt],
        dtype=torch.float64,
    )
    efficiency = torch.tensor(
        [
            record["deployed_fraction_of_nonnegative_directional_capacity"]
            for record in per_prompt
        ],
        dtype=torch.float64,
    )
    median_directional = float(torch.quantile(directional, 0.5))
    median_efficiency = float(torch.quantile(efficiency, 0.5))
    if (
        median_directional
        >= float(rules["minimum_median_prompt_directional_capacity_for_magnitude_shift"])
        and median_efficiency
        < float(rules["maximum_median_deployed_fraction_of_capacity_for_magnitude_shift"])
    ):
        classification = "magnitude_covariate_shift"
    elif median_directional < float(
        rules["minimum_median_prompt_directional_capacity_for_magnitude_shift"]
    ):
        classification = "cell_support_or_direction_shift"
    else:
        classification = "mixed_or_well_scaled"

    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": v71.file_sha256(args.preregistration),
        "authenticated_inputs": {
            "v76_receipt_sha256": v71.file_sha256(args.v76_receipt),
            "v76_checkpoint_sha256": v71.file_sha256(args.v76_checkpoint),
            "v75_hybrid_payload_sha256": v71.file_sha256(args.v75_artifact),
            "layer59_input_sha256": tensor_sha256(checkpoint["hidden"]),
        },
        "diagnostic_only": {
            "fresh_data_is_architecture_contaminated": True,
            "parameters_fit": 0,
            "artifacts_written": 0,
            "diagnostic_scalars_are_deployable_candidates": False,
        },
        "aggregate_geometry": aggregate,
        "per_prompt_geometry": per_prompt,
        "summary": {
            "prompt_directional_capacity_quantiles": quantiles(directional),
            "prompt_deployed_fraction_of_capacity_quantiles": quantiles(efficiency),
            "median_prompt_directional_capacity": median_directional,
            "median_prompt_deployed_fraction_of_capacity": median_efficiency,
            "classification": classification,
            "next_estimator_family": prereg["decision"][f"if_{classification}"],
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    v76.atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
