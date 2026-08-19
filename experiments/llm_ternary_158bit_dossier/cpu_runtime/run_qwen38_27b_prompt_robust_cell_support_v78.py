#!/usr/bin/env python3
"""Select fixed-rate INT4 cells under prompt-group distributional risk."""

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
import run_qwen38_27b_calibration_optimal_cell_scale_v75 as v75
import run_qwen38_27b_exact_marginal_tile_mixed_down_v73 as v73
import run_qwen38_27b_greedy_input_cell_mixed_down_v74 as v74
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_optimal_cell_scale_fresh_replication_v76 as v76
import run_qwen38_27b_cell_scale_covariate_shift_diagnostic_v77 as v77
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
SCHEMA = "qwen38_27b_prompt_robust_cell_support_receipt_v78"


def environment_owners(raw_owners: torch.Tensor) -> torch.Tensor:
    unique = raw_owners.unique(sorted=True)
    result = torch.empty_like(raw_owners)
    for environment, owner in enumerate(unique):
        result[raw_owners == owner] = environment
    if not torch.equal(result.unique(sorted=True), torch.arange(unique.numel())):
        raise ValueError("v78 environment owner remapping changed")
    return result


def greedy_select_prompt_robust_cells(
    code_response: torch.Tensor,
    base_response: torch.Tensor,
    residual: torch.Tensor,
    owners: torch.Tensor,
    *,
    steps: int,
    risk_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float | int]]]:
    if (
        code_response.ndim != 4
        or code_response.shape != base_response.shape
        or residual.shape
        != (code_response.shape[0], code_response.shape[1], code_response.shape[3])
        or owners.shape != (code_response.shape[0],)
        or not 0 < steps < code_response.shape[2]
        or risk_radius < 0
    ):
        raise ValueError("invalid v78 prompt-robust selection inputs")
    code = code_response.float()
    base = base_response.float()
    working = residual.float().clone()
    environments = int(owners.max()) + 1
    if environments < 2 or not torch.equal(
        owners.unique(sorted=True), torch.arange(environments)
    ):
        raise ValueError("v78 requires contiguous multiple prompt environments")
    initial_environment_energy = torch.stack(
        [working[owners == env].square().sum(dim=(0, 2)) for env in range(environments)]
    )
    if bool((initial_environment_energy <= 0).any()):
        raise ValueError("v78 prompt tile has zero initial residual energy")
    sample_weight = initial_environment_energy.reciprocal()[owners]
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
        target = working.unsqueeze(2) + base
        numerator = torch.einsum("nt,ntgr,ntgr->tgr", sample_weight, code, target)
        denominator = torch.einsum("nt,ntgr,ntgr->tgr", sample_weight, code, code)
        fitted = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-30),
            torch.zeros_like(numerator),
        ).clamp_min(0)
        stored_scale = fitted.to(torch.float16)
        delta = stored_scale.float().unsqueeze(0) * code - base
        environment_benefits = []
        for environment in range(environments):
            rows = owners == environment
            benefit = (
                2 * (working[rows].unsqueeze(2) * delta[rows]).sum(dim=(0, 3))
                - delta[rows].square().sum(dim=(0, 3))
            ) / initial_environment_energy[environment].unsqueeze(1)
            environment_benefits.append(benefit)
        benefits = torch.stack(environment_benefits)
        mean_benefit = benefits.mean(dim=0)
        robust_score = mean_benefit - risk_radius * benefits.std(dim=0, unbiased=False)
        robust_score.masked_fill_(selected_mask, -torch.inf)
        best_group = robust_score.argmax(dim=1)
        best_score = robust_score[tile_index, best_group]
        best_scale = stored_scale[tile_index, best_group]
        best_environment_benefit = benefits[:, tile_index, best_group]
        selected[:, step] = best_group.cpu()
        selected_scales[:, step] = best_scale.cpu()
        selected_mask[tile_index, best_group] = True
        chosen_delta = delta[:, tile_index, best_group, :]
        working.sub_(chosen_delta)
        history.append(
            {
                "step": step + 1,
                "minimum_selected_robust_score": float(best_score.min()),
                "median_selected_robust_score": float(best_score.median()),
                "maximum_selected_robust_score": float(best_score.max()),
                "minimum_selected_environment_benefit": float(
                    best_environment_benefit.min()
                ),
                "median_selected_environment_benefit": float(
                    best_environment_benefit.median()
                ),
                "fraction_selected_environment_benefits_positive": float(
                    (best_environment_benefit > 0).float().mean()
                ),
                "minimum_selected_scale": float(best_scale.float().min()),
                "maximum_selected_scale": float(best_scale.float().max()),
            }
        )
    return selected, selected_scales, working, history


def prompt_metrics(
    exact: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    owners: torch.Tensor,
) -> list[dict[str, Any]]:
    records = []
    for environment in owners.unique(sorted=True):
        rows = owners == environment
        base_nmse = v71.tensor_nmse(baseline[rows], exact[rows])
        candidate_nmse = v71.tensor_nmse(candidate[rows], exact[rows])
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
                "baseline_complete_mlp_nmse": base_nmse,
                "candidate_complete_mlp_nmse": candidate_nmse,
                "relative_improvement": (base_nmse - candidate_nmse)
                / max(base_nmse, 1e-30),
                "directional_capacity": v77.vector_geometry(
                    residual, correction
                )["nonnegative_scalar_directional_capacity"],
            }
        )
    return records


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
    parser.add_argument("--v76-preregistration", required=True, type=Path)
    parser.add_argument("--v76-receipt", required=True, type=Path)
    parser.add_argument("--v76-checkpoint", required=True, type=Path)
    parser.add_argument("--v77-receipt", required=True, type=Path)
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
        raise ValueError("v78 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    cache_v1.resource_guard(Path("/workspace"))
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v78 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v78 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg = json.loads(args.preregistration.read_text())
    auth = prereg["authenticated_inputs"]
    parents = {
        "full_context": (args.full_context_receipt, "full_context_receipt_sha256"),
        "v44": (args.v44_receipt, "v44_training_receipt_sha256"),
        "v75": (args.v75_receipt, "v75_receipt_sha256"),
        "v76": (args.v76_receipt, "v76_receipt_sha256"),
        "v77": (args.v77_receipt, "v77_diagnostic_receipt_sha256"),
    }
    parent_json = {name: json.loads(path.read_text()) for name, (path, _) in parents.items()}
    if (
        prereg.get("schema") != "qwen38_27b_prompt_robust_cell_support_preregistration_v78"
        or prereg.get("status") != "frozen_development_screen_before_execution"
        or prereg.get("data_status", {}).get("promotion_or_scientific_claim_from_v78_forbidden")
        is not True
        or any(v71.file_sha256(path) != auth[key] for path, key in parents.values())
        or v71.file_sha256(args.v76_checkpoint) != auth["v76_checkpoint_sha256"]
        or parent_json["full_context"].get("gate", {}).get("passed") is not True
        or parent_json["v44"].get("gate", {}).get("passed") is not True
        or parent_json["v75"].get("gate", {}).get("passed") is not True
        or parent_json["v76"].get("gate", {}).get("passed") is not False
        or parent_json["v77"].get("summary", {}).get("classification")
        != "cell_support_or_direction_shift"
    ):
        raise ValueError("v78 frozen protocol or authenticated parents changed")

    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("v78 Qwen3.8-27B model index changed")
    from transformers import AutoConfig, AutoTokenizer

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
            raise ValueError(f"v78 source tensor changed: {name}")

    context_receipt = parent_json["full_context"]
    context_path = args.full_context_root / "context.pt"
    shard_path = args.full_context_root / "layers" / f"{LAYER}.pt"
    layer_record = context_receipt["layers"][LAYER]
    if (
        v71.file_sha256(context_path) != context_receipt["context"]["sha256"]
        or v71.file_sha256(shard_path) != layer_record["shard"]["sha256"]
    ):
        raise ValueError("v78 full-context artifacts changed")
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    fit_rows = context["calibration_flat_rows"].long()
    v42_validation_rows = context["validation_flat_rows"].long()
    selected_rows = torch.cat((fit_rows, v42_validation_rows))
    fit_owners = environment_owners(torch.div(fit_rows, 64, rounding_mode="floor"))
    v42_validation_owners = environment_owners(
        torch.div(v42_validation_rows, 64, rounding_mode="floor")
    )
    hidden = shard["input"].contiguous()
    position_embeddings, masks = qat.make_context(
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
            position_embeddings=position_embeddings,
            attention_mask=masks[config.layer_types[LAYER]],
            position_ids=context["text_position_ids"].long(),
        )

    exact_output, v42_mlp_input, v42_teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, v42_exact_forward, selected_rows
        )
    )
    if not torch.equal(exact_output, shard["output"]):
        raise ValueError("v78 v42 exact layer replay changed")
    del exact_output, hidden, shard, position_embeddings, masks
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
            raise ValueError(f"v78 CQ3 payload changed: {role}")
    gate = v71.load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = v71.load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = v71.load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
    source_down = state["mlp.down_proj.weight"]
    with torch.no_grad():
        v42_exact_mlp = functional.linear(v42_teacher_down_input.float(), source_down.float())
        v42_baseline_down_input = (
            functional.silu(gate(v42_mlp_input).float()) * up(v42_mlp_input).float()
        ).to(torch.bfloat16)
        v42_baseline_mlp = down(v42_baseline_down_input).float()

    int4_codes, parent_int4_scales = v73.quantize_signed_int4_grouped(
        source_down, group_size=GROUP_SIZE, iterations=4
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
        grouped_v42 = v42_baseline_down_input.float().reshape(
            selected_rows.numel(), INPUT_GROUPS, GROUP_SIZE
        )
        v42_code_response = torch.einsum("trgd,ngd->ntgr", code, grouped_v42)
        v42_base_response = torch.einsum(
            "trgd,ngd->ntgr", base_weight, grouped_v42
        )

    fit_count = fit_rows.numel()
    v42_baseline_tiled = v42_baseline_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    v42_exact_tiled = v42_exact_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    fit_residual = v42_exact_tiled[:fit_count] - v42_baseline_tiled[:fit_count]
    estimator = prereg["estimator"]
    selected_groups, selected_scales, final_fit_residual, history = (
        greedy_select_prompt_robust_cells(
            v42_code_response[:fit_count],
            v42_base_response[:fit_count],
            fit_residual,
            fit_owners,
            steps=int(estimator["greedy_steps_per_output_tile"]),
            risk_radius=float(estimator["risk_radius"]),
        )
    )
    v42_correction = v75.selected_optimal_scale_correction(
        v42_code_response, v42_base_response, selected_groups, selected_scales
    )
    v42_candidate_tiled = v42_baseline_tiled + v42_correction
    v42_candidate_mlp = v42_candidate_tiled.reshape_as(v42_baseline_mlp)
    if not torch.allclose(
        final_fit_residual,
        v42_exact_tiled[:fit_count] - v42_candidate_tiled[:fit_count],
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError("v78 robust residual update changed")

    def relative_improvement(
        baseline_value: torch.Tensor, candidate_value: torch.Tensor, exact_value: torch.Tensor
    ) -> tuple[float, float, float]:
        base_nmse = v71.tensor_nmse(baseline_value, exact_value)
        candidate_nmse = v71.tensor_nmse(candidate_value, exact_value)
        return base_nmse, candidate_nmse, (base_nmse - candidate_nmse) / max(base_nmse, 1e-30)

    fit_metric = relative_improvement(
        v42_baseline_mlp[:fit_count], v42_candidate_mlp[:fit_count], v42_exact_mlp[:fit_count]
    )
    v42_development_metric = relative_improvement(
        v42_baseline_mlp[fit_count:],
        v42_candidate_mlp[fit_count:],
        v42_exact_mlp[fit_count:],
    )
    v42_prompt_records = prompt_metrics(
        v42_exact_mlp[fit_count:],
        v42_baseline_mlp[fit_count:],
        v42_candidate_mlp[fit_count:],
        v42_validation_owners,
    )
    del v42_code_response, v42_base_response, grouped_v42, v42_correction
    del v42_mlp_input, v42_teacher_down_input, v42_baseline_down_input
    gc.collect()

    v76_spec = json.loads(args.v76_preregistration.read_text())
    prompts = list(v76_spec["fresh_prompt_protocol"]["prompts"])
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )
    fresh_token_ids = encoded["input_ids"]
    fresh_attention_mask = encoded["attention_mask"]
    fresh_rows = cache_v1.selected_flat_rows(fresh_attention_mask, range(8), 64)
    fresh_owners = environment_owners(torch.div(fresh_rows, 64, rounding_mode="floor"))
    if (
        tensor_sha256(fresh_token_ids)
        != parent_json["v76"]["fresh_data"]["input_ids_sha256"]
        or tensor_sha256(fresh_rows)
        != parent_json["v76"]["fresh_data"]["selected_rows_sha256"]
    ):
        raise ValueError("v78 v76 development data replay changed")
    checkpoint = torch.load(args.v76_checkpoint, map_location="cpu", weights_only=False)
    fresh_hidden = checkpoint["hidden"].contiguous()
    if (
        checkpoint.get("next_layer") != LAYER
        or tensor_sha256(fresh_hidden)
        != parent_json["v76"]["exact_propagation"]["layer59_input_sha256"]
    ):
        raise ValueError("v78 v76 checkpoint replay changed")
    fresh_position_ids = torch.arange(64, dtype=torch.long).expand(8, -1)
    fresh_position_embeddings, fresh_masks = qat.make_context(
        config, fresh_hidden, fresh_attention_mask, fresh_position_ids
    )

    def fresh_exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            fresh_hidden,
            position_embeddings=fresh_position_embeddings,
            attention_mask=fresh_masks[config.layer_types[LAYER]],
            position_ids=fresh_position_ids,
        )

    fresh_exact_output, fresh_mlp_input, fresh_teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, fresh_exact_forward, fresh_rows
        )
    )
    fresh_exact_layer_replay = (
        tensor_sha256(fresh_exact_output)
        == parent_json["v76"]["exact_propagation"]["layer59_output_sha256"]
    )
    with torch.no_grad():
        fresh_exact_mlp = functional.linear(
            fresh_teacher_down_input.float(), source_down.float()
        )
        fresh_baseline_down_input = (
            functional.silu(gate(fresh_mlp_input).float()) * up(fresh_mlp_input).float()
        ).to(torch.bfloat16)
        fresh_baseline_mlp = down(fresh_baseline_down_input).float()
        grouped_fresh = fresh_baseline_down_input.float().reshape(
            fresh_rows.numel(), INPUT_GROUPS, GROUP_SIZE
        )
        fresh_code_response = torch.einsum("trgd,ngd->ntgr", code, grouped_fresh)
        fresh_base_response = torch.einsum(
            "trgd,ngd->ntgr", base_weight, grouped_fresh
        )
    fresh_correction = v75.selected_optimal_scale_correction(
        fresh_code_response,
        fresh_base_response,
        selected_groups,
        selected_scales,
    )
    fresh_baseline_tiled = fresh_baseline_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    fresh_candidate_tiled = fresh_baseline_tiled + fresh_correction
    fresh_candidate_mlp = fresh_candidate_tiled.reshape_as(fresh_baseline_mlp)
    fresh_metric = relative_improvement(
        fresh_baseline_mlp, fresh_candidate_mlp, fresh_exact_mlp
    )
    fresh_prompt_records = prompt_metrics(
        fresh_exact_mlp, fresh_baseline_mlp, fresh_candidate_mlp, fresh_owners
    )
    fresh_base_tile_error = (
        fresh_exact_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS) - fresh_baseline_tiled
    ).square().sum(dim=(0, 2))
    fresh_candidate_tile_error = (
        fresh_exact_mlp.reshape(-1, OUTPUT_TILES, TILE_ROWS) - fresh_candidate_tiled
    ).square().sum(dim=(0, 2))
    fresh_positive_tile_fraction = float(
        (fresh_candidate_tile_error < fresh_base_tile_error).float().mean()
    )
    fresh_improved_prompt_fraction = sum(
        record["relative_improvement"] > 0 for record in fresh_prompt_records
    ) / len(fresh_prompt_records)
    fresh_median_directional_capacity = float(
        torch.quantile(
            torch.tensor(
                [record["directional_capacity"] for record in fresh_prompt_records],
                dtype=torch.float64,
            ),
            0.5,
        )
    )

    optimized_int4_scales = v75.install_selected_scales(
        parent_int4_scales, selected_groups, selected_scales
    )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    hybrid_path = args.artifact_root / "blk.59.ffn_down.cellmix-cq3-int4-prompt-robust"
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
        raise ValueError("existing v78 hybrid payload changed")
    mask, checked_cq3_codes, checked_cq3_scales, checked_int4_codes, checked_int4_scales = (
        v76.decode_frozen_hybrid(hybrid_payload)
    )
    packed_roundtrip_exact = (
        torch.equal(
            checked_cq3_codes,
            packed_sources["cq3_codes"].reshape(-1, 1, GROUP_SIZE),
        )
        and torch.equal(
            checked_cq3_scales,
            packed_sources["cq3_scales"].reshape(-1, 1),
        )
        and torch.equal(
            checked_int4_codes,
            packed_sources["int4_codes"].reshape(-1, 1, GROUP_SIZE),
        )
        and torch.equal(
            checked_int4_scales,
            packed_sources["int4_scales"].reshape(-1, 1),
        )
        and int(mask.sum()) == selected_groups.numel()
    )
    full_mlp_bytes = 2 * CQ3_PAYLOAD_BYTES + storage["total_bytes"]
    storage_increase = (full_mlp_bytes - 3 * CQ3_PAYLOAD_BYTES) / (
        3 * CQ3_PAYLOAD_BYTES
    )

    gate_spec = prereg["development_gate"]
    checks = {
        "all_parent_artifacts_authenticated": True,
        "v42_exact_layer_replay": True,
        "v76_exact_layer_replay": fresh_exact_layer_replay,
        "fit_uses_only_v42_calibration_prompts": True,
        "v42_validation_used_for_fit_or_selection": False,
        "v76_prompts_used_for_fit_or_selection": False,
        "all_values_finite": bool(
            torch.isfinite(selected_scales.float()).all()
            and torch.isfinite(fresh_candidate_mlp).all()
            and torch.isfinite(v42_candidate_mlp).all()
        ),
        "required_selected_cells_met": selected_groups.numel()
        == int(gate_spec["required_selected_cells"]),
        "minimum_fit_improvement_met": fit_metric[2]
        >= float(gate_spec["minimum_relative_fit_complete_mlp_nmse_improvement"]),
        "minimum_v42_validation_improvement_met": v42_development_metric[2]
        >= float(
            gate_spec["minimum_relative_v42_validation_complete_mlp_nmse_improvement"]
        ),
        "minimum_v76_development_improvement_met": fresh_metric[2]
        >= float(
            gate_spec["minimum_relative_v76_development_complete_mlp_nmse_improvement"]
        ),
        "minimum_absolute_v76_gain_over_v75_met": fresh_metric[2]
        - parent_json["v76"]["metrics"]["relative_fresh_complete_mlp_nmse_improvement"]
        >= float(gate_spec["minimum_absolute_v76_improvement_over_v75"]),
        "minimum_v76_positive_output_tile_fraction_met": fresh_positive_tile_fraction
        >= float(gate_spec["minimum_v76_positive_output_tile_fraction"]),
        "minimum_v76_improved_prompt_fraction_met": fresh_improved_prompt_fraction
        >= float(gate_spec["minimum_v76_improved_prompt_fraction"]),
        "minimum_v76_median_prompt_directional_capacity_met": (
            fresh_median_directional_capacity
            >= float(gate_spec["minimum_v76_median_prompt_directional_capacity"])
        ),
        "packed_roundtrip_exact": packed_roundtrip_exact,
        "canonical_payload_bytes_exact": storage["total_bytes"]
        == int(prereg["canonical_layout"]["hybrid_down_payload_bytes"]),
        "maximum_full_mlp_payload_increase_met": storage_increase
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
            key: v71.file_sha256(path) for key, (path, _) in parents.items()
        }
        | {
            "v76_checkpoint_sha256": v71.file_sha256(args.v76_checkpoint),
            "source_tensor_sha256": auth["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
        },
        "estimator": {
            "method": "prompt-normalized conditional FP16 scale fit plus mean-minus-one-population-standard-deviation marginal support score",
            "risk_radius": float(estimator["risk_radius"]),
            "fit_environments": int(fit_owners.max()) + 1,
            "fit_rows": fit_rows.numel(),
            "steps_per_output_tile": selected_groups.shape[1],
            "selected_cells": selected_groups.numel(),
            "selected_groups_by_output_tile": selected_groups.tolist(),
            "selected_groups_sha256": v74.tensor_bytes_sha256(selected_groups),
            "selected_scales_sha256": v74.tensor_bytes_sha256(selected_scales),
            "minimum_selected_scale": float(selected_scales.float().min()),
            "maximum_selected_scale": float(selected_scales.float().max()),
            "zero_selected_scales": int((selected_scales == 0).sum()),
            "history": history,
            "hyperparameter_sweep": False,
        },
        "metrics": {
            "fit": {
                "baseline_complete_mlp_nmse": fit_metric[0],
                "candidate_complete_mlp_nmse": fit_metric[1],
                "relative_improvement": fit_metric[2],
            },
            "v42_validation_development": {
                "baseline_complete_mlp_nmse": v42_development_metric[0],
                "candidate_complete_mlp_nmse": v42_development_metric[1],
                "relative_improvement": v42_development_metric[2],
                "per_prompt": v42_prompt_records,
            },
            "v76_development": {
                "baseline_complete_mlp_nmse": fresh_metric[0],
                "candidate_complete_mlp_nmse": fresh_metric[1],
                "relative_improvement": fresh_metric[2],
                "v75_relative_improvement_reference": parent_json["v76"]["metrics"][
                    "relative_fresh_complete_mlp_nmse_improvement"
                ],
                "absolute_improvement_over_v75": fresh_metric[2]
                - parent_json["v76"]["metrics"][
                    "relative_fresh_complete_mlp_nmse_improvement"
                ],
                "positive_output_tile_fraction": fresh_positive_tile_fraction,
                "improved_prompt_fraction": fresh_improved_prompt_fraction,
                "median_prompt_directional_capacity": fresh_median_directional_capacity,
                "per_prompt": fresh_prompt_records,
            },
        },
        "storage": {
            **storage,
            "hybrid_payload_sha256": v71.file_sha256(hybrid_path),
            "baseline_three_cq3_mlp_payload_bytes": 3 * CQ3_PAYLOAD_BYTES,
            "hybrid_three_projection_mlp_payload_bytes": full_mlp_bytes,
            "relative_full_mlp_payload_increase": storage_increase,
            "additional_runtime_bytes_vs_v75": 0,
            "additional_runtime_operations_vs_v75": 0,
            "additional_runtime_passes_vs_v75": 0,
            "high_precision_weight_materialization_at_runtime": False,
        },
        "data_status": prereg["data_status"],
        "checks": checks,
        "gate": {
            "passed": passed,
            "decision": (
                "freeze_prompt_robust_support_for_second_new_prompt_replication"
                if passed
                else "reject_mean_minus_std_prompt_robust_cell_support"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    v76.atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_prompt_robust_cell_support_marker_v78",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": passed,
        "terminal": True,
    }
    v76.atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
