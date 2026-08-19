#!/usr/bin/env python3
"""Replicate the frozen v75 layer-59 cell payload on untouched prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
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
import run_qwen38_27b_calibration_optimal_cell_scale_v75 as v75
import run_qwen38_27b_greedy_input_cell_mixed_down_v74 as v74
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

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
TOTAL_CELLS = v74.TOTAL_CELLS
SCHEMA = "qwen38_27b_optimal_cell_scale_fresh_replication_receipt_v76"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def decode_frozen_hybrid(
    payload: bytes,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(payload) != 26_307_264:
        raise ValueError("v76 frozen hybrid payload size changed")
    header = v74.HEADER.unpack_from(payload)
    expected_header = (
        v74.MAGIC.ljust(16, b"\0"),
        HIDDEN,
        INTERMEDIATE,
        TILE_ROWS,
        GROUP_SIZE,
        INPUT_GROUPS,
        10_240,
        TOTAL_CELLS,
        10_880,
        20_889_600,
        5_406_720,
        v74.HEADER.size,
        1,
    )
    if header != expected_header:
        raise ValueError(f"v76 frozen hybrid header changed: {header}")
    mode_start = v74.HEADER.size
    cq3_start = mode_start + header[8]
    int4_start = cq3_start + header[9]
    mode = np.unpackbits(
        np.frombuffer(payload[mode_start:cq3_start], dtype=np.uint8),
        bitorder="little",
    )[:TOTAL_CELLS]
    mask = torch.from_numpy(mode.copy()).bool().reshape(OUTPUT_TILES, INPUT_GROUPS)
    if int(mask.sum()) != header[6] or not bool((mask.sum(dim=1) == 16).all()):
        raise ValueError("v76 frozen selected-cell mask changed")
    cq3_codes, cq3_scales = v74.unpack_cq3_group_stream(
        payload[cq3_start:int4_start],
        rows=(TOTAL_CELLS - header[6]) * TILE_ROWS,
        groups=1,
    )
    int4_codes, int4_scales = v74.unpack_int4_group_stream(
        payload[int4_start:], rows=header[6] * TILE_ROWS, groups=1
    )
    return mask, cq3_codes, cq3_scales, int4_codes, int4_scales


def frozen_hybrid_correction(
    grouped_input: torch.Tensor,
    baseline_codes: torch.Tensor,
    baseline_scales: torch.Tensor,
    mask: torch.Tensor,
    packed_int4_codes: torch.Tensor,
    packed_int4_scales: torch.Tensor,
    *,
    tile_chunk: int = 32,
) -> torch.Tensor:
    tile_rows = baseline_codes.shape[1]
    groups = baseline_codes.shape[2]
    group_size = baseline_codes.shape[3]
    tiles = baseline_codes.shape[0]
    selected_per_tile = int(mask.sum(dim=1)[0])
    if (
        grouped_input.ndim != 3
        or grouped_input.shape[1:] != (groups, group_size)
        or baseline_scales.shape != (tiles, tile_rows, groups)
        or mask.shape != (tiles, groups)
        or not bool((mask.sum(dim=1) == selected_per_tile).all())
        or packed_int4_codes.shape
        != (tiles * selected_per_tile * tile_rows, 1, group_size)
        or packed_int4_scales.shape
        != (tiles * selected_per_tile * tile_rows, 1)
    ):
        raise ValueError("v76 frozen hybrid correction shape changed")
    selected_groups = mask.nonzero(as_tuple=False)[:, 1].reshape(
        tiles, selected_per_tile
    )
    int4_codes = packed_int4_codes.reshape(
        tiles, selected_per_tile, tile_rows, group_size
    )
    int4_scales = packed_int4_scales.reshape(
        tiles, selected_per_tile, tile_rows
    )
    correction = torch.empty(
        grouped_input.shape[0], tiles, tile_rows, dtype=torch.float32
    )
    baseline_cells = baseline_codes.permute(0, 2, 1, 3)
    baseline_cell_scales = baseline_scales.permute(0, 2, 1)
    for start in range(0, tiles, tile_chunk):
        stop = min(tiles, start + tile_chunk)
        cell_groups = selected_groups[start:stop]
        code_index = cell_groups[:, :, None, None].expand(
            -1, -1, tile_rows, group_size
        )
        scale_index = cell_groups[:, :, None].expand(-1, -1, tile_rows)
        base_weight = torch.gather(
            baseline_cells[start:stop], 1, code_index
        ).float() * torch.gather(
            baseline_cell_scales[start:stop], 1, scale_index
        ).float().unsqueeze(-1)
        int4_weight = int4_codes[start:stop].float() * int4_scales[
            start:stop
        ].float().unsqueeze(-1)
        selected_input = grouped_input[:, cell_groups, :].float()
        correction[:, start:stop] = torch.einsum(
            "csrd,ncsd->ncr", int4_weight - base_weight, selected_input
        )
    return correction


def checkpoint_identity(
    prereg_sha: str, index_sha: str, prompt_sha: str, token_ids: torch.Tensor
) -> dict[str, Any]:
    return {
        "schema": "qwen38_27b_fresh_layer59_checkpoint_v76",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": prereg_sha,
        "index_sha256": index_sha,
        "prompt_sha256": prompt_sha,
        "input_ids_sha256": tensor_sha256(token_ids),
    }


def save_checkpoint(
    path: Path,
    identity: dict[str, Any],
    hidden: torch.Tensor,
    next_layer: int,
) -> None:
    payload = {
        **identity,
        "next_layer": next_layer,
        "hidden_sha256": tensor_sha256(hidden),
        "hidden": hidden.detach().cpu().contiguous(),
    }
    cache_v1.atomic_torch_save(payload, path)


def load_or_initialize_hidden(
    path: Path,
    identity: dict[str, Any],
    initial_hidden: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if not path.exists():
        save_checkpoint(path, identity, initial_hidden, 0)
        return initial_hidden, 0
    value = torch.load(path, map_location="cpu", weights_only=False)
    hidden = value.pop("hidden")
    expected = {
        **identity,
        "next_layer": value["next_layer"],
        "hidden_sha256": value["hidden_sha256"],
    }
    if (
        value != expected
        or not 0 <= int(value["next_layer"]) <= LAYER
        or hidden.dtype != torch.bfloat16
        or hidden.shape[1:] != (64, HIDDEN)
        or tensor_sha256(hidden) != value["hidden_sha256"]
        or not bool(torch.isfinite(hidden).all())
    ):
        raise ValueError("v76 resumable hidden-state checkpoint changed")
    return hidden.contiguous(), int(value["next_layer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--v42-receipt", required=True, type=Path)
    parser.add_argument("--v42-context", required=True, type=Path)
    parser.add_argument("--v44-receipt", required=True, type=Path)
    parser.add_argument("--v75-receipt", required=True, type=Path)
    parser.add_argument("--v75-artifact", required=True, type=Path)
    parser.add_argument(
        "--v44-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads != 4:
        raise ValueError("v76 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    cache_v1.resource_guard(Path("/workspace"))
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v76 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v76 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = v71.file_sha256(args.preregistration)
    v42_sha = v71.file_sha256(args.v42_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v75_sha = v71.file_sha256(args.v75_receipt)
    v42 = json.loads(args.v42_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    parent = json.loads(args.v75_receipt.read_text())
    auth = prereg["authenticated_inputs"]
    frozen = prereg["frozen_candidate"]
    protocol = prereg["fresh_prompt_protocol"]
    prompts = list(protocol["prompts"])
    if (
        prereg.get("schema")
        != "qwen38_27b_optimal_cell_scale_fresh_replication_preregistration_v76"
        or prereg.get("status")
        != "frozen_before_fresh_prompt_tokenization_or_layer_propagation"
        or prereg.get("scope", {}).get("layer") != LAYER
        or len(prompts) != 8
        or len(set(prompts)) != 8
        or protocol.get("benchmark_items_used") is not False
        or protocol.get("fresh_activations_may_not_change_artifact_or_gate") is not True
        or auth.get("full_context_receipt_sha256") != v42_sha
        or auth.get("v44_training_receipt_sha256") != v44_sha
        or frozen.get("v75_receipt_sha256") != v75_sha
        or v42.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
        or parent.get("gate", {}).get("decision")
        != "freeze_optimal_cell_scales_for_new_prompt_replication"
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("v76 frozen protocol or authenticated parents changed")

    if (
        v71.file_sha256(args.v42_context) != auth["prior_context_sha256"]
        or v42.get("context", {}).get("sha256") != auth["prior_context_sha256"]
    ):
        raise ValueError("v76 prior prompt context changed")
    prior_context = torch.load(args.v42_context, map_location="cpu", weights_only=False)
    prior_prompts = set(prior_context["calibration_texts"]) | set(
        prior_context["validation_texts"]
    )
    if prior_prompts & set(prompts):
        raise ValueError("v76 fresh prompts overlap prior development prompts")
    del prior_context

    if (
        args.v75_artifact.stat().st_size != frozen["hybrid_payload_bytes"]
        or v71.file_sha256(args.v75_artifact) != frozen["hybrid_payload_sha256"]
        or parent["storage"]["hybrid_payload_sha256"]
        != frozen["hybrid_payload_sha256"]
    ):
        raise ValueError("v76 frozen v75 payload changed")
    hybrid = args.v75_artifact.read_bytes()
    mask, packed_cq3_codes, packed_cq3_scales, int4_codes, int4_scales = (
        decode_frozen_hybrid(hybrid)
    )

    index_path = args.model_dir / "model.safetensors.index.json"
    index_sha = v71.file_sha256(index_path)
    if index_sha != direct_v43.INDEX_SHA256:
        raise ValueError("Qwen3.8-27B model index changed")
    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if (
        download.get("model_id") != MODEL_ID
        or download.get("revision") != MODEL_REVISION
        or download.get("index_sha256") != index_sha
    ):
        raise ValueError("Qwen3.8-27B download identity changed")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != HIDDEN
        or config.intermediate_size != INTERMEDIATE
        or config.layer_types[LAYER] != "full_attention"
    ):
        raise ValueError("Qwen3.8-27B text architecture changed")
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
    if token_ids.shape != (8, 64) or attention_mask.shape != (8, 64):
        raise ValueError("v76 frozen tokenization shape changed")
    selected_rows = cache_v1.selected_flat_rows(attention_mask, range(8), 64)
    owners = torch.div(selected_rows, 64, rounding_mode="floor")
    rows_per_prompt = [int((owners == index).sum()) for index in range(8)]
    if (
        selected_rows.numel() != 64
        or min(rows_per_prompt) < int(protocol["minimum_selected_rows_per_prompt"])
    ):
        raise ValueError(f"v76 selected-row prompt coverage changed: {rows_per_prompt}")
    prompt_sha = canonical_sha256(prompts)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    initial_hidden = cache_v1.embedding_rows(args.model_dir, weight_map, token_ids)
    if initial_hidden.dtype != torch.bfloat16 or initial_hidden.shape != (8, 64, HIDDEN):
        raise ValueError("v76 exact embedding state changed")
    position_ids = torch.arange(64, dtype=torch.long).expand(8, -1)
    position_embeddings, masks = qat.make_context(
        config, initial_hidden, attention_mask, position_ids
    )
    identity = checkpoint_identity(prereg_sha, index_sha, prompt_sha, token_ids)
    hidden, next_layer = load_or_initialize_hidden(
        args.checkpoint, identity, initial_hidden
    )
    del initial_hidden

    propagated: list[dict[str, Any]] = []
    started = time.monotonic()
    for layer_index in range(next_layer, LAYER):
        layer_started = time.monotonic()
        state, parameters = cache_v1.layer_state(args.model_dir, weight_map, layer_index)
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"v76 layer {layer_index} state mapping changed")
        layer.eval()
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[config.layer_types[layer_index]],
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            ).contiguous()
        if hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError(f"v76 non-finite state after layer {layer_index}")
        save_checkpoint(args.checkpoint, identity, hidden, layer_index + 1)
        propagated.append(
            {
                "layer": layer_index,
                "type": config.layer_types[layer_index],
                "parameters": parameters,
                "elapsed_seconds": time.monotonic() - layer_started,
            }
        )
        del layer, state
        gc.collect()

    state, parameters = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in auth["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v76 source tensor changed: {name}")
    exact_layer = block_parent.instantiate_layer(config, LAYER, state)

    def exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[config.layer_types[LAYER]],
            position_ids=position_ids,
        )

    exact_output, mlp_input, teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, exact_forward, selected_rows
        )
    )
    replay_output = exact_forward()
    exact_layer_replay = torch.equal(exact_output, replay_output)
    layer_input_sha = tensor_sha256(hidden)
    layer_output_sha = tensor_sha256(exact_output)
    del exact_layer, exact_output, replay_output, hidden
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
            raise ValueError(f"v76 trained {role} CQ3 payload changed")
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
    del gate, up, source_down
    gc.collect()

    baseline_codes = down.codes.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE
    ).contiguous()
    baseline_scales = down.scales.reshape(
        OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS
    ).contiguous()
    baseline_cells = baseline_codes.permute(0, 2, 1, 3)[~mask].contiguous()
    baseline_cell_scales = baseline_scales.permute(0, 2, 1)[~mask].contiguous()
    packed_cq3_roundtrip_exact = torch.equal(
        packed_cq3_codes,
        baseline_cells.reshape(-1, 1, GROUP_SIZE),
    ) and torch.equal(
        packed_cq3_scales,
        baseline_cell_scales.reshape(-1, 1),
    )
    grouped_input = baseline_down_input.float().reshape(64, INPUT_GROUPS, GROUP_SIZE)
    correction = frozen_hybrid_correction(
        grouped_input,
        baseline_codes,
        baseline_scales,
        mask,
        int4_codes,
        int4_scales,
    )
    baseline_tiled = baseline_mlp_output.reshape(64, OUTPUT_TILES, TILE_ROWS)
    candidate_tiled = baseline_tiled + correction
    candidate_mlp_output = candidate_tiled.reshape_as(baseline_mlp_output)
    exact_tiled = exact_mlp_output.reshape(64, OUTPUT_TILES, TILE_ROWS)

    baseline_nmse = v71.tensor_nmse(baseline_mlp_output, exact_mlp_output)
    candidate_nmse = v71.tensor_nmse(candidate_mlp_output, exact_mlp_output)
    improvement = (baseline_nmse - candidate_nmse) / max(baseline_nmse, 1e-30)
    v75_development_improvement = parent["metrics"][
        "relative_development_complete_mlp_nmse_improvement"
    ]
    replication_ratio = improvement / max(v75_development_improvement, 1e-30)
    base_tile_error = (exact_tiled - baseline_tiled).square().sum(dim=(0, 2))
    candidate_tile_error = (exact_tiled - candidate_tiled).square().sum(dim=(0, 2))
    positive_tile_fraction = float(
        (candidate_tile_error < base_tile_error).float().mean()
    )
    prompt_metrics: list[dict[str, Any]] = []
    for prompt_index in range(8):
        prompt_rows = owners == prompt_index
        prompt_base = v71.tensor_nmse(
            baseline_mlp_output[prompt_rows], exact_mlp_output[prompt_rows]
        )
        prompt_candidate = v71.tensor_nmse(
            candidate_mlp_output[prompt_rows], exact_mlp_output[prompt_rows]
        )
        prompt_metrics.append(
            {
                "prompt_index": prompt_index,
                "selected_rows": int(prompt_rows.sum()),
                "baseline_complete_mlp_nmse": prompt_base,
                "candidate_complete_mlp_nmse": prompt_candidate,
                "relative_improvement": (prompt_base - prompt_candidate)
                / max(prompt_base, 1e-30),
            }
        )
    improved_prompt_fraction = sum(
        record["relative_improvement"] > 0 for record in prompt_metrics
    ) / len(prompt_metrics)

    gate_spec = prereg["replication_gate"]
    checks = {
        "all_parent_artifacts_authenticated": True,
        "fresh_prompts_disjoint_from_prior_development_prompts": True,
        "fresh_activations_did_not_change_artifact_or_gate": True,
        "required_fresh_rows_met": selected_rows.numel()
        == int(gate_spec["required_fresh_rows"]),
        "minimum_rows_per_prompt_met": min(rows_per_prompt)
        >= int(gate_spec["minimum_selected_rows_per_prompt"]),
        "exact_layer_replay": exact_layer_replay,
        "packed_cq3_roundtrip_exact": packed_cq3_roundtrip_exact,
        "frozen_payload_sha256_exact": v71.file_sha256(args.v75_artifact)
        == frozen["hybrid_payload_sha256"],
        "all_values_finite": bool(
            torch.isfinite(candidate_mlp_output).all()
            and torch.isfinite(exact_mlp_output).all()
            and torch.isfinite(correction).all()
        ),
        "minimum_fresh_improvement_met": improvement
        >= float(gate_spec["minimum_relative_fresh_complete_mlp_nmse_improvement"]),
        "minimum_replication_ratio_met": replication_ratio
        >= float(gate_spec["minimum_fresh_to_v75_development_improvement_ratio"]),
        "minimum_positive_output_tile_fraction_met": positive_tile_fraction
        >= float(gate_spec["minimum_fresh_positive_output_tile_fraction"]),
        "minimum_improved_prompt_fraction_met": improved_prompt_fraction
        >= float(gate_spec["minimum_improved_prompt_fraction"]),
    }
    passed = all(checks.values())
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": prereg_sha,
        "authenticated_inputs": {
            "v42_receipt_sha256": v42_sha,
            "prior_context_sha256": v71.file_sha256(args.v42_context),
            "v44_receipt_sha256": v44_sha,
            "v75_receipt_sha256": v75_sha,
            "v75_hybrid_payload_sha256": v71.file_sha256(args.v75_artifact),
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
            "source_tensor_sha256": auth["source_tensor_sha256"],
        },
        "fresh_data": {
            "prompt_count": len(prompts),
            "prompt_sha256": prompt_sha,
            "input_ids_sha256": tensor_sha256(token_ids),
            "attention_mask_sha256": tensor_sha256(attention_mask),
            "selected_rows_sha256": tensor_sha256(selected_rows),
            "selected_rows": selected_rows.numel(),
            "rows_per_prompt": rows_per_prompt,
            "benchmark_items_used": False,
            "prior_prompt_overlap_count": 0,
            "artifact_fit_or_selection_operations_on_fresh_data": 0,
        },
        "exact_propagation": {
            "resumed_at_layer": next_layer,
            "newly_propagated_layers": propagated,
            "layer59_parameters": parameters,
            "layer59_input_sha256": layer_input_sha,
            "layer59_output_sha256": layer_output_sha,
            "checkpoint": {
                "path": str(args.checkpoint),
                "bytes": args.checkpoint.stat().st_size,
                "sha256": v71.file_sha256(args.checkpoint),
                "retained_states": 1,
                "next_layer": LAYER,
            },
            "elapsed_seconds": time.monotonic() - started,
        },
        "metrics": {
            "baseline_fresh_complete_mlp_nmse": baseline_nmse,
            "candidate_fresh_complete_mlp_nmse": candidate_nmse,
            "relative_fresh_complete_mlp_nmse_improvement": improvement,
            "v75_development_improvement_reference": v75_development_improvement,
            "fresh_to_v75_development_improvement_ratio": replication_ratio,
            "fresh_positive_output_tile_fraction": positive_tile_fraction,
            "improved_prompt_fraction": improved_prompt_fraction,
            "per_prompt": prompt_metrics,
        },
        "storage": {
            **prereg["canonical_layout"],
            "selected_cells": int(mask.sum()),
            "additional_runtime_bytes_vs_v75": 0,
            "additional_runtime_operations_vs_v75": 0,
            "additional_runtime_passes_vs_v75": 0,
        },
        "checks": checks,
        "gate": {
            "passed": passed,
            "decision": (
                "replicate_fixed_cell_scale_algebra_and_preregister_representative_layers"
                if passed
                else "reject_v75_as_nonreplicating_development_result"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_optimal_cell_scale_fresh_replication_marker_v76",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": passed,
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
