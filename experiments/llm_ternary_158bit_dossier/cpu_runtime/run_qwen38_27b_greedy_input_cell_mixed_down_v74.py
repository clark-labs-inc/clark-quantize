#!/usr/bin/env python3
"""Greedily route x8-by-128 down-projection cells between CQ3 and INT4."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
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
TILE_ROWS = 8
OUTPUT_TILES = HIDDEN // TILE_ROWS
GROUP_SIZE = 128
INPUT_GROUPS = INTERMEDIATE // GROUP_SIZE
TOTAL_CELLS = OUTPUT_TILES * INPUT_GROUPS
CQ3_BYTES_PER_ROW_GROUP = 34
INT4_BYTES_PER_ROW_GROUP = 66
CQ3_BYTES_PER_CELL = TILE_ROWS * CQ3_BYTES_PER_ROW_GROUP
INT4_BYTES_PER_CELL = TILE_ROWS * INT4_BYTES_PER_ROW_GROUP
HEADER = struct.Struct("<16s12I")
MAGIC = b"CQ3Q4CELLV1"
SCHEMA = "qwen38_27b_greedy_input_cell_mixed_down_receipt_v74"


def pack_cq3_group_stream(codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    if (
        codes.ndim != 3
        or codes.shape[-1] != GROUP_SIZE
        or scales.shape != codes.shape[:2]
        or codes.dtype != torch.int8
        or scales.dtype != torch.float16
        or bool(((codes < -1) | (codes > 1)).any())
    ):
        raise ValueError("invalid standalone CQ3 group stream")
    trits = (codes.detach().cpu().numpy().astype(np.int16) + 1).astype(np.uint8)
    packed = np.zeros((*trits.shape[:2], GROUP_SIZE // 4), dtype=np.uint8)
    for quarter in range(4):
        packed |= trits[..., quarter * 32 : (quarter + 1) * 32] << (2 * quarter)
    scale_bytes = (
        scales.detach()
        .cpu()
        .numpy()
        .astype("<f2", copy=False)
        .view(np.uint8)
        .reshape(*scales.shape, 2)
    )
    return np.concatenate((packed, scale_bytes), axis=-1).tobytes(order="C")


def unpack_cq3_group_stream(
    payload: bytes, *, rows: int, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = rows * groups * CQ3_BYTES_PER_ROW_GROUP
    if rows <= 0 or groups <= 0 or len(payload) != expected:
        raise ValueError("standalone CQ3 group payload size changed")
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(
        rows, groups, CQ3_BYTES_PER_ROW_GROUP
    )
    packed = raw[..., : GROUP_SIZE // 4]
    trits = np.empty((rows, groups, GROUP_SIZE), dtype=np.int8)
    for quarter in range(4):
        digits = ((packed >> (2 * quarter)) & 3).astype(np.int8)
        if bool((digits > 2).any()):
            raise ValueError("standalone CQ3 stream has a reserved code")
        trits[..., quarter * 32 : (quarter + 1) * 32] = digits - 1
    scale_bytes = raw[..., GROUP_SIZE // 4 :].copy()
    scales = scale_bytes.reshape(rows, groups, 2).view("<f2").reshape(rows, groups)
    return torch.from_numpy(trits.copy()), torch.from_numpy(scales.copy())


def pack_int4_group_stream(codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    if (
        codes.ndim != 3
        or codes.shape[-1] != GROUP_SIZE
        or scales.shape != codes.shape[:2]
        or codes.dtype != torch.int8
        or scales.dtype != torch.float16
        or bool(((codes < -8) | (codes > 7)).any())
    ):
        raise ValueError("invalid standalone INT4 group stream")
    values = codes.detach().cpu().numpy().astype(np.int8, copy=False)
    low = values[..., 0::2].astype(np.int16) & 0xF
    high = values[..., 1::2].astype(np.int16) & 0xF
    packed = (low | (high << 4)).astype(np.uint8)
    scale_bytes = (
        scales.detach()
        .cpu()
        .numpy()
        .astype("<f2", copy=False)
        .view(np.uint8)
        .reshape(*scales.shape, 2)
    )
    return np.concatenate((packed, scale_bytes), axis=-1).tobytes(order="C")


def unpack_int4_group_stream(
    payload: bytes, *, rows: int, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = rows * groups * INT4_BYTES_PER_ROW_GROUP
    if rows <= 0 or groups <= 0 or len(payload) != expected:
        raise ValueError("standalone INT4 group payload size changed")
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(
        rows, groups, INT4_BYTES_PER_ROW_GROUP
    )
    packed = raw[..., : GROUP_SIZE // 2]
    low = (packed & 0xF).astype(np.int8)
    high = ((packed >> 4) & 0xF).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    codes = np.empty((rows, groups, GROUP_SIZE), dtype=np.int8)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    scale_bytes = raw[..., GROUP_SIZE // 2 :].copy()
    scales = scale_bytes.reshape(rows, groups, 2).view("<f2").reshape(rows, groups)
    return torch.from_numpy(codes.copy()), torch.from_numpy(scales.copy())


def greedy_select_cells(
    delta: torch.Tensor, residual: torch.Tensor, *, steps: int
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float | int]]]:
    if (
        delta.ndim != 4
        or residual.shape != (delta.shape[0], delta.shape[1], delta.shape[3])
        or not 0 < steps < delta.shape[2]
    ):
        raise ValueError("invalid greedy cell-selection inputs")
    working = residual.float().clone()
    energy = delta.float().square().sum(dim=(0, 3))
    selected_mask = torch.zeros(
        delta.shape[1], delta.shape[2], dtype=torch.bool, device=delta.device
    )
    selected = torch.empty(delta.shape[1], steps, dtype=torch.long)
    tile_index = torch.arange(delta.shape[1], device=delta.device)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        correlation = torch.einsum("ntr,ntgr->tg", working, delta.float())
        benefit = 2 * correlation - energy
        benefit.masked_fill_(selected_mask, -torch.inf)
        best_group = benefit.argmax(dim=1)
        best_benefit = benefit[tile_index, best_group]
        selected[:, step] = best_group.cpu()
        selected_mask[tile_index, best_group] = True
        working.sub_(delta[:, tile_index, best_group, :].float())
        history.append(
            {
                "step": step + 1,
                "minimum_selected_marginal_benefit": float(best_benefit.min()),
                "median_selected_marginal_benefit": float(best_benefit.median()),
                "maximum_selected_marginal_benefit": float(best_benefit.max()),
            }
        )
    return selected, working, history


def selected_cell_correction(delta: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    if selected.shape[0] != delta.shape[1]:
        raise ValueError("selected cell tensor no longer matches output tiles")
    result = torch.zeros(
        delta.shape[0], delta.shape[1], delta.shape[3], dtype=torch.float32
    )
    tile_index = torch.arange(delta.shape[1])
    for step in range(selected.shape[1]):
        result.add_(delta[:, tile_index, selected[:, step], :].float())
    return result


def selection_mask(selected: torch.Tensor, *, groups: int) -> torch.Tensor:
    mask = torch.zeros(selected.shape[0], groups, dtype=torch.bool)
    tile_index = torch.arange(selected.shape[0]).unsqueeze(1).expand_as(selected)
    mask[tile_index, selected] = True
    if int(mask.sum()) != selected.numel():
        raise ValueError("greedy cell selection contains duplicates")
    return mask


def build_hybrid_payload(
    cq3_codes: torch.Tensor,
    cq3_scales: torch.Tensor,
    int4_codes: torch.Tensor,
    int4_scales: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[bytes, dict[str, int], dict[str, torch.Tensor]]:
    mask = selection_mask(selected, groups=INPUT_GROUPS)
    flat_mask = mask.numpy().reshape(-1)
    mode = np.packbits(flat_mask, bitorder="little").tobytes()

    cq3_code_cells = (
        cq3_codes.reshape(OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE)
        .permute(0, 2, 1, 3)[~mask]
        .contiguous()
    )
    cq3_scale_cells = (
        cq3_scales.reshape(OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS)
        .permute(0, 2, 1)[~mask]
        .contiguous()
    )
    int4_code_cells = (
        int4_codes.reshape(OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE)
        .permute(0, 2, 1, 3)[mask]
        .contiguous()
    )
    int4_scale_cells = (
        int4_scales.reshape(OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS)
        .permute(0, 2, 1)[mask]
        .contiguous()
    )
    cq3 = pack_cq3_group_stream(
        cq3_code_cells.reshape(-1, 1, GROUP_SIZE),
        cq3_scale_cells.reshape(-1, 1),
    )
    int4 = pack_int4_group_stream(
        int4_code_cells.reshape(-1, 1, GROUP_SIZE),
        int4_scale_cells.reshape(-1, 1),
    )
    header = HEADER.pack(
        MAGIC,
        HIDDEN,
        INTERMEDIATE,
        TILE_ROWS,
        GROUP_SIZE,
        INPUT_GROUPS,
        selected.numel(),
        TOTAL_CELLS,
        len(mode),
        len(cq3),
        len(int4),
        HEADER.size,
        1,
    )
    payload = header + mode + cq3 + int4
    return (
        payload,
        {
            "header_bytes": len(header),
            "cell_mode_bitmap_bytes": len(mode),
            "unselected_cq3_bytes": len(cq3),
            "selected_int4_bytes": len(int4),
            "total_bytes": len(payload),
        },
        {
            "cq3_codes": cq3_code_cells,
            "cq3_scales": cq3_scale_cells,
            "int4_codes": int4_code_cells,
            "int4_scales": int4_scale_cells,
        },
    )


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


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
    parser.add_argument("--v73-receipt", required=True, type=Path)
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
        raise ValueError("v74 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v74 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v74 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = v71.file_sha256(args.preregistration)
    full_context_sha = v71.file_sha256(args.full_context_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v73_sha = v71.file_sha256(args.v73_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    prior = json.loads(args.v73_receipt.read_text())
    authenticated = prereg["authenticated_inputs"]
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_greedy_input_cell_mixed_down_preregistration_v74"
        or prereg.get("status") != "frozen_before_layer59_validation_open"
        or prereg.get("scope", {}).get("layer") != LAYER
        or authenticated.get("full_context_receipt_sha256") != full_context_sha
        or authenticated.get("v44_training_receipt_sha256") != v44_sha
        or authenticated.get("v73_negative_receipt_sha256") != v73_sha
        or context_receipt.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
        or prior.get("gate", {}).get("passed") is not False
        or prior.get("gate", {}).get("decision")
        != "reject_fixed_budget_output_tile_mixed_precision_at_12_5_percent"
        or prior.get("metrics", {}).get(
            "relative_confirmation_complete_mlp_nmse_improvement", 0
        )
        <= 0.05
    ):
        raise ValueError("v74 frozen protocol or authenticated parents changed")

    cell_spec = prereg["cell_selection"]
    quantizer = prereg["int4_quantizer"]
    layout = prereg["canonical_layout"]
    if (
        int(cell_spec["output_tile_rows"]) != TILE_ROWS
        or int(cell_spec["input_group_columns"]) != GROUP_SIZE
        or int(cell_spec["output_tiles"]) != OUTPUT_TILES
        or int(cell_spec["input_groups"]) != INPUT_GROUPS
        or int(cell_spec["total_cells"]) != TOTAL_CELLS
        or int(layout["cq3_bytes_per_cell"]) != CQ3_BYTES_PER_CELL
        or int(layout["int4_bytes_per_cell"]) != INT4_BYTES_PER_CELL
    ):
        raise ValueError("v74 frozen representation constants changed")

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
        raise ValueError("v74 full-context artifacts changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in authenticated["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v74 source tensor changed: {name}")

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
        or not set(context["calibration_texts"]).isdisjoint(
            context["validation_texts"]
        )
    ):
        raise ValueError("v74 disjoint-prompt data split changed")

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
        raise ValueError("v74 exact layer replay changed")
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
            raise ValueError(f"v74 trained {role} payload changed")

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

    int4_codes, int4_scales = v73.quantize_signed_int4_grouped(
        source_down,
        group_size=int(quantizer["group_size"]),
        iterations=int(quantizer["least_squares_scale_iterations"]),
    )
    cq3_codes = down.codes.reshape(HIDDEN, INPUT_GROUPS, GROUP_SIZE).contiguous()
    cq3_scales = down.scales.contiguous()
    with torch.no_grad():
        int4_weight = int4_codes.float() * int4_scales.float().unsqueeze(-1)
        cq3_weight = cq3_codes.float() * cq3_scales.float().unsqueeze(-1)
        weight_delta = (int4_weight - cq3_weight).reshape(
            OUTPUT_TILES, TILE_ROWS, INPUT_GROUPS, GROUP_SIZE
        )
        grouped_input = baseline_down_input.float().reshape(
            selected_rows.numel(), INPUT_GROUPS, GROUP_SIZE
        )
        cell_delta = torch.einsum("trgd,ngd->ntgr", weight_delta, grouped_input)
    del down, int4_weight, cq3_weight, weight_delta, grouped_input, source_down
    gc.collect()

    calibration_count = calibration_rows.numel()
    baseline_tiled = baseline_mlp_output.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    exact_tiled = exact_mlp_output.reshape(-1, OUTPUT_TILES, TILE_ROWS)
    calibration_residual = (
        exact_tiled[:calibration_count] - baseline_tiled[:calibration_count]
    )
    selected_groups, final_calibration_residual, greedy_history = greedy_select_cells(
        cell_delta[:calibration_count],
        calibration_residual,
        steps=int(cell_spec["steps_per_output_tile"]),
    )
    correction = selected_cell_correction(cell_delta, selected_groups)
    candidate_tiled = baseline_tiled + correction
    candidate_mlp_output = candidate_tiled.reshape_as(baseline_mlp_output)
    if not torch.allclose(
        final_calibration_residual,
        exact_tiled[:calibration_count] - candidate_tiled[:calibration_count],
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError("v74 greedy residual update changed")

    baseline_calibration_nmse = v71.tensor_nmse(
        baseline_mlp_output[:calibration_count], exact_mlp_output[:calibration_count]
    )
    candidate_calibration_nmse = v71.tensor_nmse(
        candidate_mlp_output[:calibration_count], exact_mlp_output[:calibration_count]
    )
    baseline_confirmation_nmse = v71.tensor_nmse(
        baseline_mlp_output[calibration_count:], exact_mlp_output[calibration_count:]
    )
    candidate_confirmation_nmse = v71.tensor_nmse(
        candidate_mlp_output[calibration_count:], exact_mlp_output[calibration_count:]
    )
    calibration_improvement = (
        baseline_calibration_nmse - candidate_calibration_nmse
    ) / max(baseline_calibration_nmse, 1e-30)
    confirmation_improvement = (
        baseline_confirmation_nmse - candidate_confirmation_nmse
    ) / max(baseline_confirmation_nmse, 1e-30)
    generalization_ratio = confirmation_improvement / max(
        calibration_improvement, 1e-30
    )
    confirmation_base_error = (
        exact_tiled[calibration_count:] - baseline_tiled[calibration_count:]
    ).square().sum(dim=(0, 2))
    confirmation_candidate_error = (
        exact_tiled[calibration_count:] - candidate_tiled[calibration_count:]
    ).square().sum(dim=(0, 2))
    confirmation_tile_benefit = (
        confirmation_base_error - confirmation_candidate_error
    )
    positive_confirmation_tile_fraction = float(
        (confirmation_tile_benefit > 0).float().mean()
    )

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    hybrid_path = args.artifact_root / "blk.59.ffn_down.cellmix-cq3-int4"
    hybrid_payload, storage, packed_sources = build_hybrid_payload(
        cq3_codes,
        cq3_scales,
        int4_codes,
        int4_scales,
        selected_groups,
    )
    if not hybrid_path.exists():
        atomic_bytes(hybrid_path, hybrid_payload)
    elif hybrid_path.read_bytes() != hybrid_payload:
        raise ValueError("existing v74 hybrid payload changed")

    header = HEADER.unpack_from(hybrid_payload)
    if header[0].rstrip(b"\0") != MAGIC:
        raise ValueError("v74 hybrid header changed")
    cq3_offset = HEADER.size + storage["cell_mode_bitmap_bytes"]
    int4_offset = cq3_offset + storage["unselected_cq3_bytes"]
    checked_cq3_codes, checked_cq3_scales = unpack_cq3_group_stream(
        hybrid_payload[cq3_offset:int4_offset],
        rows=packed_sources["cq3_codes"].shape[0] * TILE_ROWS,
        groups=1,
    )
    checked_int4_codes, checked_int4_scales = unpack_int4_group_stream(
        hybrid_payload[int4_offset:],
        rows=packed_sources["int4_codes"].shape[0] * TILE_ROWS,
        groups=1,
    )
    cq3_roundtrip_exact = torch.equal(
        checked_cq3_codes,
        packed_sources["cq3_codes"].reshape(-1, 1, GROUP_SIZE),
    ) and torch.equal(
        checked_cq3_scales,
        packed_sources["cq3_scales"].reshape(-1, 1),
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
        record["minimum_selected_marginal_benefit"] for record in greedy_history
    ) > 0

    gate_spec = prereg["pilot_gate"]
    checks = {
        "exact_layer_replay": True,
        "all_parent_artifacts_authenticated": True,
        "selection_and_confirmation_prompts_disjoint": True,
        "confirmation_rows_unobserved_during_selection": True,
        "all_values_finite": bool(
            torch.isfinite(cell_delta).all()
            and torch.isfinite(int4_scales.float()).all()
            and torch.isfinite(candidate_mlp_output).all()
        ),
        "required_selected_cells_met": selected_groups.numel()
        == int(gate_spec["required_selected_cells"]),
        "all_selected_calibration_marginal_benefits_positive": all_selected_positive,
        "minimum_calibration_improvement_met": calibration_improvement
        >= float(gate_spec["minimum_relative_calibration_mlp_nmse_improvement"]),
        "minimum_confirmation_improvement_met": confirmation_improvement
        >= float(gate_spec["minimum_relative_confirmation_mlp_nmse_improvement"]),
        "minimum_generalization_ratio_met": generalization_ratio
        >= float(gate_spec["minimum_confirmation_to_calibration_improvement_ratio"]),
        "minimum_positive_confirmation_tile_fraction_met": (
            positive_confirmation_tile_fraction
            >= float(gate_spec["minimum_confirmation_positive_output_tile_fraction"])
        ),
        "packed_cq3_roundtrip_exact": cq3_roundtrip_exact,
        "packed_int4_roundtrip_exact": int4_roundtrip_exact,
        "canonical_payload_bytes_exact": storage["total_bytes"]
        == int(layout["hybrid_down_payload_bytes"]),
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
            "v73_negative_receipt_sha256": v73_sha,
            "layer59_context_shard_sha256": v71.file_sha256(shard_path),
            "source_tensor_sha256": authenticated["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
        },
        "data_split": {
            "selection_prompt_count": len(context["calibration_texts"]),
            "selection_row_count": calibration_rows.numel(),
            "confirmation_prompt_count": len(context["validation_texts"]),
            "confirmation_row_count": validation_rows.numel(),
            "selection_and_confirmation_prompts_disjoint": True,
            "confirmation_rows_observed_during_selection": 0,
        },
        "int4_quantizer": {
            "group_size": GROUP_SIZE,
            "least_squares_scale_iterations": int(
                quantizer["least_squares_scale_iterations"]
            ),
            "minimum_scale": float(int4_scales.float().min()),
            "maximum_scale": float(int4_scales.float().max()),
            "zero_scale_groups": int((int4_scales == 0).sum()),
            "packed_roundtrip_exact": int4_roundtrip_exact,
        },
        "cell_selection": {
            "method": "per-output-tile greedy exact current-residual SSE reduction",
            "output_tile_rows": TILE_ROWS,
            "input_group_columns": GROUP_SIZE,
            "output_tiles": OUTPUT_TILES,
            "input_groups": INPUT_GROUPS,
            "steps_per_output_tile": selected_groups.shape[1],
            "selected_cells": selected_groups.numel(),
            "selected_groups_by_output_tile": selected_groups.tolist(),
            "selected_groups_sha256": tensor_bytes_sha256(selected_groups),
            "greedy_history": greedy_history,
            "confirmation_positive_output_tile_fraction": positive_confirmation_tile_fraction,
            "confirmation_rows_observed_during_selection": 0,
        },
        "metrics": {
            "baseline_calibration_complete_mlp_nmse": baseline_calibration_nmse,
            "candidate_calibration_complete_mlp_nmse": candidate_calibration_nmse,
            "relative_calibration_complete_mlp_nmse_improvement": calibration_improvement,
            "baseline_confirmation_complete_mlp_nmse": baseline_confirmation_nmse,
            "candidate_confirmation_complete_mlp_nmse": candidate_confirmation_nmse,
            "relative_confirmation_complete_mlp_nmse_improvement": confirmation_improvement,
            "confirmation_to_calibration_improvement_ratio": generalization_ratio,
        },
        "storage": {
            **storage,
            "hybrid_payload_sha256": v71.file_sha256(hybrid_path),
            "baseline_three_cq3_mlp_payload_bytes": 3 * CQ3_PAYLOAD_BYTES,
            "hybrid_three_projection_mlp_payload_bytes": full_mlp_bytes,
            "relative_full_mlp_payload_increase": storage_increase,
            "additional_runtime_passes": 0,
            "dispatch_cell_shape": [TILE_ROWS, GROUP_SIZE],
            "high_precision_weight_materialization_at_runtime": False,
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "decision": (
                "replicate_greedy_input_cell_routing_across_layers"
                if all(checks.values())
                else "reject_source_int4_input_cell_substitution"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_greedy_input_cell_mixed_down_marker_v74",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": receipt["gate"]["passed"],
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
