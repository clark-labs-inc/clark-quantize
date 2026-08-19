#!/usr/bin/env python3
"""Test exact-marginal x8-tile CQ3/INT4 routing for Qwen3.8-27B.

For a fixed number of equal-cost output tiles, complete-MLP squared error is
separable across tiles.  Selecting the tiles with the largest CQ3-to-INT4
marginal SSE reductions is therefore the exact training optimum at that
budget.  This runner freezes that selection on 32 layer-59 rows and measures
the untouched interleaved 32-row confirmation split.
"""

from __future__ import annotations

import argparse
import gc
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
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_mixed_cq3_runtime import tensor_sha256

MODEL_ID = v71.MODEL_ID
MODEL_REVISION = v71.MODEL_REVISION
LAYER = v71.LAYER
ROWS = v71.ROWS
HIDDEN = v71.HIDDEN
INTERMEDIATE = v71.INTERMEDIATE
CQ3_PAYLOAD_BYTES = v71.PAYLOAD_BYTES
CQ3_ROW_BYTES = CQ3_PAYLOAD_BYTES // HIDDEN
TILE_ROWS = 8
TOTAL_TILES = HIDDEN // TILE_ROWS
INT4_GROUP = 32
INT4_BYTES_PER_GROUP = 18
INT4_ROW_BYTES = INTERMEDIATE // INT4_GROUP * INT4_BYTES_PER_GROUP
HEADER = struct.Struct("<16s12I")
MAGIC = b"CQ3Q4TILEV1"
SCHEMA = "qwen38_27b_exact_marginal_tile_mixed_down_receipt_v73"


def quantize_signed_int4_grouped(
    source: torch.Tensor, *, group_size: int, iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        source.ndim != 2
        or source.shape[1] % group_size
        or group_size % 2
        or iterations <= 0
    ):
        raise ValueError("invalid signed-INT4 quantization input")
    grouped = source.detach().float().reshape(
        source.shape[0], source.shape[1] // group_size, group_size
    )
    maximum = grouped.abs().amax(dim=-1)
    scale = torch.where(maximum > 0, maximum / 7.0, torch.zeros_like(maximum))
    codes = torch.zeros_like(grouped, dtype=torch.int8)
    for _ in range(iterations):
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(grouped / safe.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        code_float = codes.float()
        denominator = code_float.square().sum(dim=-1)
        numerator = (grouped * code_float).sum(dim=-1)
        scale = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(denominator),
        ).clamp_min(0)
    stored_scale = scale.to(torch.float16)
    safe = torch.where(
        stored_scale.float() > 0,
        stored_scale.float(),
        torch.ones_like(stored_scale.float()),
    )
    codes = torch.round(grouped / safe.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    return codes.contiguous(), stored_scale.contiguous()


def dequantize_signed_int4(
    codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    if (
        codes.ndim != 3
        or scales.shape != codes.shape[:2]
        or codes.dtype != torch.int8
        or scales.dtype != torch.float16
    ):
        raise ValueError("invalid signed-INT4 tensors")
    return (codes.float() * scales.float().unsqueeze(-1)).reshape(
        codes.shape[0], -1
    )


def pack_signed_int4(codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    if codes.shape[-1] != INT4_GROUP:
        raise ValueError("v73 INT4 group changed")
    code_array = codes.detach().cpu().numpy().astype(np.int8, copy=False)
    low = code_array[..., 0::2].astype(np.int16) & 0xF
    high = code_array[..., 1::2].astype(np.int16) & 0xF
    packed = (low | (high << 4)).astype(np.uint8)
    scale_bytes = (
        scales.detach()
        .cpu()
        .numpy()
        .astype("<f2", copy=False)
        .view(np.uint8)
        .reshape(*scales.shape, 2)
    )
    blocks = np.concatenate((packed, scale_bytes), axis=-1)
    return blocks.tobytes(order="C")


def unpack_signed_int4(
    payload: bytes, *, rows: int, columns: int
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = columns // INT4_GROUP
    expected = rows * groups * INT4_BYTES_PER_GROUP
    if columns % INT4_GROUP or len(payload) != expected:
        raise ValueError("signed-INT4 payload size changed")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(
        rows, groups, INT4_BYTES_PER_GROUP
    )
    packed = blocks[..., : INT4_GROUP // 2]
    low = (packed & 0xF).astype(np.int8)
    high = ((packed >> 4) & 0xF).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    codes = np.empty((rows, groups, INT4_GROUP), dtype=np.int8)
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    scale_bytes = blocks[..., INT4_GROUP // 2 :].copy()
    scales = scale_bytes.reshape(rows, groups, 2).view("<f2").reshape(rows, groups)
    return torch.from_numpy(codes.copy()), torch.from_numpy(scales.copy())


def tile_marginal_benefits(
    baseline: torch.Tensor,
    alternative: torch.Tensor,
    target: torch.Tensor,
    row_index: torch.Tensor,
    *,
    tile_rows: int,
) -> torch.Tensor:
    if baseline.shape != alternative.shape or baseline.shape != target.shape:
        raise ValueError("tile-marginal output shapes changed")
    baseline_error = (
        baseline.index_select(0, row_index) - target.index_select(0, row_index)
    ).square().sum(dim=0)
    alternative_error = (
        alternative.index_select(0, row_index) - target.index_select(0, row_index)
    ).square().sum(dim=0)
    if baseline_error.numel() % tile_rows:
        raise ValueError("output rows no longer align to tiles")
    return (baseline_error - alternative_error).reshape(-1, tile_rows).sum(dim=1)


def select_exact_fixed_budget_tiles(
    benefits: torch.Tensor, *, count: int
) -> list[int]:
    if benefits.ndim != 1 or not 0 < count < benefits.numel():
        raise ValueError("invalid fixed-budget tile selection")
    values = benefits.detach().cpu().tolist()
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[
        :count
    ]


def tile_rows(tile_indices: list[int], *, tile_size: int) -> torch.Tensor:
    return torch.tensor(
        [
            row
            for tile in sorted(tile_indices)
            for row in range(tile * tile_size, (tile + 1) * tile_size)
        ],
        dtype=torch.long,
    )


def build_hybrid_payload(
    cq3_payload: bytes,
    int4_codes: torch.Tensor,
    int4_scales: torch.Tensor,
    selected_tiles: list[int],
) -> tuple[bytes, dict[str, int]]:
    if len(cq3_payload) != CQ3_PAYLOAD_BYTES:
        raise ValueError("v73 parent CQ3 payload size changed")
    selected = set(selected_tiles)
    if len(selected) != len(selected_tiles):
        raise ValueError("duplicate selected tile")
    mode = bytearray((TOTAL_TILES + 7) // 8)
    cq3_parts: list[bytes] = []
    for tile in range(TOTAL_TILES):
        start = tile * TILE_ROWS * CQ3_ROW_BYTES
        stop = start + TILE_ROWS * CQ3_ROW_BYTES
        if tile in selected:
            mode[tile // 8] |= 1 << (tile % 8)
        else:
            cq3_parts.append(cq3_payload[start:stop])
    rows = tile_rows(selected_tiles, tile_size=TILE_ROWS)
    q4 = pack_signed_int4(
        int4_codes.index_select(0, rows), int4_scales.index_select(0, rows)
    )
    cq3 = b"".join(cq3_parts)
    header = HEADER.pack(
        MAGIC,
        HIDDEN,
        INTERMEDIATE,
        TILE_ROWS,
        INT4_GROUP,
        len(selected_tiles),
        TOTAL_TILES,
        len(mode),
        len(cq3),
        len(q4),
        HEADER.size,
        1,
        0,
    )
    payload = header + bytes(mode) + cq3 + q4
    return payload, {
        "header_bytes": len(header),
        "tile_mode_bitmap_bytes": len(mode),
        "unselected_cq3_bytes": len(cq3),
        "selected_int4_bytes": len(q4),
        "total_bytes": len(payload),
    }


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
    parser.add_argument("--v71-receipt", required=True, type=Path)
    parser.add_argument("--v72-receipt", required=True, type=Path)
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
        raise ValueError("v73 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v73 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v73 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = v71.file_sha256(args.preregistration)
    full_context_sha = v71.file_sha256(args.full_context_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v71_sha = v71.file_sha256(args.v71_receipt)
    v72_sha = v71.file_sha256(args.v72_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    prior71 = json.loads(args.v71_receipt.read_text())
    prior72 = json.loads(args.v72_receipt.read_text())
    authenticated = prereg["authenticated_inputs"]
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_exact_marginal_tile_mixed_down_preregistration_v73"
        or prereg.get("status") != "frozen_before_execution"
        or prereg.get("scope", {}).get("layer") != LAYER
        or authenticated.get("full_context_receipt_sha256") != full_context_sha
        or authenticated.get("v44_training_receipt_sha256") != v44_sha
        or authenticated.get("v71_negative_receipt_sha256") != v71_sha
        or authenticated.get("v72_negative_receipt_sha256") != v72_sha
        or context_receipt.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
        or prior71.get("gate", {}).get("passed") is not False
        or prior72.get("gate", {}).get("passed") is not False
        or prior72.get("gauge_optimization", {}).get("selected_step") != 0
    ):
        raise ValueError("v73 frozen protocol or authenticated parents changed")

    selection = prereg["tile_selection"]
    quantizer = prereg["int4_quantizer"]
    layout = prereg["canonical_layout"]
    if (
        int(selection["output_tile_rows"]) != TILE_ROWS
        or int(selection["total_tiles"]) != TOTAL_TILES
        or int(quantizer["group_size"]) != INT4_GROUP
        or int(layout["cq3_bytes_per_row"]) != CQ3_ROW_BYTES
        or int(layout["int4_bytes_per_row"]) != INT4_ROW_BYTES
    ):
        raise ValueError("v73 frozen representation constants changed")

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
        raise ValueError("v73 full-context artifacts changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in authenticated["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v73 source tensor changed: {name}")

    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    calibration_rows = context["calibration_flat_rows"].long()
    hidden = shard["input"][:8].contiguous()
    teacher_layer_output = shard["output"][:8].contiguous()
    attention_mask = context["attention_mask"].long()[:8]
    position_ids = context["text_position_ids"].long()[:8]
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
            exact_layer, exact_forward, calibration_rows
        )
    )
    if not torch.equal(exact_output, teacher_layer_output):
        raise ValueError("v73 exact layer replay changed")
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
            raise ValueError(f"v73 trained {role} payload changed")

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
    del gate, up, down
    gc.collect()

    split = prereg["row_split"]
    train_index = torch.arange(int(split["train_start"]), ROWS, int(split["stride"]))
    confirmation_index = torch.arange(
        int(split["confirmation_start"]), ROWS, int(split["stride"])
    )
    if train_index.numel() != 32 or confirmation_index.numel() != 32:
        raise ValueError("v73 frozen row split changed")

    int4_codes, int4_scales = quantize_signed_int4_grouped(
        source_down,
        group_size=int(quantizer["group_size"]),
        iterations=int(quantizer["least_squares_scale_iterations"]),
    )
    all_int4_output = torch.empty_like(exact_mlp_output)
    with torch.no_grad():
        for start in range(0, HIDDEN, 128):
            stop = start + 128
            reconstructed = dequantize_signed_int4(
                int4_codes[start:stop], int4_scales[start:stop]
            )
            all_int4_output[:, start:stop] = functional.linear(
                baseline_down_input.float(), reconstructed
            )
            del reconstructed

    training_benefits = tile_marginal_benefits(
        baseline_mlp_output,
        all_int4_output,
        exact_mlp_output,
        train_index,
        tile_rows=TILE_ROWS,
    )
    selected_tiles = select_exact_fixed_budget_tiles(
        training_benefits, count=int(selection["int4_tiles"])
    )
    selected_rows = tile_rows(selected_tiles, tile_size=TILE_ROWS)
    candidate_mlp_output = baseline_mlp_output.clone()
    candidate_mlp_output[:, selected_rows] = all_int4_output[:, selected_rows]
    confirmation_benefits = tile_marginal_benefits(
        baseline_mlp_output,
        all_int4_output,
        exact_mlp_output,
        confirmation_index,
        tile_rows=TILE_ROWS,
    )

    baseline_train_nmse = v71.tensor_nmse(
        baseline_mlp_output.index_select(0, train_index),
        exact_mlp_output.index_select(0, train_index),
    )
    candidate_train_nmse = v71.tensor_nmse(
        candidate_mlp_output.index_select(0, train_index),
        exact_mlp_output.index_select(0, train_index),
    )
    all_int4_train_nmse = v71.tensor_nmse(
        all_int4_output.index_select(0, train_index),
        exact_mlp_output.index_select(0, train_index),
    )
    baseline_confirmation_nmse = v71.tensor_nmse(
        baseline_mlp_output.index_select(0, confirmation_index),
        exact_mlp_output.index_select(0, confirmation_index),
    )
    candidate_confirmation_nmse = v71.tensor_nmse(
        candidate_mlp_output.index_select(0, confirmation_index),
        exact_mlp_output.index_select(0, confirmation_index),
    )
    all_int4_confirmation_nmse = v71.tensor_nmse(
        all_int4_output.index_select(0, confirmation_index),
        exact_mlp_output.index_select(0, confirmation_index),
    )
    train_improvement = (baseline_train_nmse - candidate_train_nmse) / max(
        baseline_train_nmse, 1e-30
    )
    confirmation_improvement = (
        baseline_confirmation_nmse - candidate_confirmation_nmse
    ) / max(baseline_confirmation_nmse, 1e-30)
    all_int4_confirmation_gain = (
        baseline_confirmation_nmse - all_int4_confirmation_nmse
    )
    confirmation_gain_fraction = (
        (baseline_confirmation_nmse - candidate_confirmation_nmse)
        / all_int4_confirmation_gain
        if all_int4_confirmation_gain > 0
        else -1.0
    )

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    hybrid_path = args.artifact_root / "blk.59.ffn_down.tilemix-cq3-int4"
    hybrid_payload, storage = build_hybrid_payload(
        payloads["down"].read_bytes(),
        int4_codes,
        int4_scales,
        selected_tiles,
    )
    if not hybrid_path.exists():
        atomic_bytes(hybrid_path, hybrid_payload)
    elif hybrid_path.read_bytes() != hybrid_payload:
        raise ValueError("existing v73 hybrid payload changed")

    unpacked = HEADER.unpack_from(hybrid_payload)
    if unpacked[0].rstrip(b"\0") != MAGIC:
        raise ValueError("v73 hybrid header magic changed")
    q4_offset = HEADER.size + storage["tile_mode_bitmap_bytes"] + storage[
        "unselected_cq3_bytes"
    ]
    packed_q4 = hybrid_payload[q4_offset:]
    checked_codes, checked_scales = unpack_signed_int4(
        packed_q4, rows=selected_rows.numel(), columns=INTERMEDIATE
    )
    selected_codes = int4_codes.index_select(0, selected_rows)
    selected_scales = int4_scales.index_select(0, selected_rows)
    int4_roundtrip_exact = torch.equal(checked_codes, selected_codes) and torch.equal(
        checked_scales, selected_scales
    )
    expected_storage = prereg["canonical_layout"]
    full_mlp_bytes = 2 * CQ3_PAYLOAD_BYTES + storage["total_bytes"]
    storage_increase = (full_mlp_bytes - 3 * CQ3_PAYLOAD_BYTES) / (
        3 * CQ3_PAYLOAD_BYTES
    )

    selected_train = training_benefits[selected_tiles]
    selected_confirmation = confirmation_benefits[selected_tiles]
    unselected = [tile for tile in range(TOTAL_TILES) if tile not in set(selected_tiles)]
    exact_fixed_budget_optimum = float(selected_train.min()) >= float(
        training_benefits[unselected].max()
    )
    gate_spec = prereg["pilot_gate"]
    checks = {
        "exact_layer_replay": True,
        "all_parent_artifacts_authenticated": True,
        "all_values_finite": bool(
            torch.isfinite(training_benefits).all()
            and torch.isfinite(confirmation_benefits).all()
            and torch.isfinite(int4_scales.float()).all()
        ),
        "fixed_budget_selection_is_exact": exact_fixed_budget_optimum,
        "required_selected_tiles_met": len(selected_tiles)
        == int(gate_spec["required_selected_tiles"]),
        "all_selected_training_marginal_benefits_positive": bool(
            (selected_train > 0).all()
        ),
        "minimum_train_improvement_met": train_improvement
        >= float(gate_spec["minimum_relative_train_mlp_nmse_improvement"]),
        "minimum_confirmation_improvement_met": confirmation_improvement
        >= float(gate_spec["minimum_relative_confirmation_mlp_nmse_improvement"]),
        "all_int4_confirmation_improves_baseline": all_int4_confirmation_gain > 0,
        "minimum_confirmation_gain_fraction_met": confirmation_gain_fraction
        >= float(gate_spec["minimum_confirmation_gain_fraction_of_all_int4_gain"]),
        "packed_int4_roundtrip_exact": int4_roundtrip_exact,
        "canonical_payload_bytes_exact": storage["total_bytes"]
        == int(expected_storage["hybrid_down_payload_bytes"]),
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
            "v71_negative_receipt_sha256": v71_sha,
            "v72_negative_receipt_sha256": v72_sha,
            "layer59_context_shard_sha256": v71.file_sha256(shard_path),
            "source_tensor_sha256": authenticated["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
        },
        "row_split": {
            "training_rows": train_index.tolist(),
            "confirmation_rows": confirmation_index.tolist(),
            "training_count": train_index.numel(),
            "confirmation_count": confirmation_index.numel(),
        },
        "int4_quantizer": {
            "group_size": INT4_GROUP,
            "least_squares_scale_iterations": int(
                quantizer["least_squares_scale_iterations"]
            ),
            "minimum_scale": float(int4_scales.float().min()),
            "maximum_scale": float(int4_scales.float().max()),
            "zero_scale_groups": int((int4_scales == 0).sum()),
            "packed_roundtrip_exact": int4_roundtrip_exact,
        },
        "tile_selection": {
            "method": "exact uniform-cost top marginal training-SSE reduction",
            "tile_rows": TILE_ROWS,
            "total_tiles": TOTAL_TILES,
            "selected_count": len(selected_tiles),
            "selected_tiles_in_benefit_order": selected_tiles,
            "selected_tiles_in_storage_order": sorted(selected_tiles),
            "minimum_selected_training_benefit": float(selected_train.min()),
            "median_selected_training_benefit": float(selected_train.median()),
            "maximum_selected_training_benefit": float(selected_train.max()),
            "selected_confirmation_positive_fraction": float(
                (selected_confirmation > 0).float().mean()
            ),
            "selected_confirmation_total_benefit": float(
                selected_confirmation.sum()
            ),
            "confirmation_rows_observed_during_selection": 0,
        },
        "metrics": {
            "baseline_train_complete_mlp_nmse": baseline_train_nmse,
            "candidate_train_complete_mlp_nmse": candidate_train_nmse,
            "all_int4_train_complete_mlp_nmse": all_int4_train_nmse,
            "relative_train_complete_mlp_nmse_improvement": train_improvement,
            "baseline_confirmation_complete_mlp_nmse": baseline_confirmation_nmse,
            "candidate_confirmation_complete_mlp_nmse": candidate_confirmation_nmse,
            "all_int4_confirmation_complete_mlp_nmse": all_int4_confirmation_nmse,
            "relative_confirmation_complete_mlp_nmse_improvement": confirmation_improvement,
            "confirmation_gain_fraction_of_all_int4_gain": confirmation_gain_fraction,
        },
        "storage": {
            **storage,
            "hybrid_payload_sha256": v71.file_sha256(hybrid_path),
            "baseline_three_cq3_mlp_payload_bytes": 3 * CQ3_PAYLOAD_BYTES,
            "hybrid_three_projection_mlp_payload_bytes": full_mlp_bytes,
            "relative_full_mlp_payload_increase": storage_increase,
            "additional_runtime_passes": 0,
            "dispatch_granularity_output_rows": TILE_ROWS,
            "high_precision_weight_materialization_at_runtime": False,
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "decision": (
                "replicate_exact_marginal_tile_routing_across_layers"
                if all(checks.values())
                else "reject_fixed_budget_output_tile_mixed_precision_at_12_5_percent"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_exact_marginal_tile_mixed_down_marker_v73",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": receipt["gate"]["passed"],
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
