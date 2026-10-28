#!/usr/bin/env python3
"""Stream Qwen3.8-27B exactly and retain bounded rows for every text block."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


RUNTIME = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUNTIME))
import collect_qwen38_27b_affine_butterfly_block_cache_v1 as pilot


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EXPECTED_PARENT_CACHE_SHA256 = (
    "564b41507542ef041600c7c2a8f57c01ad012e5b099ec5b002d5829865ec0d19"
)
PILOT_LAYERS = (0, 32, 63)
CHECKPOINT_INTERVAL = 8
SCHEMA = "qwen38_27b_all_layer_selected_block_cache_v41"
RECEIPT_SCHEMA = "qwen38_27b_all_layer_selected_block_cache_receipt_v41"
CHECKPOINT_SCHEMA = "qwen38_27b_all_layer_selected_block_cache_checkpoint_v41"


def selected_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return (
        values.reshape(-1, values.shape[-1])
        .index_select(0, rows)
        .detach()
        .cpu()
        .contiguous()
    )


def trim_allocator() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
    pilot.resource_guard(workspace)
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: all-layer cache disk floor failed")


def validate_layer_cache(
    layers: dict[str, dict[str, torch.Tensor | str]],
    *,
    expected_count: int,
    calibration_rows: int,
    validation_rows: int,
) -> None:
    if set(layers) != {str(index) for index in range(expected_count)}:
        raise ValueError("all-layer cache keys are not a contiguous prefix")
    for index in range(expected_count):
        layer = layers[str(index)]
        expected_type = "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        if layer.get("type") != expected_type:
            raise ValueError("all-layer cache layer type changed")
        for split, rows in (
            ("calibration", calibration_rows),
            ("validation", validation_rows),
        ):
            for endpoint in ("input", "output"):
                value = layer.get(f"{split}_{endpoint}")
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.bfloat16
                    or tuple(value.shape) != (rows, 5120)
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("all-layer selected tensor identity changed")


def validate_final(
    cache_path: Path,
    receipt_path: Path,
    marker_path: Path,
    preregistration: Path,
) -> dict[str, Any] | None:
    existing = [path.exists() for path in (cache_path, receipt_path, marker_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError("partial all-layer terminal cache requires inspection")
    receipt = json.loads(receipt_path.read_text())
    marker = json.loads(marker_path.read_text())
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("preregistration_sha256")
        != pilot.sha256_file(preregistration)
        or receipt.get("cache", {}).get("sha256") != pilot.sha256_file(cache_path)
        or receipt.get("cache", {}).get("bytes") != cache_path.stat().st_size
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or receipt.get("gate", {}).get("passed") is not True
        or marker.get("terminal") is not True
        or marker.get("cache_sha256") != pilot.sha256_file(cache_path)
        or marker.get("receipt_sha256") != pilot.sha256_file(receipt_path)
    ):
        raise ValueError("existing all-layer cache terminal identity changed")
    return receipt


def load_checkpoint(
    path: Path,
    *,
    preregistration_sha256: str,
    calibration_rows: int,
    validation_rows: int,
) -> tuple[int, torch.Tensor, dict[str, dict[str, torch.Tensor | str]], list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    next_layer = int(checkpoint.get("next_layer", -1))
    hidden = checkpoint.get("hidden")
    layers = checkpoint.get("layers")
    layer_receipts = checkpoint.get("layer_receipts")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("model") != MODEL_ID
        or checkpoint.get("revision") != MODEL_REVISION
        or checkpoint.get("preregistration_sha256") != preregistration_sha256
        or checkpoint.get("parent_cache_sha256") != EXPECTED_PARENT_CACHE_SHA256
        or next_layer <= 0
        or next_layer >= 64
        or next_layer % CHECKPOINT_INTERVAL
        or not isinstance(hidden, torch.Tensor)
        or hidden.dtype != torch.bfloat16
        or tuple(hidden.shape) != (12, 64, 5120)
        or not bool(torch.isfinite(hidden).all())
        or not isinstance(layers, dict)
        or not isinstance(layer_receipts, list)
        or len(layer_receipts) != next_layer
    ):
        raise ValueError("all-layer checkpoint identity changed")
    validate_layer_cache(
        layers,
        expected_count=next_layer,
        calibration_rows=calibration_rows,
        validation_rows=validation_rows,
    )
    return next_layer, hidden, layers, layer_receipts


def checkpoint_payload(
    *,
    preregistration_sha256: str,
    next_layer: int,
    hidden: torch.Tensor,
    layers: dict[str, dict[str, torch.Tensor | str]],
    layer_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": preregistration_sha256,
        "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
        "next_layer": next_layer,
        "hidden": hidden.detach().cpu().contiguous(),
        "layers": layers,
        "layer_receipts": layer_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_block_cache_v41.json",
    )
    parser.add_argument(
        "--parent-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-all-layer-block-cache-v41/cache.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-all-layer-block-cache-v41/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-block-cache-v41/receipt.json"
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-block-cache-v41/complete.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen all-layer cache CPU thread count is four")
    resource_guard(Path("/workspace"), minimum_free_disk_bytes=20 * 1024**3)

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = pilot.sha256_file(args.preregistration)
    if (
        prereg.get("schema")
        != "qwen38_27b_all_layer_block_cache_preregistration_v41"
        or prereg.get("status")
        != "frozen_before_all_layer_cache_execution"
        or prereg.get("scope", {}).get("active_model_only") != MODEL_ID
        or prereg.get("scope", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("target_layers") != list(range(64))
        or prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True
    ):
        raise ValueError("all-layer cache preregistration changed")
    existing = validate_final(
        args.cache,
        args.receipt,
        args.completion_marker,
        args.preregistration,
    )
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    download_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    index_path = args.model_dir / "model.safetensors.index.json"
    download = json.loads(download_path.read_text())
    if (
        download.get("model_id") != MODEL_ID
        or download.get("revision") != MODEL_REVISION
        or download.get("bytes") != 55_563_006_776
        or download.get("shards") != 18
        or download.get("index_sha256") != EXPECTED_INDEX_SHA256
        or pilot.sha256_file(index_path) != EXPECTED_INDEX_SHA256
        or pilot.sha256_file(args.parent_cache) != EXPECTED_PARENT_CACHE_SHA256
    ):
        raise ValueError("all-layer source or parent cache identity changed")

    parent = torch.load(args.parent_cache, map_location="cpu", weights_only=False)
    if (
        parent.get("model") != MODEL_ID
        or parent.get("revision") != MODEL_REVISION
        or set(parent.get("layers", {})) != {"0", "32", "63"}
        or parent.get("calibration_flat_rows").numel() != 64
        or parent.get("validation_flat_rows").numel() != 32
    ):
        raise ValueError("authenticated pilot cache structure changed")
    token_ids = parent["input_ids"].clone()
    attention_mask = parent["attention_mask"].to(torch.long)
    text_positions = parent["text_position_ids"].clone()
    calibration_rows = parent["calibration_flat_rows"].clone()
    validation_rows = parent["validation_flat_rows"].clone()
    calibration_texts = list(parent["calibration_texts"])
    validation_texts = list(parent["validation_texts"])
    pilot_expected = {
        layer: {
            endpoint: {
                "calibration": selected_rows(
                    parent["layers"][str(layer)][endpoint], calibration_rows
                ),
                "validation": selected_rows(
                    parent["layers"][str(layer)][endpoint], validation_rows
                ),
            }
            for endpoint in ("input", "output")
        }
        for layer in PILOT_LAYERS
    }
    del parent
    gc.collect()
    trim_allocator()

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(int(prereg["execution"]["seed"]))
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != 5120
        or tuple(config.layer_types)
        != tuple(
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(64)
        )
    ):
        raise ValueError("Qwen3.8-27B all-layer architecture changed")

    weight_map = json.loads(index_path.read_text())["weight_map"]
    if args.checkpoint.exists():
        next_layer, hidden, cached_layers, layer_receipts = load_checkpoint(
            args.checkpoint,
            preregistration_sha256=prereg_sha,
            calibration_rows=calibration_rows.numel(),
            validation_rows=validation_rows.numel(),
        )
        resumed_from_layer = next_layer
    else:
        hidden = pilot.embedding_rows(args.model_dir, weight_map, token_ids)
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (12, 64, 5120):
            raise ValueError("unexpected exact embedding state")
        next_layer = 0
        resumed_from_layer = None
        cached_layers = {}
        layer_receipts = []

    positions = torch.arange(hidden.shape[1], dtype=torch.long).view(1, 1, -1)
    positions = positions.expand(4, hidden.shape[0], -1)
    if not torch.equal(positions[0], text_positions):
        raise ValueError("text position identity changed")
    rotary_positions = positions[1:]
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, rotary_positions)
    masks = {
        "full_attention": create_causal_mask(
            config, hidden, attention_mask, None, text_positions
        ),
        "linear_attention": create_recurrent_attention_mask(
            config,
            hidden,
            attention_mask,
            past_key_values=None,
            position_ids=text_positions,
        ),
    }

    started = time.monotonic()
    for layer_index in range(next_layer, 64):
        layer_started = time.monotonic()
        layer_type = config.layer_types[layer_index]
        record: dict[str, torch.Tensor | str] = {
            "type": layer_type,
            "calibration_input": selected_rows(hidden, calibration_rows),
            "validation_input": selected_rows(hidden, validation_rows),
        }
        if layer_index in PILOT_LAYERS:
            for split in ("calibration", "validation"):
                if not torch.equal(
                    record[f"{split}_input"],
                    pilot_expected[layer_index]["input"][split],
                ):
                    raise RuntimeError("pilot block input reproduction changed")

        state, parameters = pilot.layer_state(
            args.model_dir, weight_map, layer_index
        )
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("all-layer state mapping changed")
        layer.eval()
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            ).contiguous()
        if hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError("non-finite or changed BF16 all-layer state")
        record["calibration_output"] = selected_rows(hidden, calibration_rows)
        record["validation_output"] = selected_rows(hidden, validation_rows)
        if layer_index in PILOT_LAYERS:
            for split in ("calibration", "validation"):
                if not torch.equal(
                    record[f"{split}_output"],
                    pilot_expected[layer_index]["output"][split],
                ):
                    raise RuntimeError("pilot block output reproduction changed")
        cached_layers[str(layer_index)] = record
        layer_receipts.append(
            {
                "layer": layer_index,
                "type": layer_type,
                "parameters": parameters,
                "elapsed_seconds": time.monotonic() - layer_started,
                "pilot_bit_exact": layer_index in PILOT_LAYERS,
            }
        )
        del layer, state
        gc.collect()
        trim_allocator()

        completed = layer_index + 1
        if completed < 64 and completed % CHECKPOINT_INTERVAL == 0:
            pilot.atomic_torch_save(
                checkpoint_payload(
                    preregistration_sha256=prereg_sha,
                    next_layer=completed,
                    hidden=hidden,
                    layers=cached_layers,
                    layer_receipts=layer_receipts,
                ),
                args.checkpoint,
            )

    validate_layer_cache(
        cached_layers,
        expected_count=64,
        calibration_rows=64,
        validation_rows=32,
    )
    cache = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": prereg_sha,
        "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
        "calibration_texts": calibration_texts,
        "validation_texts": validation_texts,
        "calibration_rows": calibration_rows,
        "validation_rows": validation_rows,
        "layers": cached_layers,
    }
    pilot.atomic_torch_save(cache, args.cache)
    cache_sha = pilot.sha256_file(args.cache)
    tensor_receipts = {
        layer_index: {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": pilot.tensor_sha256(value),
            }
            for key, value in layer.items()
            if isinstance(value, torch.Tensor)
        }
        for layer_index, layer in cached_layers.items()
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration": str(args.preregistration),
        "preregistration_sha256": prereg_sha,
        "download_receipt_sha256": pilot.sha256_file(download_path),
        "index_sha256": pilot.sha256_file(index_path),
        "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
        "cache": {
            "path": str(args.cache),
            "bytes": args.cache.stat().st_size,
            "sha256": cache_sha,
        },
        "layers": tensor_receipts,
        "propagated_layers": layer_receipts,
        "resumed_from_layer": resumed_from_layer,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "gate": {
            "exact_layer_count": len(cached_layers) == 64,
            "calibration_rows_per_layer": 64,
            "validation_rows_per_layer": 32,
            "all_values_finite": True,
            "pilot_layers_bit_exact": list(PILOT_LAYERS),
            "maximum_cache_bytes": int(prereg["gate"]["maximum_cache_bytes"]),
            "cache_within_maximum": args.cache.stat().st_size
            <= int(prereg["gate"]["maximum_cache_bytes"]),
            "passed": args.cache.stat().st_size
            <= int(prereg["gate"]["maximum_cache_bytes"]),
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": prereg["claim_boundary"],
    }
    pilot.atomic_json(receipt, args.receipt)
    pilot.atomic_json(
        {
            "terminal": True,
            "passed": receipt["gate"]["passed"],
            "cache_sha256": cache_sha,
            "receipt_sha256": pilot.sha256_file(args.receipt),
        },
        args.completion_marker,
    )
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
