#!/usr/bin/env python3
"""Collect the sealed post-preregistration Qwen3.8-27B mixed-mode holdout."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

import collect_qwen38_27b_affine_butterfly_block_cache_v1 as base


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
TARGET_LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention")
SCHEMA = "qwen38_27b_mixed_confirmation_cache_v1"
RECEIPT_SCHEMA = "qwen38_27b_mixed_confirmation_cache_receipt_v1"
ALLOWED_PREREGISTRATION_STATUSES = {
    "frozen_after_terminal_qat_before_mixed_execution",
    "frozen_after_terminal_v36_before_any_fresh_holdout_collection",
}


def validate_existing(
    *,
    cache_path: Path,
    receipt_path: Path,
    preregistration_path: Path,
    index_path: Path,
) -> dict[str, Any] | None:
    if not cache_path.exists() and not receipt_path.exists():
        return None
    if not cache_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("partial mixed confirmation cache requires inspection")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("preregistration_sha256")
        != base.sha256_file(preregistration_path)
        or receipt.get("index_sha256") != base.sha256_file(index_path)
        or receipt.get("cache", {}).get("sha256") != base.sha256_file(cache_path)
        or receipt.get("cache", {}).get("bytes") != cache_path.stat().st_size
        or receipt.get("confirmation_rows") != 32
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
    ):
        raise ValueError("existing mixed confirmation receipt changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name(
            "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1.json"
        ),
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/confirmation-cache.pt"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-ranked-mixed-confirmation-cache-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen mixed confirmation CPU thread count is four")
    base.resource_guard(Path("/workspace"))

    prereg = json.loads(args.preregistration.read_text())
    scope = prereg["scope"]
    protocol = prereg["data"]["fresh_confirmation"]
    texts = list(protocol["texts"])
    if (
        prereg.get("status") not in ALLOWED_PREREGISTRATION_STATUSES
        or scope.get("active_model_only") != MODEL_ID
        or scope.get("revision") != MODEL_REVISION
        or tuple(scope.get("pilot_layers", [])) != TARGET_LAYERS
        or tuple(scope.get("pilot_layer_types", [])) != TARGET_LAYER_TYPES
        or scope.get("qwen_4b_compute_forbidden") is not True
        or len(texts) != 4
        or len(set(texts)) != 4
        or protocol.get("selected_rows_per_layer") != 32
        or tuple(protocol.get("target_layers", [])) != TARGET_LAYERS
        or protocol.get("collected_after_preregistration") is not True
        or protocol.get("may_score_only_the_final_frozen_candidate") is not True
        or protocol.get("checkpoint_or_hyperparameter_selection_forbidden") is not True
    ):
        raise ValueError("mixed confirmation protocol changed")

    training_cache_sha256 = base.sha256_file(args.training_cache)
    if training_cache_sha256 != prereg["data"]["training_cache"]["sha256"]:
        raise ValueError("mixed training cache identity changed")
    training_cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    old_texts = set(training_cache["calibration_texts"]) | set(
        training_cache["validation_texts"]
    )
    if old_texts & set(texts):
        raise ValueError("fresh mixed confirmation overlaps the prior cache")
    del training_cache

    download_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    download = json.loads(download_path.read_text())
    index_path = args.model_dir / "model.safetensors.index.json"
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": base.EXPECTED_INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    } or base.sha256_file(index_path) != base.EXPECTED_INDEX_SHA256:
        raise ValueError("Qwen3.8-27B source identity changed")
    existing = validate_existing(
        cache_path=args.cache,
        receipt_path=args.receipt,
        preregistration_path=args.preregistration,
        index_path=index_path,
    )
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(int(prereg["training"]["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != 5120
        or tuple(config.layer_types[index] for index in TARGET_LAYERS)
        != TARGET_LAYER_TYPES
    ):
        raise ValueError("Qwen3.8-27B text architecture changed")

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(protocol["maximum_sequence_tokens"]),
        padding="max_length",
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    weight_map = json.loads(index_path.read_text())["weight_map"]
    hidden = base.embedding_rows(args.model_dir, weight_map, token_ids)
    if hidden.dtype != torch.bfloat16 or hidden.shape != (4, 64, 5120):
        raise ValueError(f"unexpected exact confirmation embedding: {hidden.dtype} {hidden.shape}")

    positions = torch.arange(hidden.shape[1], dtype=torch.long).view(1, 1, -1)
    positions = positions.expand(4, hidden.shape[0], -1)
    text_positions = positions[0]
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
    confirmation_rows = base.selected_flat_rows(
        attention_mask, range(0, len(texts)), int(protocol["selected_rows_per_layer"])
    )
    if confirmation_rows.numel() != 32:
        raise RuntimeError("fresh mixed confirmation denominator changed")

    cached_layers: dict[str, dict[str, torch.Tensor | str]] = {}
    layer_receipts: list[dict[str, Any]] = []
    started = time.monotonic()
    for layer_index in range(config.num_hidden_layers):
        layer_started = time.monotonic()
        if layer_index in TARGET_LAYERS:
            cached_layers[str(layer_index)] = {
                "type": config.layer_types[layer_index],
                "input": hidden.detach().cpu().contiguous(),
            }
        state, parameters = base.layer_state(args.model_dir, weight_map, layer_index)
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"confirmation layer {layer_index} state mapping changed")
        layer.eval()
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[config.layer_types[layer_index]],
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            ).contiguous()
        if hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError(
                f"non-finite or changed BF16 confirmation after layer {layer_index}"
            )
        if layer_index in TARGET_LAYERS:
            cached_layers[str(layer_index)]["output"] = hidden.detach().cpu().contiguous()
        layer_receipts.append(
            {
                "layer": layer_index,
                "type": config.layer_types[layer_index],
                "parameters": parameters,
                "elapsed_seconds": time.monotonic() - layer_started,
            }
        )
        del layer, state
        gc.collect()

    cache = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": base.sha256_file(args.preregistration),
        "training_cache_sha256": training_cache_sha256,
        "confirmation_texts": texts,
        "input_ids": token_ids,
        "attention_mask": attention_mask.to(torch.int8),
        "text_position_ids": text_positions,
        "confirmation_flat_rows": confirmation_rows,
        "layers": cached_layers,
    }
    base.atomic_torch_save(cache, args.cache)
    cache_descriptor = {
        "path": str(args.cache),
        "bytes": args.cache.stat().st_size,
        "sha256": base.sha256_file(args.cache),
    }
    tensor_receipts = {
        layer_index: {
            kind: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": base.tensor_sha256(value),
            }
            for kind, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        for layer_index, values in cached_layers.items()
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "preregistration": str(args.preregistration),
        "preregistration_sha256": base.sha256_file(args.preregistration),
        "training_cache_sha256": training_cache_sha256,
        "download_receipt_sha256": base.sha256_file(download_path),
        "index_sha256": base.sha256_file(index_path),
        "cache": cache_descriptor,
        "target_layers": list(TARGET_LAYERS),
        "target_layer_types": list(TARGET_LAYER_TYPES),
        "confirmation_texts": 4,
        "confirmation_rows": confirmation_rows.numel(),
        "cached_tensors": tensor_receipts,
        "propagated_layers": layer_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "finite": True,
        "complete": True,
        "terminal": True,
        "claim_boundary": "fresh exact BF16 three-block confirmation cache only; it remains sealed from mixed-mode training and cannot support a quality, runtime, or breakthrough claim",
    }
    base.atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
