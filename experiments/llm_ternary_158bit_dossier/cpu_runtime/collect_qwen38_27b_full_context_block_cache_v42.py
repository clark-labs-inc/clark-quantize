#!/usr/bin/env python3
"""Stream exact Qwen3.8-27B blocks into restartable full-context shards."""

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
CONTEXT_SCHEMA = "qwen38_27b_full_context_metadata_v42"
SHARD_SCHEMA = "qwen38_27b_full_context_block_shard_v42"
SHARD_DESCRIPTOR_SCHEMA = "qwen38_27b_full_context_block_shard_descriptor_v42"
RECEIPT_SCHEMA = "qwen38_27b_full_context_block_cache_receipt_v42"


def trim_allocator() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def expected_layer_type(index: int) -> str:
    if index < 0 or index >= 64:
        raise ValueError("layer index is outside the Qwen3.8 text trunk")
    return "full_attention" if (index + 1) % 4 == 0 else "linear_attention"


def validate_context(context: dict[str, Any], preregistration_sha256: str) -> None:
    if (
        context.get("schema") != CONTEXT_SCHEMA
        or context.get("model") != MODEL_ID
        or context.get("revision") != MODEL_REVISION
        or context.get("preregistration_sha256") != preregistration_sha256
        or context.get("parent_cache_sha256") != EXPECTED_PARENT_CACHE_SHA256
        or not isinstance(context.get("input_ids"), torch.Tensor)
        or tuple(context["input_ids"].shape) != (12, 64)
        or context["input_ids"].dtype != torch.int64
        or not isinstance(context.get("attention_mask"), torch.Tensor)
        or tuple(context["attention_mask"].shape) != (12, 64)
        or context["attention_mask"].dtype != torch.int8
        or not isinstance(context.get("text_position_ids"), torch.Tensor)
        or tuple(context["text_position_ids"].shape) != (12, 64)
        or context["text_position_ids"].dtype != torch.int64
        or not isinstance(context.get("calibration_flat_rows"), torch.Tensor)
        or context["calibration_flat_rows"].numel() != 64
        or not isinstance(context.get("validation_flat_rows"), torch.Tensor)
        or context["validation_flat_rows"].numel() != 32
        or len(context.get("calibration_texts", [])) != 8
        or len(context.get("validation_texts", [])) != 4
    ):
        raise ValueError("full-context metadata identity changed")


def contexts_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = {
        "schema",
        "model",
        "revision",
        "preregistration_sha256",
        "parent_cache_sha256",
        "calibration_texts",
        "validation_texts",
        "input_ids",
        "attention_mask",
        "text_position_ids",
        "calibration_flat_rows",
        "validation_flat_rows",
    }
    if set(left) != keys or set(right) != keys:
        return False
    for key in keys:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            if not isinstance(right_value, torch.Tensor) or not torch.equal(
                left_value, right_value
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def validate_shard_tensors(
    shard: dict[str, Any],
    layer_index: int,
    *,
    preregistration_sha256: str,
    producer_commit: str,
) -> None:
    if (
        shard.get("schema") != SHARD_SCHEMA
        or shard.get("model") != MODEL_ID
        or shard.get("revision") != MODEL_REVISION
        or shard.get("preregistration_sha256") != preregistration_sha256
        or shard.get("parent_cache_sha256") != EXPECTED_PARENT_CACHE_SHA256
        or shard.get("producer_commit") != producer_commit
        or shard.get("layer") != layer_index
        or shard.get("type") != expected_layer_type(layer_index)
    ):
        raise ValueError("full-context layer shard identity changed")
    for endpoint in ("input", "output"):
        value = shard.get(endpoint)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bfloat16
            or tuple(value.shape) != (12, 64, 5120)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("full-context layer tensor changed")


def validate_descriptor(
    descriptor: dict[str, Any],
    shard_path: Path,
    *,
    layer_index: int,
    preregistration_sha256: str,
    producer_commit: str,
) -> None:
    if (
        descriptor.get("schema") != SHARD_DESCRIPTOR_SCHEMA
        or descriptor.get("model") != MODEL_ID
        or descriptor.get("revision") != MODEL_REVISION
        or descriptor.get("preregistration_sha256") != preregistration_sha256
        or descriptor.get("parent_cache_sha256") != EXPECTED_PARENT_CACHE_SHA256
        or descriptor.get("producer_commit") != producer_commit
        or descriptor.get("layer") != layer_index
        or descriptor.get("type") != expected_layer_type(layer_index)
        or descriptor.get("shard", {}).get("bytes") != shard_path.stat().st_size
        or descriptor.get("shard", {}).get("sha256")
        != pilot.sha256_file(shard_path)
        or descriptor.get("input", {}).get("shape") != [12, 64, 5120]
        or descriptor.get("input", {}).get("dtype") != "torch.bfloat16"
        or descriptor.get("output", {}).get("shape") != [12, 64, 5120]
        or descriptor.get("output", {}).get("dtype") != "torch.bfloat16"
        or descriptor.get("complete") is not True
        or descriptor.get("terminal") is not True
    ):
        raise ValueError("full-context layer descriptor changed")


def scan_layer_prefix(
    layers_root: Path,
    *,
    preregistration_sha256: str,
    producer_commit: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    gap_seen = False
    for layer_index in range(64):
        shard_path = layers_root / f"{layer_index:02d}.pt"
        descriptor_path = layers_root / f"{layer_index:02d}.json"
        if shard_path.exists() != descriptor_path.exists():
            raise RuntimeError("partial full-context layer shard requires inspection")
        if not shard_path.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise RuntimeError("full-context layer shard prefix has a gap")
        descriptor = json.loads(descriptor_path.read_text())
        validate_descriptor(
            descriptor,
            shard_path,
            layer_index=layer_index,
            preregistration_sha256=preregistration_sha256,
            producer_commit=producer_commit,
        )
        records.append(descriptor)
    return records


def validate_final(
    receipt_path: Path,
    marker_path: Path,
    artifact_root: Path,
    *,
    preregistration_sha256: str,
) -> dict[str, Any] | None:
    if not receipt_path.exists() and not marker_path.exists():
        return None
    if not receipt_path.is_file() or not marker_path.is_file():
        raise RuntimeError("partial full-context terminal receipt requires inspection")
    receipt = json.loads(receipt_path.read_text())
    marker = json.loads(marker_path.read_text())
    producer_commit = receipt.get("repository_commit")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("preregistration_sha256") != preregistration_sha256
        or not isinstance(producer_commit, str)
        or len(producer_commit) != 40
        or len(receipt.get("layers", [])) != 64
        or receipt.get("gate", {}).get("passed") is not True
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or marker.get("terminal") is not True
        or marker.get("passed") is not True
        or marker.get("receipt_sha256") != pilot.sha256_file(receipt_path)
        or marker.get("layer_count") != 64
    ):
        raise ValueError("existing full-context terminal receipt changed")
    context_path = artifact_root / "context.pt"
    context_descriptor_path = artifact_root / "context.json"
    if not context_path.is_file() or not context_descriptor_path.is_file():
        raise RuntimeError("terminal full-context metadata is incomplete")
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    validate_context(context, preregistration_sha256)
    context_descriptor = json.loads(context_descriptor_path.read_text())
    if (
        context_descriptor.get("schema") != CONTEXT_SCHEMA
        or context_descriptor.get("sha256") != pilot.sha256_file(context_path)
        or context_descriptor.get("bytes") != context_path.stat().st_size
        or context_descriptor.get("preregistration_sha256")
        != preregistration_sha256
        or context_descriptor.get("complete") is not True
        or context_descriptor.get("terminal") is not True
        or receipt.get("context", {}).get("sha256")
        != context_descriptor.get("sha256")
        or receipt.get("context", {}).get("bytes")
        != context_descriptor.get("bytes")
    ):
        raise ValueError("terminal full-context metadata changed")
    records = scan_layer_prefix(
        artifact_root / "layers",
        preregistration_sha256=preregistration_sha256,
        producer_commit=producer_commit,
    )
    if (
        records != receipt["layers"]
        or marker.get("total_shard_bytes")
        != sum(record["shard"]["bytes"] for record in records)
        or receipt.get("gate", {}).get("total_shard_bytes")
        != marker.get("total_shard_bytes")
    ):
        raise ValueError("terminal full-context shard ledger changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_full_context_block_cache_v42.json",
    )
    parser.add_argument(
        "--parent-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-full-context-block-cache-v42"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-full-context-block-cache-v42/receipt.json"
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-full-context-block-cache-v42/complete.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen full-context cache uses four CPU threads")
    pilot.resource_guard(Path("/workspace"))
    stat = os.statvfs("/workspace")
    if stat.f_bavail * stat.f_frsize < 20 * 1024**3:
        raise SystemExit("infrastructure-invalid: full-context cache disk floor failed")

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = pilot.sha256_file(args.preregistration)
    producer_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_full_context_block_cache_preregistration_v42"
        or prereg.get("status") != "frozen_before_full_context_execution"
        or prereg.get("scope", {}).get("active_model_only") != MODEL_ID
        or prereg.get("scope", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("target_layers") != list(range(64))
        or prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True
    ):
        raise ValueError("full-context cache preregistration changed")
    existing = validate_final(
        args.receipt,
        args.completion_marker,
        args.artifact_root,
        preregistration_sha256=prereg_sha,
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
        raise ValueError("full-context source identity changed")

    parent = torch.load(args.parent_cache, map_location="cpu", weights_only=False)
    if (
        parent.get("model") != MODEL_ID
        or parent.get("revision") != MODEL_REVISION
        or set(parent.get("layers", {})) != {"0", "32", "63"}
    ):
        raise ValueError("authenticated pilot cache identity changed")
    context = {
        "schema": CONTEXT_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": prereg_sha,
        "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
        "calibration_texts": list(parent["calibration_texts"]),
        "validation_texts": list(parent["validation_texts"]),
        "input_ids": parent["input_ids"].clone(),
        "attention_mask": parent["attention_mask"].to(torch.int8).clone(),
        "text_position_ids": parent["text_position_ids"].clone(),
        "calibration_flat_rows": parent["calibration_flat_rows"].clone(),
        "validation_flat_rows": parent["validation_flat_rows"].clone(),
    }
    validate_context(context, prereg_sha)
    pilot_expected = {
        layer: {
            endpoint: parent["layers"][str(layer)][endpoint]
            .detach()
            .cpu()
            .contiguous()
            for endpoint in ("input", "output")
        }
        for layer in PILOT_LAYERS
    }
    del parent
    gc.collect()
    trim_allocator()

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    layers_root = args.artifact_root / "layers"
    layers_root.mkdir(parents=True, exist_ok=True)
    context_path = args.artifact_root / "context.pt"
    context_descriptor_path = args.artifact_root / "context.json"
    if context_path.exists() != context_descriptor_path.exists():
        raise RuntimeError("partial full-context metadata requires inspection")
    if context_path.exists():
        context_descriptor = json.loads(context_descriptor_path.read_text())
        observed_context = torch.load(
            context_path, map_location="cpu", weights_only=False
        )
        validate_context(observed_context, prereg_sha)
        if (
            not contexts_equal(context, observed_context)
            or context_descriptor.get("schema") != CONTEXT_SCHEMA
            or context_descriptor.get("sha256") != pilot.sha256_file(context_path)
            or context_descriptor.get("bytes") != context_path.stat().st_size
            or context_descriptor.get("preregistration_sha256") != prereg_sha
            or context_descriptor.get("complete") is not True
            or context_descriptor.get("terminal") is not True
        ):
            raise ValueError("full-context metadata descriptor changed")
        del observed_context
        gc.collect()
        trim_allocator()
    else:
        pilot.atomic_torch_save(context, context_path)
        context_descriptor = {
            "schema": CONTEXT_SCHEMA,
            "preregistration_sha256": prereg_sha,
            "bytes": context_path.stat().st_size,
            "sha256": pilot.sha256_file(context_path),
            "complete": True,
            "terminal": True,
        }
        pilot.atomic_json(context_descriptor, context_descriptor_path)

    records = scan_layer_prefix(
        layers_root,
        preregistration_sha256=prereg_sha,
        producer_commit=producer_commit,
    )
    next_layer = len(records)

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
        != tuple(expected_layer_type(index) for index in range(64))
    ):
        raise ValueError("Qwen3.8-27B full-context architecture changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if next_layer == 0:
        hidden = pilot.embedding_rows(
            args.model_dir, weight_map, context["input_ids"]
        )
    else:
        previous = torch.load(
            layers_root / f"{next_layer - 1:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        validate_shard_tensors(
            previous,
            next_layer - 1,
            preregistration_sha256=prereg_sha,
            producer_commit=producer_commit,
        )
        hidden = previous["output"].contiguous()
        del previous
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (12, 64, 5120):
        raise ValueError("full-context resume hidden state changed")

    positions = torch.arange(hidden.shape[1], dtype=torch.long).view(1, 1, -1)
    positions = positions.expand(4, hidden.shape[0], -1)
    if not torch.equal(positions[0], context["text_position_ids"]):
        raise ValueError("full-context position identity changed")
    rotary_positions = positions[1:]
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, rotary_positions)
    attention_mask = context["attention_mask"].to(torch.long)
    masks = {
        "full_attention": create_causal_mask(
            config,
            hidden,
            attention_mask,
            None,
            context["text_position_ids"],
        ),
        "linear_attention": create_recurrent_attention_mask(
            config,
            hidden,
            attention_mask,
            past_key_values=None,
            position_ids=context["text_position_ids"],
        ),
    }

    started = time.monotonic()
    resumed_from_layer = next_layer if next_layer else None
    for layer_index in range(next_layer, 64):
        layer_started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "full_context_layer_started",
                    "layer": layer_index,
                    "type": config.layer_types[layer_index],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        exact_input = hidden.detach().cpu().contiguous()
        if layer_index in PILOT_LAYERS and not torch.equal(
            exact_input, pilot_expected[layer_index]["input"]
        ):
            raise RuntimeError("full-context pilot input reproduction changed")
        state, parameters = pilot.layer_state(
            args.model_dir, weight_map, layer_index
        )
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("full-context layer state mapping changed")
        layer.eval()
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[config.layer_types[layer_index]],
                position_ids=context["text_position_ids"],
                past_key_values=None,
                use_cache=False,
            ).contiguous()
        if hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError("non-finite or changed BF16 full-context state")
        exact_output = hidden.detach().cpu().contiguous()
        if layer_index in PILOT_LAYERS and not torch.equal(
            exact_output, pilot_expected[layer_index]["output"]
        ):
            raise RuntimeError("full-context pilot output reproduction changed")
        shard = {
            "schema": SHARD_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "preregistration_sha256": prereg_sha,
            "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
            "producer_commit": producer_commit,
            "layer": layer_index,
            "type": config.layer_types[layer_index],
            "input": exact_input,
            "output": exact_output,
        }
        validate_shard_tensors(
            shard,
            layer_index,
            preregistration_sha256=prereg_sha,
            producer_commit=producer_commit,
        )
        shard_path = layers_root / f"{layer_index:02d}.pt"
        descriptor_path = layers_root / f"{layer_index:02d}.json"
        pilot.atomic_torch_save(shard, shard_path)
        descriptor = {
            "schema": SHARD_DESCRIPTOR_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "preregistration_sha256": prereg_sha,
            "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
            "producer_commit": producer_commit,
            "layer": layer_index,
            "type": config.layer_types[layer_index],
            "parameters": parameters,
            "input": {
                "shape": list(exact_input.shape),
                "dtype": str(exact_input.dtype),
                "sha256": pilot.tensor_sha256(exact_input),
            },
            "output": {
                "shape": list(exact_output.shape),
                "dtype": str(exact_output.dtype),
                "sha256": pilot.tensor_sha256(exact_output),
            },
            "shard": {
                "path": str(shard_path),
                "bytes": shard_path.stat().st_size,
                "sha256": pilot.sha256_file(shard_path),
            },
            "pilot_bit_exact": layer_index in PILOT_LAYERS,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        pilot.atomic_json(descriptor, descriptor_path)
        validate_descriptor(
            descriptor,
            shard_path,
            layer_index=layer_index,
            preregistration_sha256=prereg_sha,
            producer_commit=producer_commit,
        )
        records.append(descriptor)
        print(
            json.dumps(
                {
                    "event": "full_context_layer_terminal",
                    "layer": layer_index,
                    "shard_bytes": shard_path.stat().st_size,
                    "shard_sha256": descriptor["shard"]["sha256"],
                    "pilot_bit_exact": descriptor["pilot_bit_exact"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del layer, state, exact_input, exact_output, shard
        gc.collect()
        trim_allocator()

    if len(records) != 64 or [record["layer"] for record in records] != list(range(64)):
        raise RuntimeError("full-context terminal layer denominator changed")
    total_shard_bytes = sum(record["shard"]["bytes"] for record in records)
    maximum_individual_shard_bytes = max(
        record["shard"]["bytes"] for record in records
    )
    gate = {
        "required_layers": 64,
        "observed_layers": len(records),
        "full_context_shape_per_endpoint": [12, 64, 5120],
        "all_values_finite": True,
        "pilot_layers_bit_exact": list(PILOT_LAYERS),
        "maximum_allowed_individual_shard_bytes": int(
            prereg["gate"]["maximum_individual_shard_bytes"]
        ),
        "maximum_observed_individual_shard_bytes": maximum_individual_shard_bytes,
        "individual_shards_within_maximum": maximum_individual_shard_bytes
        <= int(prereg["gate"]["maximum_individual_shard_bytes"]),
        "maximum_total_shard_bytes": int(
            prereg["gate"]["maximum_total_shard_bytes"]
        ),
        "total_shard_bytes": total_shard_bytes,
        "storage_within_maximum": total_shard_bytes
        <= int(prereg["gate"]["maximum_total_shard_bytes"]),
    }
    gate["passed"] = bool(
        gate["observed_layers"] == gate["required_layers"]
        and gate["individual_shards_within_maximum"]
        and gate["storage_within_maximum"]
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": producer_commit,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": prereg_sha,
        "download_receipt_sha256": pilot.sha256_file(download_path),
        "index_sha256": pilot.sha256_file(index_path),
        "parent_cache_sha256": EXPECTED_PARENT_CACHE_SHA256,
        "context": {
            "path": str(context_path),
            "bytes": context_path.stat().st_size,
            "sha256": pilot.sha256_file(context_path),
        },
        "layers": records,
        "resumed_from_layer": resumed_from_layer,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "gate": gate,
        "complete": True,
        "terminal": True,
        "claim_boundary": prereg["claim_boundary"],
    }
    pilot.atomic_json(receipt, args.receipt)
    pilot.atomic_json(
        {
            "terminal": True,
            "passed": gate["passed"],
            "receipt_sha256": pilot.sha256_file(args.receipt),
            "layer_count": 64,
            "total_shard_bytes": total_shard_bytes,
        },
        args.completion_marker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
