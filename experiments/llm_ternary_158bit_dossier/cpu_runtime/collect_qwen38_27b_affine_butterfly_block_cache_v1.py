#!/usr/bin/env python3
"""Stream exact BF16 Qwen3.8-27B blocks and cache three complete block pairs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import torch
from safetensors import safe_open


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
TARGET_LAYERS = (0, 32, 63)
TARGET_LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    content = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resource_guard(workspace: Path) -> None:
    if len(Path("/proc/swaps").read_text().splitlines()) != 1:
        raise SystemExit("infrastructure-invalid: swap is active")
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
    }
    if fields.get("MemAvailable", 0) < 12 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 12 GiB memory available")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < 4 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 4 GiB disk free")


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def embedding_rows(
    model_dir: Path, weight_map: dict[str, str], token_ids: torch.Tensor
) -> torch.Tensor:
    name = "model.language_model.embed_tokens.weight"
    rows: dict[int, torch.Tensor] = {}
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
        source = handle.get_slice(name)
        for token_id in sorted(set(int(value) for value in token_ids.flatten())):
            rows[token_id] = source[token_id : token_id + 1]
    return torch.cat([rows[int(value)] for value in token_ids.flatten()]).reshape(
        *token_ids.shape, -1
    )


def selected_flat_rows(
    attention_mask: torch.Tensor, batch_rows: range, maximum: int
) -> torch.Tensor:
    selected = torch.zeros_like(attention_mask, dtype=torch.bool)
    selected[list(batch_rows)] = attention_mask[list(batch_rows)].bool()
    valid = selected.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid.numel() > maximum:
        positions = torch.linspace(0, valid.numel() - 1, maximum, dtype=torch.float64)
        valid = valid.index_select(0, positions.round().long())
    if valid.numel() == 0 or len(set(valid.tolist())) != valid.numel():
        raise RuntimeError("invalid deterministic activation-row selection")
    return valid.contiguous()


def layer_state(
    model_dir: Path, weight_map: dict[str, str], layer_index: int
) -> tuple[dict[str, torch.Tensor], int]:
    prefix = f"model.language_model.layers.{layer_index}."
    names = sorted(name for name in weight_map if name.startswith(prefix))
    state = {
        name.removeprefix(prefix): load_tensor(model_dir, weight_map, name) for name in names
    }
    return state, sum(value.numel() for value in state.values())


def validate_existing(
    cache_path: Path,
    receipt_path: Path,
    preregistration_path: Path,
    index_path: Path,
) -> dict[str, Any] | None:
    if not cache_path.exists() and not receipt_path.exists():
        return None
    if not cache_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("partial affine-butterfly block cache requires inspection")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "qwen38_27b_affine_butterfly_block_cache_receipt_v1"
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("preregistration_sha256") != sha256_file(preregistration_path)
        or receipt.get("index_sha256") != sha256_file(index_path)
        or receipt.get("cache", {}).get("sha256") != sha256_file(cache_path)
        or receipt.get("cache", {}).get("bytes") != cache_path.stat().st_size
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
    ):
        raise ValueError("existing affine-butterfly block cache receipt is invalid")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name(
            "qwen38_27b_affine_butterfly_blocks_preregistration_v1.json"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-affine-butterfly-block-cache-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen block-cache CPU thread count is four")
    resource_guard(Path("/workspace"))

    prereg = json.loads(args.preregistration.read_text())
    scope = prereg["scope"]
    protocol = prereg["activation_protocol"]
    calibration_texts = list(protocol["calibration_texts"])
    validation_texts = list(protocol["validation_texts"])
    if (
        prereg.get("status") != "frozen_before_activation_collection_or_fit"
        or scope.get("active_model_only") != MODEL_ID
        or scope.get("revision") != MODEL_REVISION
        or tuple(scope.get("pilot_layers", [])) != TARGET_LAYERS
        or tuple(scope.get("pilot_layer_types", [])) != TARGET_LAYER_TYPES
        or scope.get("qwen_4b_compute_forbidden") is not True
        or set(calibration_texts) & set(validation_texts)
        or len(calibration_texts) != 8
        or len(validation_texts) != 4
    ):
        raise ValueError("affine-butterfly activation protocol changed")

    download_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    download = json.loads(download_path.read_text())
    index_path = args.model_dir / "model.safetensors.index.json"
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    } or sha256_file(index_path) != EXPECTED_INDEX_SHA256:
        raise ValueError("Qwen3.8-27B source identity changed")
    existing = validate_existing(
        args.cache, args.receipt, args.preregistration, index_path
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
    torch.manual_seed(int(prereg["fit"]["seed"]))
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

    texts = calibration_texts + validation_texts
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
    hidden = embedding_rows(args.model_dir, weight_map, token_ids)
    if hidden.dtype != torch.bfloat16 or hidden.shape != (12, 64, 5120):
        raise ValueError(f"unexpected exact embedding state: {hidden.dtype} {hidden.shape}")

    positions = torch.arange(hidden.shape[1], dtype=torch.long).view(1, 1, -1)
    positions = positions.expand(4, hidden.shape[0], -1)
    text_positions = positions[0]
    rotary_positions = positions[1:]
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, rotary_positions)
    masks = {
        "full_attention": create_causal_mask(
            config,
            hidden,
            attention_mask,
            None,
            text_positions,
        ),
        "linear_attention": create_recurrent_attention_mask(
            config,
            hidden,
            attention_mask,
            past_key_values=None,
            position_ids=text_positions,
        ),
    }
    calibration_rows = selected_flat_rows(
        attention_mask,
        range(0, len(calibration_texts)),
        int(protocol["maximum_calibration_rows_per_layer"]),
    )
    validation_rows = selected_flat_rows(
        attention_mask,
        range(len(calibration_texts), len(texts)),
        int(protocol["maximum_validation_rows_per_layer"]),
    )

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
        state, parameters = layer_state(args.model_dir, weight_map, layer_index)
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"layer {layer_index} state mapping changed: {incompatible}")
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
            raise RuntimeError(f"non-finite or changed BF16 state after layer {layer_index}")
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
        "schema": "qwen38_27b_affine_butterfly_block_cache_v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "preregistration_sha256": sha256_file(args.preregistration),
        "calibration_texts": calibration_texts,
        "validation_texts": validation_texts,
        "input_ids": token_ids,
        "attention_mask": attention_mask.to(torch.int8),
        "text_position_ids": text_positions,
        "calibration_flat_rows": calibration_rows,
        "validation_flat_rows": validation_rows,
        "layers": cached_layers,
    }
    atomic_torch_save(cache, args.cache)
    cache_descriptor = {
        "path": str(args.cache),
        "bytes": args.cache.stat().st_size,
        "sha256": sha256_file(args.cache),
    }
    tensor_receipts = {}
    for layer_index, values in cached_layers.items():
        tensor_receipts[layer_index] = {
            kind: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
            for kind, value in values.items()
            if isinstance(value, torch.Tensor)
        }
    receipt = {
        "schema": "qwen38_27b_affine_butterfly_block_cache_receipt_v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "preregistration": str(args.preregistration),
        "preregistration_sha256": sha256_file(args.preregistration),
        "download_receipt_sha256": sha256_file(download_path),
        "index_sha256": sha256_file(index_path),
        "cache": cache_descriptor,
        "target_layers": list(TARGET_LAYERS),
        "target_layer_types": list(TARGET_LAYER_TYPES),
        "calibration_rows": calibration_rows.numel(),
        "validation_rows": validation_rows.numel(),
        "cached_tensors": tensor_receipts,
        "propagated_layers": layer_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "finite": True,
        "complete": True,
        "terminal": True,
        "claim_boundary": "exact BF16 complete-block activation cache only; no quantization, quality, storage, runtime, or breakthrough result",
    }
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
