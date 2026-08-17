#!/usr/bin/env python3
"""Collect bounded, exact-source Qwen3.8-27B layer-0 MLP activations on CPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import safe_open


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYER_PREFIX = "model.language_model.layers.0."
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def embedding_rows(
    model_dir: Path,
    weight_map: dict[str, str],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    shard = model_dir / weight_map[EMBEDDING_NAME]
    rows: dict[int, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source = handle.get_slice(EMBEDDING_NAME)
        for token_id in sorted(set(int(value) for value in token_ids.flatten())):
            rows[token_id] = source[token_id : token_id + 1]
    return torch.cat([rows[int(value)] for value in token_ids.flatten()], dim=0).reshape(
        *token_ids.shape, -1
    )


def bounded_rows(tensors: list[torch.Tensor], maximum: int) -> torch.Tensor:
    if maximum <= 0 or not tensors:
        raise ValueError("positive maximum and at least one activation tensor are required")
    combined = torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)
    if combined.shape[0] <= maximum:
        return combined.contiguous()
    indices = torch.linspace(0, combined.shape[0] - 1, maximum, dtype=torch.float64).round().long()
    if len(set(int(value) for value in indices)) != maximum:
        raise AssertionError("bounded activation row selection produced duplicates")
    return combined.index_select(0, indices).contiguous()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).with_name(
            "qwen38_27b_layer0_mlp_activation_qat_preregistration_v1.json"
        ),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path("/workspace/quant-runs/qwen38-27b-layer0-mlp-activation-qat-v1/activations.pt"),
    )
    parser.add_argument(
        "--receipt", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json"),
    )
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    cache_contract = prereg["activation_cache"]
    if (
        prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or set(cache_contract["calibration_texts"]) & set(cache_contract["validation_texts"])
    ):
        raise ValueError("Qwen3.8-27B activation-cache preregistration identity is invalid")
    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download.get("model_id") != MODEL_ID or download.get("revision") != MODEL_REVISION:
        raise ValueError("Qwen3.8-27B checkpoint receipt identity mismatch")
    index_path = args.model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, layer_idx=0)

    source_hashes: dict[str, dict[str, Any]] = {}
    layer_state: dict[str, torch.Tensor] = {}
    for name in sorted(weight_map):
        if not name.startswith(LAYER_PREFIX) or ".mlp." in name:
            continue
        tensor = load_tensor(args.model_dir, weight_map, name)
        relative = name.removeprefix(LAYER_PREFIX)
        layer_state[relative] = tensor
        source_hashes[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
            "shard": weight_map[name],
        }
    incompatible = layer.load_state_dict(layer_state, strict=False, assign=True)
    if incompatible.unexpected_keys or any(
        not name.startswith("mlp.") for name in incompatible.missing_keys
    ):
        raise ValueError(
            f"unexpected layer-0 state mapping: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    layer.eval()

    def collect(texts: list[str]) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=int(cache_contract["maximum_sequence_tokens"]),
                return_tensors="pt",
            )
            token_ids = encoded["input_ids"]
            hidden = embedding_rows(args.model_dir, weight_map, token_ids)
            with torch.no_grad():
                normalized = layer.input_layernorm(hidden)
                mixed = layer.linear_attn(
                    hidden_states=normalized,
                    cache_params=None,
                    attention_mask=None,
                )
                result.append(layer.post_attention_layernorm(hidden + mixed).float().cpu())
            del hidden, normalized, mixed
        return result

    torch.set_num_threads(int(prereg["training"]["resource_limits"]["cpu_threads"]))
    torch.manual_seed(int(prereg["training"]["seed"]))
    calibration_inputs = bounded_rows(
        collect(list(cache_contract["calibration_texts"])),
        int(cache_contract["maximum_calibration_rows"]),
    )
    validation_inputs = bounded_rows(
        collect(list(cache_contract["validation_texts"])),
        int(cache_contract["maximum_validation_rows"]),
    )
    del layer, layer_state
    gc.collect()

    gate_name = f"{LAYER_PREFIX}mlp.gate_proj.weight"
    up_name = f"{LAYER_PREFIX}mlp.up_proj.weight"
    gate = load_tensor(args.model_dir, weight_map, gate_name)
    up = load_tensor(args.model_dir, weight_map, up_name)
    source_hashes[gate_name] = {
        "shape": list(gate.shape), "dtype": str(gate.dtype),
        "sha256": tensor_sha256(gate), "shard": weight_map[gate_name],
    }
    source_hashes[up_name] = {
        "shape": list(up.shape), "dtype": str(up.dtype),
        "sha256": tensor_sha256(up), "shard": weight_map[up_name],
    }
    with torch.no_grad():
        down_calibration = functional.silu(
            functional.linear(calibration_inputs, gate.float())
        ) * functional.linear(calibration_inputs, up.float())
        down_validation = functional.silu(
            functional.linear(validation_inputs, gate.float())
        ) * functional.linear(validation_inputs, up.float())
    del gate, up
    gc.collect()

    cache = {
        "schema": "qwen38_27b_layer0_mlp_activation_cache_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preregistration_sha256": sha256_file(args.preregistration),
        "calibration_texts": cache_contract["calibration_texts"],
        "validation_texts": cache_contract["validation_texts"],
        "mlp_input_calibration": calibration_inputs.to(torch.float16),
        "mlp_input_validation": validation_inputs.to(torch.float16),
        "down_input_calibration": down_calibration.to(torch.float16),
        "down_input_validation": down_validation.to(torch.float16),
    }
    atomic_torch_save(cache, args.cache)
    receipt = {
        "schema": "qwen38_27b_layer0_mlp_activation_cache_receipt_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_file(args.preregistration),
        },
        "checkpoint": {
            "path": str(args.model_dir),
            "download_receipt_sha256": sha256_file(args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"),
            "index_sha256": sha256_file(index_path),
        },
        "source_tensors": source_hashes,
        "cache": {
            "path": str(args.cache),
            "sha256": sha256_file(args.cache),
            "bytes": args.cache.stat().st_size,
        },
        "rows": {
            "mlp_calibration": calibration_inputs.shape[0],
            "mlp_validation": validation_inputs.shape[0],
            "down_calibration": down_calibration.shape[0],
            "down_validation": down_validation.shape[0],
        },
        "dimensions": {
            "hidden": calibration_inputs.shape[1],
            "intermediate": down_calibration.shape[1],
        },
        "finite": all(
            bool(torch.isfinite(tensor).all())
            for tensor in (calibration_inputs, validation_inputs, down_calibration, down_validation)
        ),
        "complete": True,
        "claim_boundary": "Qwen3.8-27B layer-0 activation cache only; no training or quality result",
    }
    if not receipt["finite"]:
        raise FloatingPointError("non-finite Qwen3.8-27B activation cache")
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
