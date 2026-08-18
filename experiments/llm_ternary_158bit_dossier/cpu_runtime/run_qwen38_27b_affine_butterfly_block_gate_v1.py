#!/usr/bin/env python3
"""Run the frozen three-complete-block affine-butterfly falsification gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_affine_artifact import (
    load_affine_tensor,
    sha256_file,
    verify_descriptor,
    write_affine_tensor,
)
from clark_quantize.qwen38_affine_butterfly import (
    AffineTernary,
    _direct_symmetric,
    apply_givens_product,
    deterministic_givens_pairings,
    fit_shared_affine_rotation,
)
from clark_quantize.qwen38_affine_codebook import compress_affine_parameters


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
TARGET_LAYERS = (0, 32, 63)
GROUP_SIZE = 128


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_bytes(path, canonical_json(payload))


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
    if stat.f_bavail * stat.f_frsize < 6 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 6 GiB disk free")


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def layer_state(
    model_dir: Path, weight_map: dict[str, str], layer_index: int
) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_index}."
    names = sorted(name for name in weight_map if name.startswith(prefix))
    return {
        name.removeprefix(prefix): load_tensor(model_dir, weight_map, name)
        for name in names
    }


def site_layout(layer_type: str) -> dict[str, tuple[str, ...]]:
    if layer_type == "linear_attention":
        attention = (
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
        )
        out = ("linear_attn.out_proj.weight",)
    elif layer_type == "full_attention":
        attention = (
            "self_attn.k_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        )
        out = ("self_attn.o_proj.weight",)
    else:
        raise ValueError(f"unsupported Qwen3.8 layer type {layer_type!r}")
    return {
        "attention_input": attention,
        "attention_output": out,
        "mlp_input": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "mlp_down": ("mlp.down_proj.weight",),
    }


def capture_modules(layer_type: str) -> dict[str, str]:
    if layer_type == "linear_attention":
        result = {
            "attention_input": "linear_attn.in_proj_qkv",
            "attention_output": "linear_attn.out_proj",
            "mlp_input": "mlp.gate_proj",
            "mlp_down": "mlp.down_proj",
            "linear_qkv_output": "linear_attn.in_proj_qkv",
        }
    else:
        result = {
            "attention_input": "self_attn.q_proj",
            "attention_output": "self_attn.o_proj",
            "mlp_input": "mlp.gate_proj",
            "mlp_down": "mlp.down_proj",
        }
    return result


def pairing_seed(base_seed: int, layer_index: int, site: str) -> int:
    content = f"{MODEL_REVISION}:{base_seed}:{layer_index}:{site}".encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little") % (2**63 - 1)


def instantiate_layer(config: Any, layer_index: int, state: Mapping[str, torch.Tensor]) -> Any:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, layer_idx=layer_index)
    incompatible = layer.load_state_dict(dict(state), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"layer {layer_index} state mapping changed: {incompatible}")
    layer.eval()
    return layer


def forward_layer(
    layer: Any,
    hidden: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        ).contiguous()


def capture_parent_sites(
    layer: Any,
    layer_type: str,
    hidden: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for site, module_name in capture_modules(layer_type).items():
        module = layer.get_submodule(module_name)
        if site == "linear_qkv_output":
            def output_hook(_module: Any, _inputs: Any, output: torch.Tensor, key: str = site) -> None:
                captured[key] = output.detach().cpu().contiguous()

            handles.append(module.register_forward_hook(output_hook))
        else:
            def input_hook(_module: Any, inputs: Any, key: str = site) -> None:
                captured[key] = inputs[0].detach().cpu().contiguous()

            handles.append(module.register_forward_pre_hook(input_hook))
    try:
        output = forward_layer(
            layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    finally:
        for handle in handles:
            handle.remove()
    expected = set(capture_modules(layer_type))
    if set(captured) != expected:
        raise RuntimeError(f"complete activation-site capture failed: {set(captured)}")
    return output, captured


@torch.no_grad()
def fit_depthwise_conv_affine(
    weight: torch.Tensor, windows: torch.Tensor
) -> AffineTernary:
    """Fit each depthwise convolution row under its own exact input windows."""
    if (
        weight.ndim != 2
        or windows.ndim != 3
        or windows.shape[0] != weight.shape[0]
        or windows.shape[2] != weight.shape[1]
        or windows.shape[1] == 0
    ):
        raise ValueError("depthwise affine activation dimensions changed")
    source = weight.float().contiguous()
    x = windows.float().contiguous()
    energy = x.square().sum(1)
    denominator = energy.sum(-1).clamp_min(1e-12)
    center = (source * energy).sum(-1) / denominator
    centered = source - center[:, None]
    radius = torch.sqrt((centered.square() * energy).sum(-1) / denominator)
    threshold = radius[:, None] * 0.5
    codes = torch.where(
        centered > threshold,
        torch.ones_like(source, dtype=torch.int8),
        torch.where(
            centered < -threshold,
            -torch.ones_like(source, dtype=torch.int8),
            torch.zeros_like(source, dtype=torch.int8),
        ),
    )

    target = (x * source[:, None, :]).sum(-1)
    mu = center
    alpha = radius

    def solve() -> torch.Tensor:
        nonlocal mu, alpha, codes
        sums = x.sum(-1)
        projection = (x * codes.float()[:, None, :]).sum(-1)
        ss = sums.square().sum(-1)
        sz = (sums * projection).sum(-1)
        zz = projection.square().sum(-1)
        sy = (sums * target).sum(-1)
        zy = (projection * target).sum(-1)
        determinant = ss * zz - sz.square()
        safe = determinant.abs() > 1e-12
        mu = torch.where(
            safe,
            (sy * zz - zy * sz) / torch.where(safe, determinant, torch.ones_like(determinant)),
            sy / ss.clamp_min(1e-12),
        )
        alpha = torch.where(
            safe,
            (ss * zy - sz * sy) / torch.where(safe, determinant, torch.ones_like(determinant)),
            torch.zeros_like(zy),
        )
        negative = alpha < 0
        codes[negative] *= -1
        alpha = alpha.abs()
        return target - sums * mu[:, None] - (
            (x * codes.float()[:, None, :]).sum(-1) * alpha[:, None]
        )

    residual = solve()
    trits = torch.tensor([-1.0, 0.0, 1.0], device=source.device)
    for column in range(source.shape[1]):
        column_x = x[:, :, column]
        current = codes[:, column].float()
        residual_without = residual + column_x * (alpha * current)[:, None]
        correlation = (residual_without * column_x).sum(-1)
        column_energy = column_x.square().sum(-1)
        amplitudes = alpha[:, None] * trits[None, :]
        score = amplitudes.square() * column_energy[:, None] - 2 * amplitudes * correlation[:, None]
        selected = score.argmin(-1).to(torch.int8) - 1
        selected = torch.where(alpha > 0, selected, torch.zeros_like(selected))
        codes[:, column] = selected
        residual = residual_without - column_x * (alpha * selected.float())[:, None]
    solve()
    return AffineTernary(
        codes=codes,
        mu=mu[:, None].contiguous(),
        alpha=alpha[:, None].contiguous(),
        group_size=GROUP_SIZE,
    )


def conv_windows(qkv_output: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    mixed = qkv_output.transpose(1, 2).float()
    windows = torch.nn.functional.pad(mixed, (3, 0)).unfold(-1, 4, 1)
    selected = windows.permute(0, 2, 1, 3).reshape(-1, windows.shape[1], 4)
    return selected.index_select(0, row_indices).permute(1, 0, 2).contiguous()


def write_angles(root: Path, layer_index: int, site: str, angles: torch.Tensor) -> dict[str, Any]:
    relative = Path("layers") / str(layer_index) / "sites" / site / "angles.f16"
    content = angles.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
    atomic_bytes(root / relative, content)
    restored = torch.from_numpy(
        np.frombuffer((root / relative).read_bytes(), dtype="<f2").astype(np.float32)
    ).reshape(angles.shape)
    if not torch.equal(restored, angles.cpu()):
        raise AssertionError("stored FP16 Givens angles changed after decode")
    return {
        "path": relative.as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(angles.shape),
        "dtype": "float16",
    }


def read_angles(root: Path, descriptor: dict[str, Any]) -> torch.Tensor:
    path = root / descriptor["path"]
    content = path.read_bytes()
    if (
        len(content) != descriptor["bytes"]
        or hashlib.sha256(content).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("stored affine angles failed authentication")
    return torch.from_numpy(np.frombuffer(content, dtype="<f2").astype(np.float32)).reshape(
        descriptor["shape"]
    )


def fit_site(
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    layer_index: int,
    site: str,
    tensor_names: tuple[str, ...],
    state: Mapping[str, torch.Tensor],
    calibration: torch.Tensor,
    validation: torch.Tensor,
    fit_config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = artifact_root / "layers" / str(layer_index) / "sites" / site / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("schema") != "qwen38_affine_butterfly_site_v1"
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != preregistration_sha256
            or existing.get("layer") != layer_index
            or existing.get("site") != site
            or tuple(existing.get("tensor_names", [])) != tensor_names
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError(f"partial or changed affine site requires inspection: {site}")
        for descriptor in existing["tensors"]:
            verify_descriptor(artifact_root, descriptor)
        read_angles(artifact_root, existing["angles"])
        return existing

    dimension = state[tensor_names[0]].shape[1]
    seed = pairing_seed(int(fit_config["seed"]), layer_index, site)
    pairings = deterministic_givens_pairings(
        dimension, stages=int(fit_config["rotation_stages"]), seed=seed
    )
    fitted = fit_shared_affine_rotation(
        {name: state[name] for name in tensor_names},
        calibration,
        validation,
        pairings=pairings,
        steps=int(fit_config["rotation_steps"]),
        learning_rate=float(fit_config["rotation_learning_rate"]),
        row_shard=int(fit_config["row_shard"]),
        group_size=int(fit_config["group_size"]),
    )
    angles = write_angles(artifact_root, layer_index, site, fitted.angles)
    tensors = []
    for name in tensor_names:
        compressed = compress_affine_parameters(fitted.quantized[name])
        tensors.append(
            write_affine_tensor(
                artifact_root, f"layers.{layer_index}.{name}", compressed
            )
        )
    manifest = {
        "schema": "qwen38_affine_butterfly_site_v1",
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "layer": layer_index,
        "site": site,
        "tensor_names": list(tensor_names),
        "input_dimension": dimension,
        "pairing_seed": seed,
        "angles": angles,
        "tensors": tensors,
        "fit_metrics_before_parameter_codebooks": fitted.metrics,
        "actual_storage_bytes": angles["bytes"]
        + sum(item["storage_bytes"] for item in tensors),
        "complete": True,
        "terminal": True,
    }
    atomic_json(manifest, manifest_path)
    return manifest


def fit_conv(
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    layer_index: int,
    state: Mapping[str, torch.Tensor],
    windows: torch.Tensor,
) -> dict[str, Any]:
    name = "linear_attn.conv1d.weight"
    manifest_path = artifact_root / "layers" / str(layer_index) / "sites" / "depthwise_conv" / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != preregistration_sha256
            or existing.get("complete") is not True
        ):
            raise ValueError("partial or changed affine convolution requires inspection")
        verify_descriptor(artifact_root, existing["tensor"])
        return existing
    shape = tuple(state[name].shape)
    fitted = fit_depthwise_conv_affine(state[name].reshape(shape[0], -1), windows)
    compressed = compress_affine_parameters(fitted)
    descriptor = write_affine_tensor(
        artifact_root, f"layers.{layer_index}.{name}", compressed
    )
    manifest = {
        "schema": "qwen38_affine_depthwise_conv_site_v1",
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "layer": layer_index,
        "site": "depthwise_conv",
        "tensor_name": name,
        "source_shape": list(shape),
        "tensor": descriptor,
        "actual_storage_bytes": descriptor["storage_bytes"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(manifest, manifest_path)
    return manifest


def replace_weight(layer: Any, name: str, value: torch.Tensor) -> None:
    module_name, parameter_name = name.rsplit(".", 1)
    if parameter_name != "weight":
        raise ValueError(f"only matrix weights are replaceable: {name}")
    module = layer.get_submodule(module_name)
    module.weight = torch.nn.Parameter(value.contiguous(), requires_grad=False)


def direct_weight(source: torch.Tensor, *, row_shard: int) -> torch.Tensor:
    shape = source.shape
    matrix = source.reshape(shape[0], -1)
    result = torch.empty_like(matrix)
    for start in range(0, matrix.shape[0], row_shard):
        stop = min(start + row_shard, matrix.shape[0])
        result[start:stop] = _direct_symmetric(
            matrix[start:stop].float(), GROUP_SIZE
        ).to(source.dtype)
    return result.reshape(shape)


def materialize_candidate(
    payload: AffineTernary,
    *,
    dtype: torch.dtype,
    angles: torch.Tensor | None,
    pairings: torch.Tensor | None,
    row_shard: int,
    source_shape: tuple[int, ...],
) -> torch.Tensor:
    result = torch.empty(payload.codes.shape, dtype=dtype)
    for start in range(0, payload.codes.shape[0], row_shard):
        stop = min(start + row_shard, payload.codes.shape[0])
        shard = AffineTernary(
            codes=payload.codes[start:stop],
            mu=payload.mu[start:stop],
            alpha=payload.alpha[start:stop],
            group_size=payload.group_size,
        ).reconstruct()
        if angles is not None:
            if pairings is None:
                raise AssertionError("candidate rotation pairings are missing")
            shard = apply_givens_product(shard, angles, pairings, inverse=True)
        result[start:stop] = shard.to(dtype)
    return result.reshape(source_shape)


def validation_nmse(
    observed: torch.Tensor,
    parent: torch.Tensor,
    validation_rows: torch.Tensor,
) -> float:
    selected_observed = observed.reshape(-1, observed.shape[-1]).index_select(
        0, validation_rows
    ).float()
    selected_parent = parent.reshape(-1, parent.shape[-1]).index_select(
        0, validation_rows
    ).float()
    energy = float(selected_parent.square().sum())
    if energy <= 0:
        raise ValueError("complete-block parent output energy is zero")
    return float((selected_observed - selected_parent).square().sum()) / energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"),
    )
    parser.add_argument(
        "--cache-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-affine-butterfly-block-cache-v1.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-affine-butterfly-block-gate-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen affine block gate uses four CPU threads")
    resource_guard(Path("/workspace"))
    torch.set_num_threads(args.threads)

    runtime = Path(__file__).resolve().parent
    preregistration_path = runtime / "qwen38_27b_affine_butterfly_blocks_preregistration_v1.json"
    prereg = json.loads(preregistration_path.read_text())
    preregistration_sha256 = sha256_file(preregistration_path)
    fit_config = prereg["fit"]
    gate_config = prereg["falsification_gate"]
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.receipt.exists():
        existing = json.loads(args.receipt.read_text())
        if (
            existing.get("schema") != "qwen38_27b_affine_butterfly_block_gate_receipt_v1"
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != preregistration_sha256
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing affine block-gate receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    index_path = args.model_dir / "model.safetensors.index.json"
    cache_receipt = json.loads(args.cache_receipt.read_text())
    if (
        prereg.get("status") != "frozen_before_activation_collection_or_fit"
        or prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or tuple(prereg["scope"]["pilot_layers"]) != TARGET_LAYERS
        or fit_config["group_size"] != GROUP_SIZE
        or sha256_file(index_path) != INDEX_SHA256
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("terminal") is not True
        or cache_receipt.get("cache", {}).get("sha256") != sha256_file(args.cache)
        or cache_receipt.get("preregistration_sha256") != preregistration_sha256
    ):
        raise ValueError("affine block-gate source identity changed")

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    attention_mask = cache["attention_mask"].long()
    position_ids = cache["text_position_ids"]
    calibration_rows = cache["calibration_flat_rows"].long()
    validation_rows = cache["validation_flat_rows"].long()
    representative = cache["layers"]["0"]["input"]
    positions = position_ids.unsqueeze(0).expand(3, -1, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(representative, positions)
    masks = {
        "full_attention": create_causal_mask(
            config, representative, attention_mask, None, position_ids
        ),
        "linear_attention": create_recurrent_attention_mask(
            config,
            representative,
            attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        ),
    }
    weight_map = json.loads(index_path.read_text())["weight_map"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    layer_results = []
    total_actual_storage = 0

    for layer_index in TARGET_LAYERS:
        layer_started = time.monotonic()
        layer_type = config.layer_types[layer_index]
        state = layer_state(args.model_dir, weight_map, layer_index)
        hidden = cache["layers"][str(layer_index)]["input"]
        parent = cache["layers"][str(layer_index)]["output"]
        exact_layer = instantiate_layer(config, layer_index, state)
        replayed, captured = capture_parent_sites(
            exact_layer,
            layer_type,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
        )
        if not torch.equal(replayed, parent):
            raise ValueError(f"exact BF16 block replay changed at layer {layer_index}")
        del exact_layer, replayed

        sites = []
        for site, tensor_names in site_layout(layer_type).items():
            values = captured[site].reshape(-1, captured[site].shape[-1])
            manifest = fit_site(
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=preregistration_sha256,
                layer_index=layer_index,
                site=site,
                tensor_names=tensor_names,
                state=state,
                calibration=values.index_select(0, calibration_rows),
                validation=values.index_select(0, validation_rows),
                fit_config=fit_config,
            )
            sites.append(manifest)
        conv = None
        if layer_type == "linear_attention":
            windows = conv_windows(captured["linear_qkv_output"], calibration_rows)
            conv = fit_conv(
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=preregistration_sha256,
                layer_index=layer_index,
                state=state,
                windows=windows,
            )
            del windows
        captured.clear()
        gc.collect()

        baseline_layer = instantiate_layer(config, layer_index, state)
        quantized_names = sorted(name for name, value in state.items() if value.ndim >= 2)
        for name in quantized_names:
            replace_weight(
                baseline_layer,
                name,
                direct_weight(state[name], row_shard=int(fit_config["row_shard"])),
            )
        baseline_output = forward_layer(
            baseline_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
        )
        direct_nmse = validation_nmse(baseline_output, parent, validation_rows)
        del baseline_layer, baseline_output
        gc.collect()

        candidate_layer = instantiate_layer(config, layer_index, state)
        covered: set[str] = set()
        for manifest in sites:
            angles = read_angles(args.artifact_root, manifest["angles"])
            pairings = deterministic_givens_pairings(
                int(manifest["input_dimension"]),
                stages=int(fit_config["rotation_stages"]),
                seed=int(manifest["pairing_seed"]),
            )
            for name, descriptor in zip(manifest["tensor_names"], manifest["tensors"]):
                payload = load_affine_tensor(args.artifact_root, descriptor)
                replace_weight(
                    candidate_layer,
                    name,
                    materialize_candidate(
                        payload,
                        dtype=state[name].dtype,
                        angles=angles,
                        pairings=pairings,
                        row_shard=int(fit_config["row_shard"]),
                        source_shape=tuple(state[name].shape),
                    ),
                )
                covered.add(name)
        if conv is not None:
            name = conv["tensor_name"]
            payload = load_affine_tensor(args.artifact_root, conv["tensor"])
            replace_weight(
                candidate_layer,
                name,
                materialize_candidate(
                    payload,
                    dtype=state[name].dtype,
                    angles=None,
                    pairings=None,
                    row_shard=int(fit_config["row_shard"]),
                    source_shape=tuple(state[name].shape),
                ),
            )
            covered.add(name)
        if covered != set(quantized_names):
            raise RuntimeError(
                f"candidate layer {layer_index} coverage changed: missing {set(quantized_names) - covered}"
            )
        candidate_output = forward_layer(
            candidate_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
        )
        candidate_nmse = validation_nmse(candidate_output, parent, validation_rows)
        del candidate_layer, candidate_output
        gc.collect()

        reduction = (direct_nmse - candidate_nmse) / max(direct_nmse, 1e-30)
        actual_storage = sum(site["actual_storage_bytes"] for site in sites)
        if conv is not None:
            actual_storage += conv["actual_storage_bytes"]
        total_actual_storage += actual_storage
        result = {
            "layer": layer_index,
            "type": layer_type,
            "rank2_or_higher_tensor_count": len(quantized_names),
            "rank2_or_higher_parameter_count": sum(state[name].numel() for name in quantized_names),
            "direct_complete_block_validation_nmse": direct_nmse,
            "candidate_complete_block_validation_nmse": candidate_nmse,
            "relative_validation_nmse_reduction_vs_direct": reduction,
            "minimum_required_reduction": gate_config[
                "minimum_relative_validation_nmse_reduction_vs_direct_ternary"
            ],
            "passed": reduction
            >= gate_config["minimum_relative_validation_nmse_reduction_vs_direct_ternary"],
            "actual_packed_tensor_and_rotation_bytes": actual_storage,
            "elapsed_seconds": time.monotonic() - layer_started,
        }
        layer_results.append(result)
        layer_result_path = args.artifact_root / "layers" / str(layer_index) / "terminal-result.json"
        atomic_json(result, layer_result_path)
        print(json.dumps(result, sort_keys=True), flush=True)
        del state
        gc.collect()

    projected = prereg["representation"]["complete_text_projection"]
    gate = {
        "minimum_relative_validation_nmse_reduction_vs_direct_ternary": gate_config[
            "minimum_relative_validation_nmse_reduction_vs_direct_ternary"
        ],
        "maximum_projected_complete_text_file_bpw": gate_config[
            "maximum_projected_complete_text_file_bpw"
        ],
        "observed_projected_complete_text_file_bpw": projected[
            "maximum_projected_complete_bundle_bpw"
        ],
        "all_selected_blocks_passed": all(item["passed"] for item in layer_results),
        "all_payload_pack_unpack_roundtrips_exact": True,
        "all_values_finite": all(
            np.isfinite(item["direct_complete_block_validation_nmse"])
            and np.isfinite(item["candidate_complete_block_validation_nmse"])
            and np.isfinite(item["relative_validation_nmse_reduction_vs_direct"])
            for item in layer_results
        ),
    }
    gate["passed"] = (
        gate["all_selected_blocks_passed"]
        and gate["all_payload_pack_unpack_roundtrips_exact"]
        and gate["all_values_finite"]
        and gate["observed_projected_complete_text_file_bpw"]
        <= gate["maximum_projected_complete_text_file_bpw"]
    )
    receipt = {
        "schema": "qwen38_27b_affine_butterfly_block_gate_receipt_v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "cache_receipt_sha256": sha256_file(args.cache_receipt),
        "cache_sha256": sha256_file(args.cache),
        "layers": layer_results,
        "actual_selected_block_tensor_and_rotation_bytes": total_actual_storage,
        "projected_complete_text_bundle": projected,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "terminal three-complete-block falsification result only; not a whole-model artifact, capability evaluation, CPU runtime result, or breakthrough",
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
