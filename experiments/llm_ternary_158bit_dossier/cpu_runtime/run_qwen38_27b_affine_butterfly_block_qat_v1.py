#!/usr/bin/env python3
"""Sensitivity-aware complete-block QAT for the stored Qwen3.8 affine design."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
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
RUNTIME = Path(__file__).resolve().parent
PARENT_RUNNER_PATH = RUNTIME / "run_qwen38_27b_affine_butterfly_block_gate_v1.py"
PARENT_SPEC = importlib.util.spec_from_file_location("qwen38_affine_ptq_parent", PARENT_RUNNER_PATH)
if PARENT_SPEC is None or PARENT_SPEC.loader is None:
    raise RuntimeError("unable to load affine PTQ parent runner")
PARENT = importlib.util.module_from_spec(PARENT_SPEC)
PARENT_SPEC.loader.exec_module(PARENT)

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_affine_artifact import (
    load_compressed_affine_tensor,
    sha256_file,
    verify_descriptor,
    write_affine_tensor,
)
from clark_quantize.qwen38_affine_butterfly import deterministic_givens_pairings
from clark_quantize.qwen38_affine_qat import (
    StoredAffineConvWeight,
    StoredAffineLinear,
    TrainableAffinePayload,
    TrainableStoredRotation,
    qat_parameter_groups,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
PARENT_RUNNER_COMMIT = "1e524a9eb48e223b259a4d3714afc765bb107dcd"
TARGET_LAYERS = (0, 32, 63)


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
    if fields.get("MemAvailable", 0) < 18 * 1024 * 1024:
        raise SystemExit("infrastructure-invalid: less than 18 GiB memory available")
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < 8 * 1024**3:
        raise SystemExit("infrastructure-invalid: less than 8 GiB disk free")


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def layer_state(
    model_dir: Path, weight_map: dict[str, str], layer_index: int
) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_index}."
    return {
        name.removeprefix(prefix): load_tensor(model_dir, weight_map, name)
        for name in sorted(name for name in weight_map if name.startswith(prefix))
    }


def set_child_module(root: torch.nn.Module, path: str, child: torch.nn.Module) -> None:
    if "." in path:
        parent_path, name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent, name = root, path
    setattr(parent, name, child)


def make_context(
    config: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    positions = position_ids.unsqueeze(0).expand(3, -1, -1)
    position_embeddings = Qwen3_5TextRotaryEmbedding(config)(hidden, positions)
    masks = {
        "full_attention": create_causal_mask(
            config, hidden, attention_mask, None, position_ids
        ),
        "linear_attention": create_recurrent_attention_mask(
            config,
            hidden,
            attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        ),
    }
    return position_embeddings, masks


def forward_with_grad(
    layer: Any,
    hidden: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    return layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    ).contiguous()


def selected_nmse(
    observed: torch.Tensor, parent: torch.Tensor, rows: torch.Tensor
) -> float:
    selected_observed = observed.reshape(-1, observed.shape[-1]).index_select(0, rows).float()
    selected_parent = parent.reshape(-1, parent.shape[-1]).index_select(0, rows).float()
    denominator = float(selected_parent.square().sum())
    if denominator <= 0:
        raise ValueError("QAT parent block-output energy is zero")
    return float((selected_observed - selected_parent).square().sum()) / denominator


def parent_site_manifests(
    artifact_root: Path, layer_index: int, layer_type: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    sites: dict[str, dict[str, Any]] = {}
    for site in PARENT.site_layout(layer_type):
        path = artifact_root / "layers" / str(layer_index) / "sites" / site / "manifest.json"
        manifest = json.loads(path.read_text())
        if (
            manifest.get("schema") != "qwen38_affine_butterfly_site_v1"
            or manifest.get("repository_commit") != PARENT_RUNNER_COMMIT
            or manifest.get("layer") != layer_index
            or manifest.get("site") != site
            or manifest.get("complete") is not True
            or manifest.get("terminal") is not True
        ):
            raise ValueError(f"QAT parent site is incomplete or changed: {layer_index}/{site}")
        for descriptor in manifest["tensors"]:
            verify_descriptor(artifact_root, descriptor)
        PARENT.read_angles(artifact_root, manifest["angles"])
        sites[site] = manifest
    conv = None
    if layer_type == "linear_attention":
        path = artifact_root / "layers" / str(layer_index) / "sites/depthwise_conv/manifest.json"
        conv = json.loads(path.read_text())
        if (
            conv.get("schema") != "qwen38_affine_depthwise_conv_site_v1"
            or conv.get("repository_commit") != PARENT_RUNNER_COMMIT
            or conv.get("layer") != layer_index
            or conv.get("complete") is not True
            or conv.get("terminal") is not True
        ):
            raise ValueError(f"QAT parent convolution is incomplete or changed: {layer_index}")
        verify_descriptor(artifact_root, conv["tensor"])
    return sites, conv


@dataclass
class TrainableBlock:
    layer: torch.nn.Module
    payloads: dict[str, TrainableAffinePayload]
    rotations: dict[str, TrainableStoredRotation]
    names_by_site: dict[str, tuple[str, ...]]
    source_shapes: dict[str, tuple[int, ...]]


def build_trainable_block(
    *,
    config: Any,
    layer_index: int,
    state: Mapping[str, torch.Tensor],
    parent_artifact_root: Path,
    selected_sites: set[str] | None = None,
) -> TrainableBlock:
    layer_type = config.layer_types[layer_index]
    sites, conv = parent_site_manifests(parent_artifact_root, layer_index, layer_type)
    layer = PARENT.instantiate_layer(config, layer_index, state)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    payloads: dict[str, TrainableAffinePayload] = {}
    rotations: dict[str, TrainableStoredRotation] = {}
    names_by_site: dict[str, tuple[str, ...]] = {}
    source_shapes: dict[str, tuple[int, ...]] = {}
    allowed = set(sites) | ({"depthwise_conv"} if conv is not None else set())
    if selected_sites is not None and not selected_sites <= allowed:
        raise ValueError("QAT selected site set changed")

    for site, manifest in sites.items():
        if selected_sites is not None and site not in selected_sites:
            continue
        angles = PARENT.read_angles(parent_artifact_root, manifest["angles"])
        pairings = deterministic_givens_pairings(
            int(manifest["input_dimension"]),
            stages=angles.shape[0],
            seed=int(manifest["pairing_seed"]),
        )
        rotation = TrainableStoredRotation(angles, pairings)
        rotations[site] = rotation
        tensor_names = tuple(manifest["tensor_names"])
        names_by_site[site] = tensor_names
        for name, descriptor in zip(tensor_names, manifest["tensors"]):
            payload = TrainableAffinePayload(
                load_compressed_affine_tensor(parent_artifact_root, descriptor)
            )
            payloads[name] = payload
            source_shapes[name] = tuple(state[name].shape)
            module_name, parameter_name = name.rsplit(".", 1)
            if parameter_name != "weight":
                raise ValueError(f"QAT parent tensor is not a weight: {name}")
            set_child_module(layer, module_name, StoredAffineLinear(payload, rotation))

    if conv is not None and (selected_sites is None or "depthwise_conv" in selected_sites):
        name = conv["tensor_name"]
        payload = TrainableAffinePayload(
            load_compressed_affine_tensor(parent_artifact_root, conv["tensor"])
        )
        payloads[name] = payload
        source_shapes[name] = tuple(state[name].shape)
        names_by_site["depthwise_conv"] = (name,)
        set_child_module(
            layer,
            "linear_attn.conv1d",
            StoredAffineConvWeight(payload, tuple(state[name].shape)),
        )

    expected = {
        name for site, names in PARENT.site_layout(layer_type).items()
        if selected_sites is None or site in selected_sites
        for name in names
    }
    if conv is not None and (selected_sites is None or "depthwise_conv" in selected_sites):
        expected.add(conv["tensor_name"])
    if set(payloads) != expected:
        raise RuntimeError(f"QAT block payload coverage changed: {expected - set(payloads)}")
    layer.eval()
    return TrainableBlock(
        layer=layer,
        payloads=payloads,
        rotations=rotations,
        names_by_site=names_by_site,
        source_shapes=source_shapes,
    )


def score_block(
    block: TrainableBlock,
    hidden: torch.Tensor,
    parent: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    calibration_rows: torch.Tensor,
    validation_rows: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        output = forward_with_grad(
            block.layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    return {
        "calibration_complete_block_nmse": selected_nmse(
            output, parent, calibration_rows
        ),
        "validation_complete_block_nmse": selected_nmse(
            output, parent, validation_rows
        ),
    }


def sensitivity_report(
    *,
    config: Any,
    layer_index: int,
    state: Mapping[str, torch.Tensor],
    parent_artifact_root: Path,
    hidden: torch.Tensor,
    parent: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    calibration_rows: torch.Tensor,
    validation_rows: torch.Tensor,
) -> list[dict[str, Any]]:
    layer_type = config.layer_types[layer_index]
    sites = list(PARENT.site_layout(layer_type))
    if layer_type == "linear_attention":
        sites.append("depthwise_conv")
    records = []
    for site in sites:
        block = build_trainable_block(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=parent_artifact_root,
            selected_sites={site},
        )
        metrics = score_block(
            block,
            hidden,
            parent,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            calibration_rows=calibration_rows,
            validation_rows=validation_rows,
        )
        records.append({"site": site, **metrics})
        print(
            json.dumps(
                {"event": "sensitivity_site_terminal", "layer": layer_index, **records[-1]},
                sort_keys=True,
            ),
            flush=True,
        )
        del block
        gc.collect()
    calibration_order = {
        item["site"]: rank + 1
        for rank, item in enumerate(
            sorted(
                records,
                key=lambda item: item["calibration_complete_block_nmse"],
                reverse=True,
            )
        )
    }
    for item in records:
        item["calibration_sensitivity_rank"] = calibration_order[item["site"]]
    return records


def export_trained_block(
    *,
    block: TrainableBlock,
    artifact_root: Path,
    layer_index: int,
    repository_commit: str,
    preregistration_sha256: str,
    parent_receipt_sha256: str,
) -> tuple[list[dict[str, Any]], int]:
    manifests = []
    storage = 0
    for site, names in block.names_by_site.items():
        site_root = artifact_root / "layers" / str(layer_index) / "sites" / site
        tensors = []
        for name in names:
            compressed = block.payloads[name].export_compressed()
            descriptor = write_affine_tensor(
                artifact_root,
                f"layers.{layer_index}.{name}",
                compressed,
            )
            verify_descriptor(artifact_root, descriptor)
            decoded = load_compressed_affine_tensor(artifact_root, descriptor)
            if (
                not torch.equal(decoded.payload.codes, compressed.payload.codes)
                or not np.array_equal(decoded.alpha.indices, compressed.alpha.indices)
                or not np.array_equal(decoded.mu.indices, compressed.mu.indices)
                or not np.array_equal(decoded.alpha.reconstructed, compressed.alpha.reconstructed)
                or not np.array_equal(decoded.mu.reconstructed, compressed.mu.reconstructed)
            ):
                raise AssertionError(f"QAT packed export changed stored state for {name}")
            tensors.append(descriptor)
        angle_descriptor = None
        if site != "depthwise_conv":
            angle_descriptor = PARENT.write_angles(
                artifact_root,
                layer_index,
                site,
                block.rotations[site].export_angles(),
            )
        actual_storage = sum(item["storage_bytes"] for item in tensors) + (
            angle_descriptor["bytes"] if angle_descriptor is not None else 0
        )
        manifest = {
            "schema": "qwen38_affine_butterfly_qat_site_v1",
            "repository_commit": repository_commit,
            "preregistration_sha256": preregistration_sha256,
            "parent_receipt_sha256": parent_receipt_sha256,
            "layer": layer_index,
            "site": site,
            "tensor_names": list(names),
            "tensors": tensors,
            "angles": angle_descriptor,
            "actual_storage_bytes": actual_storage,
            "complete": True,
            "terminal": True,
        }
        atomic_json(manifest, site_root / "manifest.json")
        manifests.append(manifest)
        storage += actual_storage
    return manifests, storage


def validate_terminal_layer(
    *,
    result: dict[str, Any],
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    parent_receipt_sha256: str,
) -> None:
    if (
        result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("parent_receipt_sha256") != parent_receipt_sha256
        or result.get("terminal") is not True
        or result.get("complete") is not True
    ):
        raise ValueError("QAT terminal layer identity changed")
    for relative in result["site_manifests"]:
        manifest = json.loads((artifact_root / relative).read_text())
        if (
            manifest.get("schema") != "qwen38_affine_butterfly_qat_site_v1"
            or manifest.get("repository_commit") != repository_commit
            or manifest.get("preregistration_sha256") != preregistration_sha256
            or manifest.get("parent_receipt_sha256") != parent_receipt_sha256
            or manifest.get("complete") is not True
            or manifest.get("terminal") is not True
        ):
            raise ValueError("QAT terminal site identity changed")
        for descriptor in manifest["tensors"]:
            verify_descriptor(artifact_root, descriptor)
            load_compressed_affine_tensor(artifact_root, descriptor)
        if manifest["angles"] is not None:
            PARENT.read_angles(artifact_root, manifest["angles"])


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
        "--parent-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"),
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-affine-butterfly-block-gate-v1.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-block-qat-v1/artifact"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-affine-butterfly-block-qat-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen QAT gate uses four CPU threads")
    resource_guard(Path("/workspace"))
    torch.set_num_threads(args.threads)

    preregistration_path = RUNTIME / "qwen38_27b_affine_butterfly_block_qat_preregistration_v1.json"
    prereg = json.loads(preregistration_path.read_text())
    preregistration_sha256 = sha256_file(preregistration_path)
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.receipt.exists():
        existing = json.loads(args.receipt.read_text())
        if (
            existing.get("schema") != "qwen38_27b_affine_butterfly_block_qat_receipt_v1"
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != preregistration_sha256
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing QAT receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    parent_receipt = json.loads(args.parent_receipt.read_text())
    parent_receipt_sha256 = sha256_file(args.parent_receipt)
    cache_receipt = json.loads(args.cache_receipt.read_text())
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        prereg.get("status") != "frozen_before_parent_ptq_terminal_result"
        or prereg["pre_execution_correction"]["qat_execution_had_started"] is not False
        or prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or prereg["scope"]["parent_ptq_runner_commit"] != PARENT_RUNNER_COMMIT
        or tuple(prereg["scope"]["pilot_layers"]) != TARGET_LAYERS
        or parent_receipt.get("schema")
        != "qwen38_27b_affine_butterfly_block_gate_receipt_v1"
        or parent_receipt.get("repository_commit") != PARENT_RUNNER_COMMIT
        or parent_receipt.get("complete") is not True
        or parent_receipt.get("terminal") is not True
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("terminal") is not True
        or cache_receipt.get("cache", {}).get("sha256") != sha256_file(args.cache)
        or sha256_file(index_path) != INDEX_SHA256
    ):
        raise ValueError("QAT source, parent, or cache identity changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    attention_mask = cache["attention_mask"].long()
    position_ids = cache["text_position_ids"].long()
    calibration_rows = cache["calibration_flat_rows"].long()
    validation_rows = cache["validation_flat_rows"].long()
    weight_map = json.loads(index_path.read_text())["weight_map"]
    parent_by_layer = {int(item["layer"]): item for item in parent_receipt["layers"]}
    training = prereg["training"]
    gate_config = prereg["falsification_gate"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []

    for layer_index in TARGET_LAYERS:
        layer_started = time.monotonic()
        layer_type = config.layer_types[layer_index]
        terminal_path = (
            args.artifact_root / "layers" / str(layer_index) / "terminal-result.json"
        )
        if terminal_path.exists():
            resumed = json.loads(terminal_path.read_text())
            validate_terminal_layer(
                result=resumed,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=preregistration_sha256,
                parent_receipt_sha256=parent_receipt_sha256,
            )
            results.append(resumed)
            print(
                json.dumps(
                    {"event": "qat_layer_reused", "layer": layer_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {"event": "qat_layer_started", "layer": layer_index, "type": layer_type},
                sort_keys=True,
            ),
            flush=True,
        )
        state = layer_state(args.model_dir, weight_map, layer_index)
        hidden = cache["layers"][str(layer_index)]["input"]
        parent = cache["layers"][str(layer_index)]["output"]
        position_embeddings, masks = make_context(
            config, hidden, attention_mask, position_ids
        )
        sensitivity = sensitivity_report(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
            hidden=hidden,
            parent=parent,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
            calibration_rows=calibration_rows,
            validation_rows=validation_rows,
        )

        block = build_trainable_block(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
        )
        initial_metrics = score_block(
            block,
            hidden,
            parent,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
            calibration_rows=calibration_rows,
            validation_rows=validation_rows,
        )
        groups = qat_parameter_groups(
            block.payloads.values(),
            block.rotations.values(),
            ternary_learning_rate=float(training["ternary_shadow_learning_rate"]),
            index_learning_rate=float(training["four_bit_index_shadow_learning_rate"]),
            continuous_learning_rate=float(training["continuous_learning_rate"]),
        )
        optimizer = torch.optim.Adam(groups.groups)
        optimizer_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]

        training_batches = 8
        train_hidden = hidden[:training_batches]
        train_parent = parent[:training_batches]
        train_attention = attention_mask[:training_batches]
        train_positions = position_ids[:training_batches]
        train_embeddings, train_masks = make_context(
            config, train_hidden, train_attention, train_positions
        )
        selected_parent = train_parent.reshape(-1, train_parent.shape[-1]).index_select(
            0, calibration_rows
        ).float()
        parent_energy = selected_parent.square().sum().detach()
        if float(parent_energy) <= 0:
            raise ValueError("QAT training parent energy is zero")
        history = []
        for step in range(1, int(training["steps"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            output = forward_with_grad(
                block.layer,
                train_hidden,
                position_embeddings=train_embeddings,
                attention_mask=train_masks[layer_type],
                position_ids=train_positions,
            )
            selected = output.reshape(-1, output.shape[-1]).index_select(
                0, calibration_rows
            ).float()
            loss = (selected - selected_parent).square().sum() / parent_energy
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite QAT loss at layer {layer_index} step {step}")
            loss.backward()
            if any(parameter.grad is None for parameter in optimizer_parameters):
                raise RuntimeError(f"QAT parameter received no gradient at layer {layer_index}")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters, float(training["gradient_clip_norm"])
                )
            )
            optimizer.step()
            for payload in block.payloads.values():
                payload.project_training_state()
            record = {
                "step": step,
                "hard_forward_calibration_nmse": float(loss.detach()),
                "gradient_norm_before_clip": gradient_norm,
            }
            history.append(record)
            print(
                json.dumps(
                    {"event": "qat_step_terminal", "layer": layer_index, **record},
                    sort_keys=True,
                ),
                flush=True,
            )

        final_metrics = score_block(
            block,
            hidden,
            parent,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
            calibration_rows=calibration_rows,
            validation_rows=validation_rows,
        )
        manifests, actual_storage = export_trained_block(
            block=block,
            artifact_root=args.artifact_root,
            layer_index=layer_index,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            parent_receipt_sha256=parent_receipt_sha256,
        )
        direct_nmse = float(
            parent_by_layer[layer_index]["direct_complete_block_validation_nmse"]
        )
        parent_ptq_nmse = initial_metrics["validation_complete_block_nmse"]
        trained_nmse = final_metrics["validation_complete_block_nmse"]
        relative_vs_direct = (direct_nmse - trained_nmse) / max(direct_nmse, 1e-30)
        regression_vs_parent = trained_nmse - parent_ptq_nmse
        passed = (
            relative_vs_direct
            >= gate_config["minimum_relative_validation_nmse_reduction_vs_direct_ternary"]
            and regression_vs_parent <= gate_config["maximum_regression_vs_parent_ptq_candidate"]
        )
        result = {
            "repository_commit": repository_commit,
            "preregistration_sha256": preregistration_sha256,
            "parent_receipt_sha256": parent_receipt_sha256,
            "layer": layer_index,
            "type": layer_type,
            "sensitivity": sensitivity,
            "initial_packed_ptq_metrics": initial_metrics,
            "final_packed_qat_metrics": final_metrics,
            "direct_parent_validation_nmse": direct_nmse,
            "relative_validation_nmse_reduction_vs_direct": relative_vs_direct,
            "validation_nmse_change_vs_packed_ptq": regression_vs_parent,
            "training": {
                "history": history,
                "ternary_shadow_parameters": groups.ternary_parameters,
                "index_shadow_parameters": groups.index_parameters,
                "continuous_parameters": groups.continuous_parameters,
            },
            "site_manifests": [
                str(
                    Path("layers")
                    / str(layer_index)
                    / "sites"
                    / manifest["site"]
                    / "manifest.json"
                )
                for manifest in manifests
            ],
            "actual_packed_tensor_and_rotation_bytes": actual_storage,
            "passed": passed,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        atomic_json(
            result,
            terminal_path,
        )
        results.append(result)
        print(
            json.dumps({"event": "qat_layer_terminal", **result}, sort_keys=True),
            flush=True,
        )
        del optimizer, optimizer_parameters, block, state
        gc.collect()

    projected = prereg["whole_model_budget"]
    gate = {
        "minimum_relative_validation_nmse_reduction_vs_direct_ternary": gate_config[
            "minimum_relative_validation_nmse_reduction_vs_direct_ternary"
        ],
        "maximum_validation_nmse_change_vs_parent_ptq": gate_config[
            "maximum_regression_vs_parent_ptq_candidate"
        ],
        "maximum_projected_complete_text_file_bpw": gate_config[
            "maximum_projected_complete_text_file_bpw"
        ],
        "observed_projected_complete_text_file_bpw": projected[
            "base_projected_complete_bundle_bpw"
        ],
        "all_selected_blocks_passed": all(item["passed"] for item in results),
        "all_payloads_terminal": len(results) == len(TARGET_LAYERS),
        "all_values_finite": all(
            np.isfinite(item["final_packed_qat_metrics"]["validation_complete_block_nmse"])
            for item in results
        ),
    }
    gate["passed"] = (
        gate["all_selected_blocks_passed"]
        and gate["all_payloads_terminal"]
        and gate["all_values_finite"]
        and gate["observed_projected_complete_text_file_bpw"]
        <= gate["maximum_projected_complete_text_file_bpw"]
    )
    receipt = {
        "schema": "qwen38_27b_affine_butterfly_block_qat_receipt_v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "parent_receipt_sha256": parent_receipt_sha256,
        "cache_receipt_sha256": sha256_file(args.cache_receipt),
        "layers": results,
        "whole_model_budget": projected,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "terminal sensitivity-aware three-block QAT result only; not a whole-model artifact, capability result, CPU runtime result, or breakthrough",
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
