#!/usr/bin/env python3
"""Build and score the frozen mixed AFF3/CQ3 direct candidate on all 64 layers."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUNTIME))
import collect_qwen38_27b_affine_butterfly_block_cache_v1 as cache_v1
import collect_qwen38_27b_full_context_block_cache_v42 as cache_v42
import run_qwen38_27b_affine_butterfly_block_gate_v1 as parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_mixed_direct_screen_v1 as pilot_direct

from clark_quantize.qwen38_affine_artifact import (
    load_compressed_affine_tensor,
    sha256_file,
    verify_descriptor,
)
from clark_quantize.qwen38_affine_butterfly import deterministic_givens_pairings
from clark_quantize.qwen38_affine_qat import (
    StoredAffineLinear,
    TrainableAffinePayload,
    TrainableStoredRotation,
)
from clark_quantize.qwen38_mixed_cq3_runtime import (
    StreamedCq3Linear,
    tensor_sha256,
    write_step_zero_cq3_tensor,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
FULL_CONTEXT_RECEIPT_SHA256 = (
    "8653749c9616d88cec3509cfd3c87262145d370da2fc613aab88ee02263aca28"
)
ORIGINAL_AFFINE_PREREGISTRATION_SHA256 = (
    "8fa0b3159a0b6b7bdea1b00a0373b946a73e4ce0cd860bb303d1eb8ad3d63b09"
)
PILOT_DIRECT_RECEIPT_SHA256 = (
    "cbf2adb0a26522ecb03c4225f91df67e03b3c694bc3af342d059987775ec3320"
)
PILOT_LAYERS = (0, 32, 63)
SCHEMA = "qwen38_27b_all_layer_mixed_direct_receipt_v43"
LAYER_SCHEMA = "qwen38_27b_all_layer_mixed_direct_layer_v43"
CQ3_SCHEMA = "qwen38_27b_all_layer_mixed_direct_cq3_tensor_v43"


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


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
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
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: all-layer direct disk floor failed")


def selected_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, values.shape[-1]).index_select(0, rows).contiguous()


def attention_output_module(layer_type: str) -> str:
    if layer_type == "linear_attention":
        return "linear_attn.out_proj"
    if layer_type == "full_attention":
        return "self_attn.o_proj"
    raise ValueError("Qwen3.8 attention layer type changed")


def capture_exact_attention_output_inputs(
    layer: torch.nn.Module,
    *,
    layer_type: str,
    hidden: torch.Tensor,
    teacher_output: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    calibration_rows: torch.Tensor,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(selected_rows(inputs[0].detach(), calibration_rows))

    handle = layer.get_submodule(attention_output_module(layer_type)).register_forward_pre_hook(
        hook
    )
    try:
        with torch.no_grad():
            replayed = qat.forward_with_grad(
                layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
    finally:
        handle.remove()
    if not torch.equal(replayed, teacher_output):
        raise ValueError("exact BF16 layer no longer reproduces the full-context shard")
    if len(captured) != 1 or captured[0].shape[0] != calibration_rows.numel():
        raise RuntimeError("attention-output calibration capture changed")
    return captured[0]


def manifest_payload_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tensors = []
    for descriptor in manifest["tensors"]:
        tensors.append(
            {
                "tensor": descriptor["tensor"],
                "shape": descriptor["shape"],
                "group_size": descriptor["group_size"],
                "codes": {
                    "sha256": descriptor["codes"]["sha256"],
                    "unpacked_sha256": descriptor["codes"]["unpacked_sha256"],
                },
                "alpha": {
                    key: descriptor["alpha"][key]["sha256"]
                    for key in ("indices", "anchors", "codebook")
                },
                "alpha_unpacked_indices_sha256": descriptor["alpha"][
                    "unpacked_indices_sha256"
                ],
                "mu": {
                    key: descriptor["mu"][key]["sha256"]
                    for key in ("indices", "anchors", "codebook")
                },
                "mu_unpacked_indices_sha256": descriptor["mu"][
                    "unpacked_indices_sha256"
                ],
                "storage_bytes": descriptor["storage_bytes"],
            }
        )
    return {
        "site": manifest["site"],
        "tensor_names": manifest["tensor_names"],
        "input_dimension": manifest["input_dimension"],
        "pairing_seed": manifest["pairing_seed"],
        "angles": {
            "sha256": manifest["angles"]["sha256"],
            "shape": manifest["angles"]["shape"],
        },
        "tensors": tensors,
        "actual_storage_bytes": manifest["actual_storage_bytes"],
    }


def validate_aff3_manifest(
    manifest: dict[str, Any],
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    layer_index: int,
) -> None:
    tensor_names = parent.site_layout(expected_layer_type(layer_index))[
        "attention_output"
    ]
    if (
        manifest.get("schema") != "qwen38_affine_butterfly_site_v1"
        or manifest.get("repository_commit") != repository_commit
        or manifest.get("preregistration_sha256") != preregistration_sha256
        or manifest.get("layer") != layer_index
        or manifest.get("site") != "attention_output"
        or tuple(manifest.get("tensor_names", [])) != tensor_names
        or manifest.get("complete") is not True
        or manifest.get("terminal") is not True
    ):
        raise ValueError("all-layer AFF3 direct manifest identity changed")
    for descriptor in manifest["tensors"]:
        verify_descriptor(artifact_root, descriptor)
    parent.read_angles(artifact_root, manifest["angles"])


def build_aff3_block(
    *,
    config: Any,
    layer_index: int,
    state: Mapping[str, torch.Tensor],
    artifact_root: Path,
    manifest: dict[str, Any],
) -> torch.nn.Module:
    layer = parent.instantiate_layer(config, layer_index, state)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    angles = parent.read_angles(artifact_root, manifest["angles"])
    pairings = deterministic_givens_pairings(
        int(manifest["input_dimension"]),
        stages=angles.shape[0],
        seed=int(manifest["pairing_seed"]),
    )
    rotation = TrainableStoredRotation(angles, pairings)
    for name, descriptor in zip(manifest["tensor_names"], manifest["tensors"]):
        payload = TrainableAffinePayload(
            load_compressed_affine_tensor(artifact_root, descriptor)
        )
        module_name, parameter_name = name.rsplit(".", 1)
        if parameter_name != "weight":
            raise ValueError("AFF3 direct tensor is not a matrix weight")
        qat.set_child_module(layer, module_name, StoredAffineLinear(payload, rotation))
    layer.eval()
    return layer


def validate_or_write_direct_cq3(
    *,
    source: torch.Tensor,
    target: dict[str, Any],
    payload_path: Path,
    descriptor_path: Path,
    repository_commit: str,
    preregistration_sha256: str,
    full_context_receipt_sha256: str,
    layer_index: int,
) -> dict[str, Any]:
    if tuple(source.shape) != target["shape"] or source.dtype != torch.bfloat16:
        raise ValueError("all-layer direct CQ3 source tensor changed")
    source_sha = tensor_sha256(source)
    identities = {
        "schema": CQ3_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "full_context_receipt_sha256": full_context_receipt_sha256,
        "layer": layer_index,
        "state_tensor": target["state_name"],
        "gguf_tensor": target["gguf_name"],
        "shape": list(target["shape"]),
        "source_dtype": str(source.dtype),
        "source_sha256": source_sha,
    }
    if payload_path.exists() or descriptor_path.exists():
        if not payload_path.is_file() or not descriptor_path.is_file():
            raise RuntimeError("partial all-layer direct CQ3 tensor requires inspection")
        descriptor = json.loads(descriptor_path.read_text())
        if (
            any(descriptor.get(key) != value for key, value in identities.items())
            or descriptor.get("payload", {}).get("bytes") != payload_path.stat().st_size
            or descriptor.get("payload", {}).get("sha256")
            != sha256_file(payload_path)
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("existing all-layer direct CQ3 tensor identity changed")
        return descriptor
    payload = write_step_zero_cq3_tensor(source, payload_path, row_shard=128)
    descriptor = {
        **identities,
        "payload": payload,
        "complete": True,
        "terminal": True,
    }
    cache_v1.atomic_json(descriptor, descriptor_path)
    return descriptor


def load_direct_mlp(
    *,
    layer_index: int,
    cq3_root: Path,
    calibration_rows: torch.Tensor,
) -> pilot_direct.SelectedRowsCq3Mlp:
    modules = []
    for target in pilot_direct.mlp_targets(layer_index):
        modules.append(
            StreamedCq3Linear.from_payload(
                cq3_root / f"{target['gguf_name']}.cq3",
                rows=target["shape"][0],
                columns=target["shape"][1],
                row_shard=128,
                output_dtype=torch.bfloat16,
            )
        )
    return pilot_direct.SelectedRowsCq3Mlp(
        *modules, selected_rows=calibration_rows
    )


def validate_terminal_layer(
    result: dict[str, Any],
    *,
    artifact_root: Path,
    aff3_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    full_context_receipt_sha256: str,
) -> None:
    layer_index = result.get("layer")
    if (
        result.get("schema") != LAYER_SCHEMA
        or result.get("model") != MODEL_ID
        or result.get("revision") != MODEL_REVISION
        or result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("full_context_receipt_sha256")
        != full_context_receipt_sha256
        or not isinstance(layer_index, int)
        or result.get("type") != expected_layer_type(layer_index)
        or result.get("complete") is not True
        or result.get("terminal") is not True
    ):
        raise ValueError("all-layer direct terminal layer identity changed")
    manifest_path = aff3_root / result["aff3_manifest"]
    manifest = json.loads(manifest_path.read_text())
    validate_aff3_manifest(
        manifest,
        artifact_root=aff3_root,
        repository_commit=repository_commit,
        preregistration_sha256=preregistration_sha256,
        layer_index=layer_index,
    )
    for record in result["cq3_payloads"]:
        descriptor_path = artifact_root / record["descriptor"]
        descriptor = json.loads(descriptor_path.read_text())
        payload_path = descriptor_path.with_suffix(".cq3")
        if (
            descriptor.get("schema") != CQ3_SCHEMA
            or descriptor.get("repository_commit") != repository_commit
            or descriptor.get("preregistration_sha256")
            != preregistration_sha256
            or descriptor.get("full_context_receipt_sha256")
            != full_context_receipt_sha256
            or descriptor.get("payload", {}).get("sha256")
            != sha256_file(payload_path)
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("all-layer direct terminal CQ3 payload changed")


def validate_final(
    receipt_path: Path,
    marker_path: Path,
    *,
    artifact_root: Path,
    aff3_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
) -> dict[str, Any] | None:
    if not receipt_path.exists() and not marker_path.exists():
        return None
    if not receipt_path.is_file() or not marker_path.is_file():
        raise RuntimeError("partial all-layer direct terminal receipt requires inspection")
    receipt = json.loads(receipt_path.read_text())
    marker = json.loads(marker_path.read_text())
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("repository_commit") != repository_commit
        or receipt.get("preregistration_sha256") != preregistration_sha256
        or receipt.get("full_context_receipt_sha256")
        != FULL_CONTEXT_RECEIPT_SHA256
        or len(receipt.get("layers", [])) != 64
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or marker.get("receipt_sha256") != sha256_file(receipt_path)
        or marker.get("layer_count") != 64
        or marker.get("passed") != receipt.get("gate", {}).get("passed")
        or marker.get("terminal") is not True
    ):
        raise ValueError("existing all-layer direct terminal receipt changed")
    for result in receipt["layers"]:
        validate_terminal_layer(
            result,
            artifact_root=artifact_root,
            aff3_root=aff3_root,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            full_context_receipt_sha256=FULL_CONTEXT_RECEIPT_SHA256,
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_mixed_direct_v43.json",
    )
    parser.add_argument(
        "--full-context-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-context-block-cache-v42"),
    )
    parser.add_argument(
        "--full-context-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-full-context-block-cache-v42/receipt.json"
        ),
    )
    parser.add_argument(
        "--pilot-aff3-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"
        ),
    )
    parser.add_argument(
        "--pilot-cq3-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/direct"
        ),
    )
    parser.add_argument(
        "--pilot-direct-receipt",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/direct-screen.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-direct-v43"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/receipt.json"
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/complete.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen all-layer direct screen uses four CPU threads")

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = sha256_file(args.preregistration)
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_all_layer_mixed_direct_preregistration_v43"
        or prereg.get("status") != "frozen_before_all_layer_direct_execution"
        or prereg.get("scope", {}).get("active_model_only") != MODEL_ID
        or prereg.get("scope", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("target_layers") != list(range(64))
        or prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True
    ):
        raise ValueError("all-layer direct preregistration changed")
    resource_guard(
        Path("/workspace"), int(prereg["resources"]["minimum_free_disk_bytes"])
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(int(prereg["execution"]["seed"]))

    aff3_root = args.artifact_root / "aff3-base"
    cq3_root = args.artifact_root / "cq3-direct"
    layer_results_root = args.artifact_root / "layers"
    existing = validate_final(
        args.receipt,
        args.completion_marker,
        artifact_root=args.artifact_root,
        aff3_root=aff3_root,
        repository_commit=repository_commit,
        preregistration_sha256=prereg_sha,
    )
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    index_path = args.model_dir / "model.safetensors.index.json"
    full_context_receipt = json.loads(args.full_context_receipt.read_text())
    pilot_receipt = json.loads(args.pilot_direct_receipt.read_text())
    original_prereg_path = (
        RUNTIME / "qwen38_27b_affine_butterfly_blocks_preregistration_v1.json"
    )
    original_prereg = json.loads(original_prereg_path.read_text())
    if (
        sha256_file(index_path) != INDEX_SHA256
        or sha256_file(args.full_context_receipt) != FULL_CONTEXT_RECEIPT_SHA256
        or full_context_receipt.get("schema")
        != cache_v42.RECEIPT_SCHEMA
        or full_context_receipt.get("gate", {}).get("passed") is not True
        or len(full_context_receipt.get("layers", [])) != 64
        or full_context_receipt.get("complete") is not True
        or full_context_receipt.get("terminal") is not True
        or sha256_file(original_prereg_path)
        != ORIGINAL_AFFINE_PREREGISTRATION_SHA256
        or sha256_file(args.pilot_direct_receipt) != PILOT_DIRECT_RECEIPT_SHA256
        or pilot_receipt.get("gate", {}).get("passed") is not True
        or pilot_receipt.get("complete") is not True
        or pilot_receipt.get("terminal") is not True
    ):
        raise ValueError("all-layer direct source evidence changed")
    fit_config = original_prereg["fit"]
    if (
        int(fit_config["group_size"]) != 128
        or int(fit_config["rotation_stages"]) != 6
        or int(fit_config["rotation_steps"]) != 8
        or float(fit_config["rotation_learning_rate"]) != 0.002
        or int(fit_config["row_shard"]) != 128
        or int(fit_config["seed"]) != 20260818
    ):
        raise ValueError("frozen AFF3 fit configuration changed")

    context_path = args.full_context_root / "context.pt"
    if (
        context_path.stat().st_size
        != full_context_receipt["context"]["bytes"]
        or sha256_file(context_path) != full_context_receipt["context"]["sha256"]
    ):
        raise ValueError("full-context metadata artifact changed")
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    calibration_rows = context["calibration_flat_rows"].long()
    full_attention_mask = context["attention_mask"].long()
    full_position_ids = context["text_position_ids"].long()
    attention_mask = full_attention_mask[:8]
    position_ids = full_position_ids[:8]
    if (
        calibration_rows.numel() != 64
        or int(calibration_rows.max()) >= 8 * 64
        or context["input_ids"].shape != (12, 64)
        or full_attention_mask.shape != (12, 64)
        or full_position_ids.shape != (12, 64)
    ):
        raise ValueError("all-layer calibration identity changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != 5120
        or tuple(config.layer_types)
        != tuple(expected_layer_type(index) for index in range(64))
    ):
        raise ValueError("Qwen3.8-27B all-layer direct architecture changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    full_by_layer = {
        int(record["layer"]): record for record in full_context_receipt["layers"]
    }
    pilot_by_layer = {
        int(record["layer"]): record for record in pilot_receipt["layers"]
    }
    threshold = float(prereg["gate"]["maximum_calibration_nmse_per_layer"])
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.monotonic()

    for layer_index in range(64):
        terminal_path = layer_results_root / f"{layer_index:02d}.json"
        if terminal_path.exists():
            result = json.loads(terminal_path.read_text())
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                aff3_root=aff3_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                full_context_receipt_sha256=FULL_CONTEXT_RECEIPT_SHA256,
            )
            results.append(result)
            print(
                json.dumps(
                    {"event": "all_layer_direct_layer_reused", "layer": layer_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        layer_started = time.monotonic()
        layer_type = config.layer_types[layer_index]
        print(
            json.dumps(
                {
                    "event": "all_layer_direct_layer_started",
                    "layer": layer_index,
                    "type": layer_type,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        shard_record = full_by_layer[layer_index]
        shard_path = args.full_context_root / "layers" / f"{layer_index:02d}.pt"
        if (
            shard_path.stat().st_size != shard_record["shard"]["bytes"]
            or sha256_file(shard_path) != shard_record["shard"]["sha256"]
        ):
            raise ValueError("authenticated full-context layer shard changed")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        cache_v42.validate_shard_tensors(
            shard,
            layer_index,
            preregistration_sha256=full_context_receipt["preregistration_sha256"],
            producer_commit=full_context_receipt["repository_commit"],
        )
        full_hidden = shard["input"].contiguous()
        full_teacher_output = shard["output"].contiguous()
        state, parameters = cache_v1.layer_state(
            args.model_dir, weight_map, layer_index
        )
        full_position_embeddings, full_masks = qat.make_context(
            config, full_hidden, full_attention_mask, full_position_ids
        )
        exact_layer = parent.instantiate_layer(config, layer_index, state)
        attention_inputs = capture_exact_attention_output_inputs(
            exact_layer,
            layer_type=layer_type,
            hidden=full_hidden,
            teacher_output=full_teacher_output,
            position_embeddings=full_position_embeddings,
            attention_mask=full_masks[layer_type],
            position_ids=full_position_ids,
            calibration_rows=calibration_rows,
        )
        del exact_layer
        gc.collect()

        tensor_names = parent.site_layout(layer_type)["attention_output"]
        aff3_manifest = parent.fit_site(
            artifact_root=aff3_root,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            layer_index=layer_index,
            site="attention_output",
            tensor_names=tensor_names,
            state=state,
            calibration=attention_inputs,
            validation=attention_inputs,
            fit_config=fit_config,
        )
        validate_aff3_manifest(
            aff3_manifest,
            artifact_root=aff3_root,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            layer_index=layer_index,
        )
        aff3_pilot_exact: bool | None = None
        if layer_index in PILOT_LAYERS:
            old_manifest_path = (
                args.pilot_aff3_root
                / "layers"
                / str(layer_index)
                / "sites/attention_output/manifest.json"
            )
            old_manifest = json.loads(old_manifest_path.read_text())
            aff3_pilot_exact = manifest_payload_fingerprint(
                aff3_manifest
            ) == manifest_payload_fingerprint(old_manifest)
            if not aff3_pilot_exact:
                raise ValueError("all-layer AFF3 pilot algebra reproduction changed")

        descriptors: list[dict[str, Any]] = []
        layer_cq3_root = cq3_root / "layers" / f"{layer_index:02d}"
        for target in pilot_direct.mlp_targets(layer_index):
            descriptor = validate_or_write_direct_cq3(
                source=state[target["state_name"]],
                target=target,
                payload_path=layer_cq3_root / f"{target['gguf_name']}.cq3",
                descriptor_path=layer_cq3_root / f"{target['gguf_name']}.json",
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                full_context_receipt_sha256=FULL_CONTEXT_RECEIPT_SHA256,
                layer_index=layer_index,
            )
            descriptors.append(descriptor)
            if layer_index in PILOT_LAYERS:
                old_payload = (
                    args.pilot_cq3_root
                    / "layers"
                    / str(layer_index)
                    / "payloads"
                    / f"{target['gguf_name']}.cq3"
                )
                if descriptor["payload"]["sha256"] != sha256_file(old_payload):
                    raise ValueError("all-layer CQ3 pilot algebra reproduction changed")

        candidate = build_aff3_block(
            config=config,
            layer_index=layer_index,
            state=state,
            artifact_root=aff3_root,
            manifest=aff3_manifest,
        )
        hidden = full_hidden[:8].contiguous()
        teacher_output = full_teacher_output[:8].contiguous()
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        qat.set_child_module(
            candidate,
            "mlp",
            load_direct_mlp(
                layer_index=layer_index,
                cq3_root=layer_cq3_root,
                calibration_rows=calibration_rows,
            ),
        )
        for target in pilot_direct.mlp_targets(layer_index):
            state.pop(target["state_name"])
        with torch.no_grad():
            candidate_output = qat.forward_with_grad(
                candidate,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )
        calibration_nmse = qat.selected_nmse(
            candidate_output, teacher_output, calibration_rows
        )
        if not math.isfinite(calibration_nmse):
            raise RuntimeError("non-finite all-layer direct calibration NMSE")
        score_pilot_exact: bool | None = None
        if layer_index in PILOT_LAYERS:
            old_score = float(
                pilot_by_layer[layer_index][
                    "direct_mixed_calibration_complete_block_nmse"
                ]
            )
            score_pilot_exact = abs(calibration_nmse - old_score) <= 1e-8
            if not score_pilot_exact:
                raise ValueError("all-layer direct pilot score reproduction changed")

        aff3_manifest_relative = str(
            Path("layers")
            / str(layer_index)
            / "sites/attention_output/manifest.json"
        )
        result = {
            "schema": LAYER_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "repository_commit": repository_commit,
            "preregistration_sha256": prereg_sha,
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "full_context_shard_sha256": shard_record["shard"]["sha256"],
            "layer": layer_index,
            "type": layer_type,
            "parameters": parameters,
            "calibration_rows": calibration_rows.numel(),
            "direct_mixed_calibration_complete_block_nmse": calibration_nmse,
            "maximum_allowed_nmse": threshold,
            "passed": calibration_nmse <= threshold,
            "aff3_manifest": aff3_manifest_relative,
            "aff3_payload_fingerprint": manifest_payload_fingerprint(aff3_manifest),
            "aff3_pilot_algebra_exact": aff3_pilot_exact,
            "cq3_payloads": [
                {
                    "state_tensor": item["state_tensor"],
                    "gguf_tensor": item["gguf_tensor"],
                    "source_sha256": item["source_sha256"],
                    "payload": item["payload"],
                    "descriptor": str(
                        Path("cq3-direct")
                        / "layers"
                        / f"{layer_index:02d}"
                        / f"{item['gguf_tensor']}.json"
                    ),
                }
                for item in descriptors
            ],
            "cq3_pilot_algebra_exact": (
                True if layer_index in PILOT_LAYERS else None
            ),
            "pilot_score_exact_within_1e_8": score_pilot_exact,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        cache_v1.atomic_json(result, terminal_path)
        validate_terminal_layer(
            result,
            artifact_root=args.artifact_root,
            aff3_root=aff3_root,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            full_context_receipt_sha256=FULL_CONTEXT_RECEIPT_SHA256,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "all_layer_direct_layer_terminal",
                    "layer": layer_index,
                    "type": layer_type,
                    "calibration_nmse": calibration_nmse,
                    "passed": result["passed"],
                    "pilot_algebra_exact": (
                        aff3_pilot_exact and score_pilot_exact
                        if layer_index in PILOT_LAYERS
                        else None
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            shard,
            hidden,
            teacher_output,
            state,
            candidate,
            candidate_output,
            attention_inputs,
            position_embeddings,
            masks,
            full_hidden,
            full_teacher_output,
            full_position_embeddings,
            full_masks,
            descriptors,
        )
        gc.collect()
        trim_allocator()

    if len(results) != 64 or [item["layer"] for item in results] != list(range(64)):
        raise RuntimeError("all-layer direct terminal denominator changed")
    calibration_values = [
        float(item["direct_mixed_calibration_complete_block_nmse"])
        for item in results
    ]
    total_payload_bytes = sum(
        int(item["aff3_payload_fingerprint"]["actual_storage_bytes"])
        + sum(int(payload["payload"]["bytes"]) for payload in item["cq3_payloads"])
        for item in results
    )
    gate = {
        "required_layers": 64,
        "observed_layers": len(results),
        "all_layers_terminal": all(item["terminal"] for item in results),
        "all_values_finite": all(math.isfinite(value) for value in calibration_values),
        "maximum_calibration_nmse_per_layer": threshold,
        "maximum_observed_calibration_nmse": max(calibration_values),
        "maximum_observed_layer": max(
            range(64), key=lambda index: calibration_values[index]
        ),
        "all_layers_below_maximum": all(item["passed"] for item in results),
        "pilot_layers": list(PILOT_LAYERS),
        "all_pilot_aff3_algebras_exact": all(
            results[index]["aff3_pilot_algebra_exact"] is True
            for index in PILOT_LAYERS
        ),
        "all_pilot_cq3_algebras_exact": all(
            results[index]["cq3_pilot_algebra_exact"] is True
            for index in PILOT_LAYERS
        ),
        "all_pilot_scores_exact_within_1e_8": all(
            results[index]["pilot_score_exact_within_1e_8"] is True
            for index in PILOT_LAYERS
        ),
        "total_selected_payload_bytes": total_payload_bytes,
        "maximum_selected_payload_bytes": int(
            prereg["gate"]["maximum_selected_payload_bytes"]
        ),
        "payload_storage_within_maximum": total_payload_bytes
        <= int(prereg["gate"]["maximum_selected_payload_bytes"]),
    }
    gate["passed"] = bool(
        gate["observed_layers"] == gate["required_layers"]
        and gate["all_layers_terminal"]
        and gate["all_values_finite"]
        and gate["all_layers_below_maximum"]
        and gate["all_pilot_aff3_algebras_exact"]
        and gate["all_pilot_cq3_algebras_exact"]
        and gate["all_pilot_scores_exact_within_1e_8"]
        and gate["payload_storage_within_maximum"]
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
        "original_affine_preregistration_sha256": (
            ORIGINAL_AFFINE_PREREGISTRATION_SHA256
        ),
        "pilot_direct_receipt_sha256": PILOT_DIRECT_RECEIPT_SHA256,
        "layers": results,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "claim_boundary": prereg["claim_boundary"],
    }
    cache_v1.atomic_json(receipt, args.receipt)
    cache_v1.atomic_json(
        {
            "terminal": True,
            "passed": gate["passed"],
            "receipt_sha256": sha256_file(args.receipt),
            "layer_count": len(results),
            "maximum_observed_calibration_nmse": gate[
                "maximum_observed_calibration_nmse"
            ],
            "total_selected_payload_bytes": total_payload_bytes,
        },
        args.completion_marker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
