#!/usr/bin/env python3
"""Train the frozen mixed AFF3/CQ3 candidate on all 64 Qwen3.8-27B layers."""

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
import run_qwen38_27b_all_layer_mixed_direct_v43 as direct
import run_qwen38_27b_mixed_direct_screen_v1 as pilot_direct
import run_qwen38_27b_mixed_train_blocks_v1 as pilot_training

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
from clark_quantize.qwen38_layer0_qat import RowShardTrainingConfig
from clark_quantize.qwen38_mixed_aff3_trust_region import (
    calibration_only_hard_forward_trust_region,
)
from clark_quantize.qwen38_mixed_cq3_runtime import StreamedCq3Linear


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
FULL_CONTEXT_RECEIPT_SHA256 = direct.FULL_CONTEXT_RECEIPT_SHA256
PILOT_TRAINING_RECEIPT_SHA256 = (
    "4454761a8519db03ca88a91c21cf8344cab12445266f1936767c04ebe3b76128"
)
PILOT_LAYERS = (0, 32, 63)
SCHEMA = "qwen38_27b_all_layer_mixed_training_receipt_v44"
LAYER_SCHEMA = "qwen38_27b_all_layer_mixed_trained_layer_v44"


def trim_allocator() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


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
        raise SystemExit("infrastructure-invalid: all-layer training disk floor failed")


def trained_aff3_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
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
        "angles": {
            "sha256": manifest["angles"]["sha256"],
            "shape": manifest["angles"]["shape"],
        },
        "tensors": tensors,
        "actual_storage_bytes": manifest["actual_storage_bytes"],
    }


def build_trainable_aff3_block(
    *,
    config: Any,
    layer_index: int,
    state: Mapping[str, torch.Tensor],
    direct_aff3_root: Path,
    direct_repository_commit: str,
    direct_preregistration_sha256: str,
) -> qat.TrainableBlock:
    manifest_path = (
        direct_aff3_root
        / "layers"
        / str(layer_index)
        / "sites/attention_output/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    direct.validate_aff3_manifest(
        manifest,
        artifact_root=direct_aff3_root,
        repository_commit=direct_repository_commit,
        preregistration_sha256=direct_preregistration_sha256,
        layer_index=layer_index,
    )
    layer = parent.instantiate_layer(config, layer_index, state)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    angles = parent.read_angles(direct_aff3_root, manifest["angles"])
    pairings = deterministic_givens_pairings(
        int(manifest["input_dimension"]),
        stages=angles.shape[0],
        seed=int(manifest["pairing_seed"]),
    )
    rotation = TrainableStoredRotation(angles, pairings)
    payloads: dict[str, TrainableAffinePayload] = {}
    source_shapes: dict[str, tuple[int, ...]] = {}
    names = tuple(manifest["tensor_names"])
    for name, descriptor in zip(names, manifest["tensors"]):
        payload = TrainableAffinePayload(
            load_compressed_affine_tensor(direct_aff3_root, descriptor)
        )
        payloads[name] = payload
        source_shapes[name] = tuple(state[name].shape)
        module_name, parameter_name = name.rsplit(".", 1)
        if parameter_name != "weight":
            raise ValueError("all-layer trainable AFF3 source is not a weight")
        qat.set_child_module(layer, module_name, StoredAffineLinear(payload, rotation))
    layer.eval()
    return qat.TrainableBlock(
        layer=layer,
        payloads=payloads,
        rotations={"attention_output": rotation},
        names_by_site={"attention_output": names},
        source_shapes=source_shapes,
    )


def direct_mlp(
    *,
    layer_index: int,
    direct_cq3_root: Path,
    direct_layer_result: dict[str, Any],
    calibration_rows: torch.Tensor,
) -> pilot_direct.SelectedRowsCq3Mlp:
    by_name = {
        record["state_tensor"]: record for record in direct_layer_result["cq3_payloads"]
    }
    modules = []
    for target in pilot_direct.mlp_targets(layer_index):
        record = by_name[target["state_name"]]
        path = (
            direct_cq3_root
            / "layers"
            / f"{layer_index:02d}"
            / f"{target['gguf_name']}.cq3"
        )
        if (
            path.stat().st_size != record["payload"]["bytes"]
            or sha256_file(path) != record["payload"]["sha256"]
        ):
            raise ValueError("all-layer direct CQ3 payload changed before training")
        modules.append(
            StreamedCq3Linear.from_payload(
                path,
                rows=target["shape"][0],
                columns=target["shape"][1],
                row_shard=128,
                output_dtype=torch.bfloat16,
            )
        )
    return pilot_direct.SelectedRowsCq3Mlp(
        *modules, selected_rows=calibration_rows
    )


def load_trained_cq3_linear(
    tensor_root: Path, target: dict[str, Any]
) -> StreamedCq3Linear:
    return StreamedCq3Linear.from_payload(
        tensor_root / f"{target['gguf_name']}.cq3",
        rows=target["shape"][0],
        columns=target["shape"][1],
        row_shard=128,
        output_dtype=torch.bfloat16,
    )


def validate_terminal_layer(
    result: dict[str, Any],
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    direct_receipt_sha256: str,
) -> None:
    if (
        result.get("schema") != LAYER_SCHEMA
        or result.get("model") != MODEL_ID
        or result.get("revision") != MODEL_REVISION
        or result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("direct_receipt_sha256") != direct_receipt_sha256
        or result.get("complete") is not True
        or result.get("terminal") is not True
    ):
        raise ValueError("all-layer trained terminal layer identity changed")
    qat.validate_terminal_layer(
        result=result,
        artifact_root=artifact_root,
        repository_commit=repository_commit,
        preregistration_sha256=preregistration_sha256,
        parent_receipt_sha256=direct_receipt_sha256,
    )
    for relative in result["cq3_descriptors"]:
        descriptor_path = artifact_root / relative
        descriptor = json.loads(descriptor_path.read_text())
        payload_path = descriptor_path.with_suffix(".cq3")
        if (
            descriptor.get("schema") != pilot_training.TENSOR_SCHEMA
            or descriptor.get("repository_commit") != repository_commit
            or descriptor.get("preregistration_sha256")
            != preregistration_sha256
            or descriptor.get("training_amendment_sha256")
            != preregistration_sha256
            or descriptor.get("training_cache_sha256")
            != FULL_CONTEXT_RECEIPT_SHA256
            or descriptor.get("payload", {}).get("sha256")
            != sha256_file(payload_path)
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("all-layer trained terminal CQ3 descriptor changed")


def validate_direct_parent_artifacts(
    direct_receipt: dict[str, Any],
    *,
    direct_artifact_root: Path,
) -> None:
    """Authenticate the complete v43 parent before starting any optimizer work."""
    layers = direct_receipt.get("layers", [])
    if (
        len(layers) != 64
        or [record.get("layer") for record in layers] != list(range(64))
        or not all(record.get("passed") is True for record in layers)
    ):
        raise ValueError("all-layer direct parent denominator changed")
    for result in layers:
        direct.validate_terminal_layer(
            result,
            artifact_root=direct_artifact_root,
            aff3_root=direct_artifact_root / "aff3-base",
            repository_commit=direct_receipt["repository_commit"],
            preregistration_sha256=direct_receipt["preregistration_sha256"],
            full_context_receipt_sha256=FULL_CONTEXT_RECEIPT_SHA256,
        )


def validate_full_context_parent_artifacts(
    full_context_receipt: dict[str, Any],
    *,
    full_context_root: Path,
) -> None:
    """Hash every v42 layer shard before the all-layer training launch."""
    layers = full_context_receipt.get("layers", [])
    if (
        len(layers) != 64
        or [record.get("layer") for record in layers] != list(range(64))
    ):
        raise ValueError("all-layer full-context parent denominator changed")
    context_path = full_context_root / "context.pt"
    if (
        context_path.stat().st_size != full_context_receipt["context"]["bytes"]
        or sha256_file(context_path)
        != full_context_receipt["context"]["sha256"]
    ):
        raise ValueError("all-layer training context artifact changed")
    for layer_index, record in enumerate(layers):
        shard_path = full_context_root / "layers" / f"{layer_index:02d}.pt"
        if (
            shard_path.stat().st_size != record["shard"]["bytes"]
            or sha256_file(shard_path) != record["shard"]["sha256"]
        ):
            raise ValueError(
                f"all-layer training full-context shard {layer_index} changed"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_all_layer_mixed_train_v44.json",
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
        "--direct-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-direct-v43"),
    )
    parser.add_argument(
        "--direct-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/receipt.json"
        ),
    )
    parser.add_argument(
        "--pilot-trained-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/trained"
        ),
    )
    parser.add_argument(
        "--pilot-training-receipt",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/training.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json"
        ),
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/complete.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate all frozen parents and exit without creating artifacts",
    )
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen all-layer mixed training uses four CPU threads")
    torch.set_num_threads(args.threads)

    prereg = json.loads(args.preregistration.read_text())
    prereg_sha = sha256_file(args.preregistration)
    direct_receipt_sha = sha256_file(args.direct_receipt)
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_all_layer_mixed_training_preregistration_v44"
        or prereg.get("status") != "frozen_after_v43_pass_before_training"
        or prereg.get("scope", {}).get("active_model_only") != MODEL_ID
        or prereg.get("scope", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("target_layers") != list(range(64))
        or prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True
        or prereg.get("parent_direct", {}).get("receipt_sha256")
        != direct_receipt_sha
        or prereg.get("frozen_inputs", {}).get("runner", {}).get("sha256")
        != sha256_file(Path(__file__))
        or prereg.get("frozen_inputs", {}).get("direct_receipt_sha256")
        != direct_receipt_sha
        or prereg.get("frozen_inputs", {}).get("full_context_receipt_sha256")
        != FULL_CONTEXT_RECEIPT_SHA256
        or prereg.get("frozen_inputs", {}).get("pilot_training_receipt_sha256")
        != PILOT_TRAINING_RECEIPT_SHA256
    ):
        raise ValueError("all-layer mixed training preregistration changed")
    resource_guard(
        Path("/workspace"), int(prereg["resources"]["minimum_free_disk_bytes"])
    )
    torch.manual_seed(int(prereg["training"]["seed"]))

    direct_receipt = json.loads(args.direct_receipt.read_text())
    full_context_receipt = json.loads(args.full_context_receipt.read_text())
    pilot_receipt = json.loads(args.pilot_training_receipt.read_text())
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        sha256_file(index_path) != INDEX_SHA256
        or sha256_file(args.full_context_receipt) != FULL_CONTEXT_RECEIPT_SHA256
        or full_context_receipt.get("gate", {}).get("passed") is not True
        or full_context_receipt.get("complete") is not True
        or full_context_receipt.get("terminal") is not True
        or direct_receipt.get("schema") != direct.SCHEMA
        or direct_receipt.get("gate", {}).get("passed") is not True
        or len(direct_receipt.get("layers", [])) != 64
        or direct_receipt.get("complete") is not True
        or direct_receipt.get("terminal") is not True
        or sha256_file(args.pilot_training_receipt)
        != PILOT_TRAINING_RECEIPT_SHA256
        or pilot_receipt.get("gate", {}).get("passed") is not True
        or pilot_receipt.get("complete") is not True
        or pilot_receipt.get("terminal") is not True
    ):
        raise ValueError("all-layer mixed training source evidence changed")
    validate_direct_parent_artifacts(
        direct_receipt,
        direct_artifact_root=args.direct_artifact_root,
    )
    validate_full_context_parent_artifacts(
        full_context_receipt,
        full_context_root=args.full_context_root,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "event": "all_layer_mixed_training_preflight_terminal",
                    "model": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "direct_receipt_sha256": direct_receipt_sha,
                    "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
                    "direct_layers_authenticated": 64,
                    "full_context_layers_authenticated": 64,
                    "writes": 0,
                    "passed": True,
                    "terminal": True,
                },
                sort_keys=True,
            )
        )
        return
    if args.receipt.exists() or args.completion_marker.exists():
        if not args.receipt.is_file() or not args.completion_marker.is_file():
            raise RuntimeError("partial all-layer training terminal receipt")
        existing = json.loads(args.receipt.read_text())
        marker = json.loads(args.completion_marker.read_text())
        if (
            existing.get("schema") != SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("preregistration_sha256") != prereg_sha
            or existing.get("direct_receipt_sha256") != direct_receipt_sha
            or len(existing.get("layers", [])) != 64
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
            or marker.get("receipt_sha256") != sha256_file(args.receipt)
            or marker.get("passed") != existing.get("gate", {}).get("passed")
            or marker.get("layer_count") != 64
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing all-layer training terminal receipt changed")
        for result in existing["layers"]:
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                direct_receipt_sha256=direct_receipt_sha,
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    context_path = args.full_context_root / "context.pt"
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    calibration_rows = context["calibration_flat_rows"].long()
    attention_mask = context["attention_mask"].long()[:8]
    position_ids = context["text_position_ids"].long()[:8]
    if (
        calibration_rows.numel() != 64
        or int(calibration_rows.max()) >= 8 * 64
        or attention_mask.shape != (8, 64)
        or position_ids.shape != (8, 64)
    ):
        raise ValueError("all-layer training calibration identity changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if (
        config.num_hidden_layers != 64
        or config.hidden_size != 5120
        or tuple(config.layer_types)
        != tuple(direct.expected_layer_type(index) for index in range(64))
    ):
        raise ValueError("Qwen3.8-27B all-layer training architecture changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    full_by_layer = {
        int(record["layer"]): record for record in full_context_receipt["layers"]
    }
    direct_by_layer = {
        int(record["layer"]): record for record in direct_receipt["layers"]
    }
    pilot_by_layer = {
        int(record["layer"]): record for record in pilot_receipt["layers"]
    }
    training = prereg["training"]
    cq3_config = RowShardTrainingConfig(
        steps=int(training["mlp_cq3"]["steps_per_row_shard"]),
        learning_rate=float(training["mlp_cq3"]["learning_rate"]),
        gradient_clip_norm=float(training["mlp_cq3"]["gradient_clip_norm"]),
        weight_reconstruction_weight=float(
            training["mlp_cq3"]["weight_reconstruction_weight"]
        ),
        group_size=int(training["mlp_cq3"]["group_size"]),
    )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.monotonic()

    for layer_index in range(64):
        terminal_path = (
            args.artifact_root / "layers" / f"{layer_index:02d}" / "terminal-result.json"
        )
        if terminal_path.exists():
            result = json.loads(terminal_path.read_text())
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                direct_receipt_sha256=direct_receipt_sha,
            )
            results.append(result)
            print(
                json.dumps(
                    {"event": "all_layer_trained_layer_reused", "layer": layer_index},
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
                    "event": "all_layer_trained_layer_started",
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
            raise ValueError("all-layer training full-context shard changed")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        cache_v42.validate_shard_tensors(
            shard,
            layer_index,
            preregistration_sha256=full_context_receipt["preregistration_sha256"],
            producer_commit=full_context_receipt["repository_commit"],
        )
        hidden = shard["input"][:8].contiguous()
        teacher_output = shard["output"][:8].contiguous()
        state, _ = cache_v1.layer_state(args.model_dir, weight_map, layer_index)
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        exact_layer = parent.instantiate_layer(config, layer_index, state)

        def exact_forward() -> torch.Tensor:
            return qat.forward_with_grad(
                exact_layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )

        exact_output, teacher_mlp_input, teacher_down_input = (
            pilot_training.capture_exact_teacher_activations(
                exact_layer, exact_forward, calibration_rows
            )
        )
        if not torch.equal(exact_output, teacher_output):
            raise ValueError("all-layer training exact BF16 shard replay changed")

        block = build_trainable_aff3_block(
            config=config,
            layer_index=layer_index,
            state=state,
            direct_aff3_root=args.direct_artifact_root / "aff3-base",
            direct_repository_commit=direct_receipt["repository_commit"],
            direct_preregistration_sha256=direct_receipt["preregistration_sha256"],
        )
        qat.set_child_module(
            block.layer,
            "mlp",
            direct_mlp(
                layer_index=layer_index,
                direct_cq3_root=args.direct_artifact_root / "cq3-direct",
                direct_layer_result=direct_by_layer[layer_index],
                calibration_rows=calibration_rows,
            ),
        )

        def candidate_forward() -> torch.Tensor:
            return qat.forward_with_grad(
                block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
            )

        direct_output, candidate_mlp_input = (
            pilot_training.capture_candidate_mlp_input(
                block.layer, candidate_forward, calibration_rows
            )
        )
        reproduced_direct_nmse = qat.selected_nmse(
            direct_output, teacher_output, calibration_rows
        )
        frozen_direct_nmse = float(
            direct_by_layer[layer_index][
                "direct_mixed_calibration_complete_block_nmse"
            ]
        )
        if abs(reproduced_direct_nmse - frozen_direct_nmse) > 1e-8:
            raise ValueError("all-layer direct score changed before training")
        qat.set_child_module(block.layer, "mlp", torch.nn.Identity())
        del direct_output, exact_output, exact_layer
        gc.collect()

        tensor_root = (
            args.artifact_root / "layers" / f"{layer_index:02d}" / "cq3"
        )
        targets = pilot_direct.mlp_targets(layer_index)
        descriptors: list[dict[str, Any]] = []
        source_by_name = {
            item["state_tensor"]: item for item in direct_by_layer[layer_index]["cq3_payloads"]
        }
        for target in targets[:2]:
            descriptors.append(
                pilot_training.train_cq3_tensor(
                    source=state[target["state_name"]],
                    candidate_inputs=candidate_mlp_input,
                    teacher_inputs=teacher_mlp_input,
                    target=target,
                    layer_index=layer_index,
                    tensor_root=tensor_root,
                    repository_commit=repository_commit,
                    preregistration_sha256=prereg_sha,
                    training_amendment_sha256=prereg_sha,
                    training_cache_sha256=FULL_CONTEXT_RECEIPT_SHA256,
                    expected_source_sha256=source_by_name[target["state_name"]][
                        "source_sha256"
                    ],
                    config=cq3_config,
                    row_shard=int(training["mlp_cq3"]["row_shard"]),
                )
            )
        gate_module = load_trained_cq3_linear(tensor_root, targets[0])
        up_module = load_trained_cq3_linear(tensor_root, targets[1])
        with torch.no_grad():
            candidate_down_input = (
                torch.nn.functional.silu(gate_module(candidate_mlp_input).float())
                * up_module(candidate_mlp_input).float()
            ).to(torch.bfloat16)
        descriptors.append(
            pilot_training.train_cq3_tensor(
                source=state[targets[2]["state_name"]],
                candidate_inputs=candidate_down_input,
                teacher_inputs=teacher_down_input,
                target=targets[2],
                layer_index=layer_index,
                tensor_root=tensor_root,
                repository_commit=repository_commit,
                preregistration_sha256=prereg_sha,
                training_amendment_sha256=prereg_sha,
                training_cache_sha256=FULL_CONTEXT_RECEIPT_SHA256,
                expected_source_sha256=source_by_name[targets[2]["state_name"]][
                    "source_sha256"
                ],
                config=cq3_config,
                row_shard=int(training["mlp_cq3"]["row_shard"]),
            )
        )
        down_module = load_trained_cq3_linear(tensor_root, targets[2])
        qat.set_child_module(
            block.layer,
            "mlp",
            pilot_direct.SelectedRowsCq3Mlp(
                gate_module,
                up_module,
                down_module,
                selected_rows=calibration_rows,
            ),
        )
        for target in targets:
            state.pop(target["state_name"])
        del candidate_down_input, teacher_down_input, teacher_mlp_input
        gc.collect()

        selected_parent = pilot_training.selected_rows(
            teacher_output, calibration_rows
        ).float()
        parent_energy = selected_parent.square().sum().detach()
        if float(parent_energy) <= 0:
            raise ValueError("all-layer training parent block energy is zero")

        def complete_block_calibration_loss() -> torch.Tensor:
            output = candidate_forward()
            selected = pilot_training.selected_rows(output, calibration_rows).float()
            return (selected - selected_parent).square().sum() / parent_energy

        continuous: list[torch.nn.Parameter] = []
        for payload in block.payloads.values():
            payload.codes_shadow.requires_grad_(False)
            payload.alpha_index_shadow.requires_grad_(False)
            payload.mu_index_shadow.requires_grad_(False)
            continuous.extend(
                [
                    payload.alpha_anchors,
                    payload.alpha_codebook,
                    payload.mu_anchors,
                    payload.mu_codebook,
                ]
            )
        continuous.extend(rotation.angles for rotation in block.rotations.values())

        def project_aff3() -> None:
            for payload in block.payloads.values():
                payload.project_training_state()

        aff3_config = training["attention_output_aff3"]
        aff3_training = calibration_only_hard_forward_trust_region(
            continuous,
            complete_block_calibration_loss,
            project_aff3,
            outer_steps=int(aff3_config["outer_steps"]),
            global_l2_radius=float(aff3_config["initial_global_l2_radius"]),
            backtracking_factors=tuple(
                float(value) for value in aff3_config["backtracking_factors"]
            ),
        )
        for record in aff3_training["history"]:
            print(
                json.dumps(
                    {
                        "event": "all_layer_aff3_outer_step_terminal",
                        "layer": layer_index,
                        **record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        with torch.no_grad():
            final_output = candidate_forward()
        final_calibration_nmse = qat.selected_nmse(
            final_output, teacher_output, calibration_rows
        )
        if abs(
            final_calibration_nmse
            - aff3_training["final_hard_forward_calibration_nmse"]
        ) > 1e-8:
            raise AssertionError("all-layer final AFF3 score changed before export")
        site_manifests, aff3_storage = qat.export_trained_block(
            block=block,
            artifact_root=args.artifact_root,
            layer_index=layer_index,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            parent_receipt_sha256=direct_receipt_sha,
        )
        cq3_storage = sum(item["payload"]["bytes"] for item in descriptors)
        relative_improvement = (
            frozen_direct_nmse - final_calibration_nmse
        ) / max(frozen_direct_nmse, 1e-30)
        maximum_churn = max(
            item["aggregate_calibration_metrics"][
                "weighted_selected_code_churn_fraction"
            ]
            for item in descriptors
        )

        pilot_payload_exact: bool | None = None
        pilot_score_exact: bool | None = None
        if layer_index in PILOT_LAYERS:
            old_root = args.pilot_trained_root / "layers" / str(layer_index)
            cq3_exact = all(
                item["payload"]["sha256"]
                == sha256_file(
                    old_root / "cq3" / f"{item['gguf_tensor']}.cq3"
                )
                for item in descriptors
            )
            old_manifest = json.loads(
                (old_root / "sites/attention_output/manifest.json").read_text()
            )
            pilot_payload_exact = cq3_exact and trained_aff3_fingerprint(
                site_manifests[0]
            ) == trained_aff3_fingerprint(old_manifest)
            pilot_score_exact = abs(
                final_calibration_nmse
                - float(
                    pilot_by_layer[layer_index][
                        "final_trained_calibration_complete_block_nmse"
                    ]
                )
            ) <= 1e-8
            if not pilot_payload_exact or not pilot_score_exact:
                raise ValueError("all-layer trained pilot reproduction changed")

        layer_passed = bool(
            math.isfinite(final_calibration_nmse)
            and final_calibration_nmse <= frozen_direct_nmse
            and maximum_churn
            <= float(training["mlp_cq3"]["maximum_weighted_code_churn_fraction"])
        )
        result = {
            "schema": LAYER_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "repository_commit": repository_commit,
            "preregistration_sha256": prereg_sha,
            "direct_receipt_sha256": direct_receipt_sha,
            "parent_receipt_sha256": direct_receipt_sha,
            "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
            "full_context_shard_sha256": shard_record["shard"]["sha256"],
            "layer": layer_index,
            "type": layer_type,
            "calibration_rows": calibration_rows.numel(),
            "frozen_direct_calibration_complete_block_nmse": frozen_direct_nmse,
            "reproduced_direct_calibration_complete_block_nmse": (
                reproduced_direct_nmse
            ),
            "final_trained_calibration_complete_block_nmse": (
                final_calibration_nmse
            ),
            "relative_calibration_nmse_improvement_vs_direct": (
                relative_improvement
            ),
            "maximum_tensor_weighted_code_churn_fraction": maximum_churn,
            "cq3_descriptors": [
                str(
                    Path("layers")
                    / f"{layer_index:02d}"
                    / "cq3"
                    / f"{target['gguf_name']}.json"
                )
                for target in targets
            ],
            "cq3_training": [
                {
                    "state_tensor": item["state_tensor"],
                    "gguf_tensor": item["gguf_tensor"],
                    "payload": item["payload"],
                    "aggregate_calibration_metrics": item[
                        "aggregate_calibration_metrics"
                    ],
                    "all_pack_unpack_mappings_exact": item[
                        "all_pack_unpack_mappings_exact"
                    ],
                }
                for item in descriptors
            ],
            "aff3_training": aff3_training,
            "site_manifests": [
                str(
                    Path("layers")
                    / str(layer_index)
                    / "sites"
                    / manifest["site"]
                    / "manifest.json"
                )
                for manifest in site_manifests
            ],
            "actual_persistent_payload_bytes": cq3_storage + aff3_storage,
            "calibration_not_regressed": final_calibration_nmse
            <= frozen_direct_nmse,
            "code_churn_within_maximum": maximum_churn
            <= float(training["mlp_cq3"]["maximum_weighted_code_churn_fraction"]),
            "pilot_payload_algebra_exact": pilot_payload_exact,
            "pilot_score_exact_within_1e_8": pilot_score_exact,
            "passed": layer_passed,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        cache_v1.atomic_json(result, terminal_path)
        validate_terminal_layer(
            result,
            artifact_root=args.artifact_root,
            repository_commit=repository_commit,
            preregistration_sha256=prereg_sha,
            direct_receipt_sha256=direct_receipt_sha,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "all_layer_trained_layer_terminal",
                    "layer": layer_index,
                    "direct_nmse": frozen_direct_nmse,
                    "trained_nmse": final_calibration_nmse,
                    "relative_improvement": relative_improvement,
                    "passed": layer_passed,
                    "pilot_exact": (
                        pilot_payload_exact and pilot_score_exact
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
            block,
            final_output,
            descriptors,
            gate_module,
            up_module,
            down_module,
            position_embeddings,
            masks,
        )
        gc.collect()
        trim_allocator()

    if len(results) != 64 or [item["layer"] for item in results] != list(range(64)):
        raise RuntimeError("all-layer trained terminal denominator changed")
    mean_improvement = sum(
        item["relative_calibration_nmse_improvement_vs_direct"] for item in results
    ) / len(results)
    total_storage = sum(item["actual_persistent_payload_bytes"] for item in results)
    gate = {
        "required_layers": 64,
        "observed_layers": len(results),
        "all_layers_terminal": all(item["terminal"] for item in results),
        "all_layers_calibration_not_regressed_vs_direct": all(
            item["calibration_not_regressed"] for item in results
        ),
        "mean_relative_calibration_nmse_reduction_vs_direct": mean_improvement,
        "minimum_mean_relative_calibration_nmse_reduction_vs_direct": float(
            prereg["gate"][
                "minimum_mean_relative_calibration_nmse_reduction_vs_direct"
            ]
        ),
        "all_code_churn_within_maximum": all(
            item["code_churn_within_maximum"] for item in results
        ),
        "all_payload_roundtrips_exact": all(
            all(
                tensor["payload"]["bytes"] > 0
                and tensor["all_pack_unpack_mappings_exact"] is True
                for tensor in item["cq3_training"]
            )
            for item in results
        ),
        "all_values_finite": all(
            math.isfinite(item["final_trained_calibration_complete_block_nmse"])
            for item in results
        ),
        "all_pilot_payload_algebras_exact": all(
            results[index]["pilot_payload_algebra_exact"] is True
            for index in PILOT_LAYERS
        ),
        "all_pilot_scores_exact_within_1e_8": all(
            results[index]["pilot_score_exact_within_1e_8"] is True
            for index in PILOT_LAYERS
        ),
        "total_selected_payload_bytes": total_storage,
        "maximum_selected_payload_bytes": int(
            prereg["gate"]["maximum_selected_payload_bytes"]
        ),
        "payload_storage_within_maximum": total_storage
        <= int(prereg["gate"]["maximum_selected_payload_bytes"]),
    }
    gate["passed"] = bool(
        gate["observed_layers"] == gate["required_layers"]
        and gate["all_layers_terminal"]
        and gate["all_layers_calibration_not_regressed_vs_direct"]
        and gate["mean_relative_calibration_nmse_reduction_vs_direct"]
        >= gate["minimum_mean_relative_calibration_nmse_reduction_vs_direct"]
        and gate["all_code_churn_within_maximum"]
        and gate["all_payload_roundtrips_exact"]
        and gate["all_values_finite"]
        and gate["all_pilot_payload_algebras_exact"]
        and gate["all_pilot_scores_exact_within_1e_8"]
        and gate["payload_storage_within_maximum"]
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "direct_receipt_sha256": direct_receipt_sha,
        "full_context_receipt_sha256": FULL_CONTEXT_RECEIPT_SHA256,
        "pilot_training_receipt_sha256": PILOT_TRAINING_RECEIPT_SHA256,
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
            "mean_relative_calibration_nmse_reduction_vs_direct": mean_improvement,
            "total_selected_payload_bytes": total_storage,
        },
        args.completion_marker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
