#!/usr/bin/env python3
"""Train the frozen Qwen3.8-27B CQ3/AFF3 three-block pilot on CPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUNTIME = Path(__file__).resolve().parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_mixed_direct_screen_v1 as direct

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_affine_artifact import sha256_file
from clark_quantize.qwen38_layer0_qat import (
    RowShardTrainingConfig,
    expected_cq3_payload_bytes,
    pack_cq3_rows,
    unpack_cq3_rows,
)
from clark_quantize.qwen38_mixed_aff3_trust_region import (
    calibration_only_hard_forward_trust_region,
)
from clark_quantize.qwen38_mixed_cq3_qat import (
    aggregate_calibration_metrics,
    train_row_shard_calibration_only,
)
from clark_quantize.qwen38_mixed_cq3_runtime import (
    StreamedCq3Linear,
    tensor_sha256,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
SCHEMA = "qwen38_27b_calibration_ranked_mixed_training_receipt_v1"
LAYER_SCHEMA = "qwen38_27b_calibration_ranked_mixed_trained_layer_v1"
TENSOR_SCHEMA = "qwen38_27b_calibration_ranked_mixed_trained_cq3_tensor_v1"
SHARD_SCHEMA = "qwen38_27b_calibration_ranked_mixed_trained_cq3_shard_v1"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    direct.atomic_json(payload, path)


def int64_tensor_sha256(tensor: torch.Tensor) -> str:
    if tensor.dtype != torch.int64:
        raise ValueError("row identity tensor must be int64")
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
    qat.resource_guard(workspace)
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: mixed training disk floor failed")


def selected_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, values.shape[-1]).index_select(0, rows).contiguous()


def capture_exact_teacher_activations(
    layer: torch.nn.Module,
    forward: Callable[[], torch.Tensor],
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def post_norm_hook(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured["mlp_input"] = selected_rows(output.detach(), rows)

    def down_pre_hook(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured["down_input"] = selected_rows(inputs[0].detach(), rows)

    handles = [
        layer.post_attention_layernorm.register_forward_hook(post_norm_hook),
        layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook),
    ]
    try:
        with torch.no_grad():
            output = forward()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"mlp_input", "down_input"}:
        raise RuntimeError("exact teacher MLP activation capture changed")
    return output, captured["mlp_input"], captured["down_input"]


def capture_candidate_mlp_input(
    layer: torch.nn.Module,
    forward: Callable[[], torch.Tensor],
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured.append(selected_rows(output.detach(), rows))

    handle = layer.post_attention_layernorm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = forward()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("candidate MLP activation capture changed")
    return output, captured[0]


def _validate_shard(
    descriptor: dict[str, Any],
    payload_path: Path,
    *,
    identities: dict[str, Any],
    start: int,
    stop: int,
    source_sha256: str,
) -> None:
    if (
        descriptor.get("schema") != SHARD_SCHEMA
        or any(descriptor.get(key) != value for key, value in identities.items())
        or descriptor.get("row_start") != start
        or descriptor.get("row_stop") != stop
        or descriptor.get("source_shard_sha256") != source_sha256
        or descriptor.get("payload", {}).get("bytes") != payload_path.stat().st_size
        or descriptor.get("payload", {}).get("sha256") != sha256_file(payload_path)
        or descriptor.get("complete") is not True
        or descriptor.get("terminal") is not True
    ):
        raise ValueError("existing CQ3 training shard identity changed")


def train_cq3_tensor(
    *,
    source: torch.Tensor,
    candidate_inputs: torch.Tensor,
    teacher_inputs: torch.Tensor,
    target: dict[str, Any],
    layer_index: int,
    tensor_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    training_amendment_sha256: str,
    training_cache_sha256: str,
    expected_source_sha256: str,
    config: RowShardTrainingConfig,
    row_shard: int,
) -> dict[str, Any]:
    if tuple(source.shape) != target["shape"] or source.dtype != torch.bfloat16:
        raise ValueError("CQ3 training source identity changed")
    if row_shard != 128 or source.shape[0] % row_shard:
        raise ValueError("CQ3 frozen row shard changed")
    source_sha = tensor_sha256(source)
    if source_sha != expected_source_sha256:
        raise ValueError("CQ3 source no longer matches the authenticated gate-0 tensor")
    candidate_sha = tensor_sha256(candidate_inputs)
    teacher_sha = tensor_sha256(teacher_inputs)
    identities = {
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "training_amendment_sha256": training_amendment_sha256,
        "training_cache_sha256": training_cache_sha256,
        "layer": layer_index,
        "state_tensor": target["state_name"],
        "gguf_tensor": target["gguf_name"],
        "candidate_inputs_sha256": candidate_sha,
        "teacher_inputs_sha256": teacher_sha,
    }
    payload_path = tensor_root / f"{target['gguf_name']}.cq3"
    descriptor_path = tensor_root / f"{target['gguf_name']}.json"
    if payload_path.exists() or descriptor_path.exists():
        if not payload_path.is_file() or not descriptor_path.is_file():
            raise RuntimeError("partial terminal CQ3 tensor requires inspection")
        descriptor = json.loads(descriptor_path.read_text())
        if (
            descriptor.get("schema") != TENSOR_SCHEMA
            or any(descriptor.get(key) != value for key, value in identities.items())
            or descriptor.get("source_sha256") != source_sha
            or tuple(descriptor.get("shape", [])) != tuple(source.shape)
            or descriptor.get("payload", {}).get("bytes") != payload_path.stat().st_size
            or descriptor.get("payload", {}).get("sha256")
            != sha256_file(payload_path)
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("existing terminal CQ3 tensor identity changed")
        codes, scales = unpack_cq3_rows(
            payload_path.read_bytes(), rows=source.shape[0], columns=source.shape[1]
        )
        if not bool(torch.isfinite(scales.float()).all()):
            raise ValueError("existing terminal CQ3 tensor has non-finite scales")
        del codes, scales
        return descriptor

    shards_root = tensor_root / "active-shards" / target["gguf_name"]
    shard_records: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for start in range(0, source.shape[0], row_shard):
        stop = start + row_shard
        shard_source = source[start:stop]
        source_shard_sha = tensor_sha256(shard_source)
        shard_payload_path = shards_root / f"{start:06d}.cq3"
        shard_descriptor_path = shards_root / f"{start:06d}.json"
        if shard_payload_path.exists() or shard_descriptor_path.exists():
            if not shard_payload_path.is_file() or not shard_descriptor_path.is_file():
                raise RuntimeError("partial CQ3 row shard requires inspection")
            shard_descriptor = json.loads(shard_descriptor_path.read_text())
            _validate_shard(
                shard_descriptor,
                shard_payload_path,
                identities=identities,
                start=start,
                stop=stop,
                source_sha256=source_shard_sha,
            )
            payload = shard_payload_path.read_bytes()
            codes, scales = unpack_cq3_rows(
                payload, rows=row_shard, columns=source.shape[1]
            )
            del codes, scales
        else:
            codes, scales, metrics = train_row_shard_calibration_only(
                shard_source,
                candidate_inputs,
                teacher_calibration_inputs=teacher_inputs,
                config=config,
            )
            payload = pack_cq3_rows(codes, scales)
            checked_codes, checked_scales = unpack_cq3_rows(
                payload, rows=row_shard, columns=source.shape[1]
            )
            if not torch.equal(codes, checked_codes) or not torch.equal(
                scales, checked_scales
            ):
                raise AssertionError("trained CQ3 shard changed on exact roundtrip")
            atomic_bytes(shard_payload_path, payload)
            shard_descriptor = {
                "schema": SHARD_SCHEMA,
                **identities,
                "row_start": start,
                "row_stop": stop,
                "source_shard_sha256": source_shard_sha,
                "payload": {
                    "bytes": len(payload),
                    "sha256": sha256_file(shard_payload_path),
                },
                "metrics": metrics,
                "complete": True,
                "terminal": True,
            }
            atomic_json(shard_descriptor, shard_descriptor_path)
            print(
                json.dumps(
                    {
                        "event": "mixed_cq3_shard_terminal",
                        "layer": layer_index,
                        "tensor": target["state_name"],
                        "row_start": start,
                        "row_stop": stop,
                        "selected_step": metrics["selected_step"],
                        "relative_calibration_nmse_improvement": metrics[
                            "relative_calibration_nmse_improvement"
                        ],
                        "code_churn_fraction": metrics[
                            "selected_code_churn_fraction"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del codes, scales, checked_codes, checked_scales
        shard_records.append(shard_descriptor)
        payload_parts.append(payload)

    metrics = aggregate_calibration_metrics(
        [record["metrics"] for record in shard_records]
    )
    payload = b"".join(payload_parts)
    expected = expected_cq3_payload_bytes(source.shape[0], source.shape[1])
    if len(payload) != expected:
        raise AssertionError("trained terminal CQ3 tensor byte count changed")
    atomic_bytes(payload_path, payload)
    codes, scales = unpack_cq3_rows(
        payload, rows=source.shape[0], columns=source.shape[1]
    )
    if not bool(torch.isfinite(scales.float()).all()):
        raise ValueError("trained terminal CQ3 tensor has non-finite scales")
    descriptor = {
        "schema": TENSOR_SCHEMA,
        **identities,
        "shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "source_sha256": source_sha,
        "row_shard": row_shard,
        "group_size": 128,
        "steps_per_shard": config.steps,
        "payload": {
            "bytes": len(payload),
            "sha256": sha256_file(payload_path),
        },
        "aggregate_calibration_metrics": metrics,
        "shards": [
            {
                "row_start": record["row_start"],
                "row_stop": record["row_stop"],
                "source_shard_sha256": record["source_shard_sha256"],
                "payload": record["payload"],
                "metrics": record["metrics"],
            }
            for record in shard_records
        ],
        "all_pack_unpack_mappings_exact": True,
        "complete": True,
        "terminal": True,
    }
    atomic_json(descriptor, descriptor_path)
    shutil.rmtree(shards_root)
    try:
        shards_root.parent.rmdir()
    except OSError:
        # Another tensor may have restartable active shards in the same block.
        pass
    del codes, scales, payload, payload_parts
    return descriptor


def load_cq3_linear(
    tensor_root: Path, target: dict[str, Any]
) -> StreamedCq3Linear:
    return StreamedCq3Linear.from_payload(
        tensor_root / f"{target['gguf_name']}.cq3",
        rows=target["shape"][0],
        columns=target["shape"][1],
        row_shard=128,
        output_dtype=torch.bfloat16,
    )


def direct_mlp(
    *,
    layer_index: int,
    direct_root: Path,
    direct_layer_result: dict[str, Any],
    rows: torch.Tensor,
) -> direct.SelectedRowsCq3Mlp:
    by_name = {
        item["state_tensor"]: item for item in direct_layer_result["cq3_payloads"]
    }
    modules = []
    for target in direct.mlp_targets(layer_index):
        record = by_name[target["state_name"]]
        path = (
            direct_root
            / "layers"
            / str(layer_index)
            / "payloads"
            / f"{target['gguf_name']}.cq3"
        )
        if (
            path.stat().st_size != record["payload"]["bytes"]
            or sha256_file(path) != record["payload"]["sha256"]
        ):
            raise ValueError("gate-0 CQ3 payload identity changed")
        modules.append(
            StreamedCq3Linear.from_payload(
                path,
                rows=target["shape"][0],
                columns=target["shape"][1],
                row_shard=128,
                output_dtype=torch.bfloat16,
            )
        )
    return direct.SelectedRowsCq3Mlp(*modules, selected_rows=rows)


def validate_terminal_layer(
    result: dict[str, Any],
    *,
    artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    training_amendment_sha256: str,
    direct_receipt_sha256: str,
) -> None:
    if (
        result.get("schema") != LAYER_SCHEMA
        or result.get("repository_commit") != repository_commit
        or result.get("preregistration_sha256") != preregistration_sha256
        or result.get("training_amendment_sha256") != training_amendment_sha256
        or result.get("direct_receipt_sha256") != direct_receipt_sha256
        or result.get("complete") is not True
        or result.get("terminal") is not True
    ):
        raise ValueError("existing mixed trained layer identity changed")
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
            descriptor.get("schema") != TENSOR_SCHEMA
            or descriptor.get("repository_commit") != repository_commit
            or descriptor.get("training_amendment_sha256")
            != training_amendment_sha256
            or descriptor.get("payload", {}).get("sha256")
            != sha256_file(payload_path)
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("existing mixed CQ3 tensor descriptor changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=RUNTIME / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1.json",
    )
    parser.add_argument(
        "--pre-execution-amendment",
        type=Path,
        default=RUNTIME
        / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_pre_execution_amendment_v1.json",
    )
    parser.add_argument(
        "--training-amendment",
        type=Path,
        default=RUNTIME
        / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_training_execution_amendment_v1.json",
    )
    parser.add_argument(
        "--direct-receipt",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/direct-screen.json",
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"
        ),
    )
    parser.add_argument(
        "--parent-artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"
        ),
    )
    parser.add_argument(
        "--direct-artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/direct"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/trained"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-ranked-mixed-training-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen mixed training pilot uses four CPU threads")
    torch.set_num_threads(args.threads)

    preregistration_sha256 = sha256_file(args.preregistration)
    pre_execution_amendment_sha256 = sha256_file(args.pre_execution_amendment)
    training_amendment_sha256 = sha256_file(args.training_amendment)
    direct_receipt_sha256 = sha256_file(args.direct_receipt)
    training_cache_sha256 = sha256_file(args.training_cache)
    prereg = json.loads(args.preregistration.read_text())
    pre_execution_amendment = json.loads(args.pre_execution_amendment.read_text())
    training_amendment = json.loads(args.training_amendment.read_text())
    direct_receipt = json.loads(args.direct_receipt.read_text())
    execution = prereg["execution_policy"]
    resource_guard(Path("/workspace"), int(execution["minimum_free_disk_bytes"]))
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        preregistration_sha256
        != training_amendment["parent_preregistration_sha256"]
        or pre_execution_amendment_sha256
        != training_amendment["parent_pre_execution_amendment_sha256"]
        or direct_receipt_sha256 != training_amendment["gate_0_receipt_sha256"]
        or training_cache_sha256 != prereg["data"]["training_cache"]["sha256"]
        or prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or tuple(prereg["scope"]["pilot_layers"]) != TARGET_LAYERS
        or prereg["scope"]["qwen_4b_compute_forbidden"] is not True
        or pre_execution_amendment["execution_state_at_freeze"][
            "mixed_training_started"
        ]
        is not False
        or training_amendment["state_at_freeze"]["mixed_training_had_started"]
        is not False
        or training_amendment["state_at_freeze"][
            "fresh_confirmation_had_been_scored"
        ]
        is not False
        or direct_receipt.get("schema")
        != direct.SCHEMA
        or direct_receipt.get("gate", {}).get("passed") is not True
        or direct_receipt.get("complete") is not True
        or direct_receipt.get("terminal") is not True
        or sha256_file(index_path) != qat.INDEX_SHA256
    ):
        raise ValueError("mixed training identity or gate-0 handoff changed")
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("schema") != SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("training_amendment_sha256")
            != training_amendment_sha256
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing mixed training receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    hidden_by_layer = cache["layers"]
    attention_mask = cache["attention_mask"].long()[:8]
    position_ids = cache["text_position_ids"].long()[:8]
    calibration_rows = cache["calibration_flat_rows"].long()
    if (
        calibration_rows.numel() != 64
        or int(calibration_rows.max()) >= 8 * 64
        or int64_tensor_sha256(calibration_rows)
        != training_amendment["calibration_identity"][
            "selected_flat_int64_rows_sha256"
        ]
    ):
        raise ValueError("mixed training calibration denominator changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    direct_by_layer = {
        int(item["layer"]): item for item in direct_receipt["layers"]
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
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    for layer_index in TARGET_LAYERS:
        terminal_path = (
            args.artifact_root / "layers" / str(layer_index) / "terminal-result.json"
        )
        if terminal_path.exists():
            result = json.loads(terminal_path.read_text())
            validate_terminal_layer(
                result,
                artifact_root=args.artifact_root,
                repository_commit=repository_commit,
                preregistration_sha256=preregistration_sha256,
                training_amendment_sha256=training_amendment_sha256,
                direct_receipt_sha256=direct_receipt_sha256,
            )
            results.append(result)
            print(
                json.dumps(
                    {"event": "mixed_trained_layer_reused", "layer": layer_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        layer_started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "mixed_trained_layer_started",
                    "layer": layer_index,
                    "type": config.layer_types[layer_index],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        state = qat.layer_state(args.model_dir, weight_map, layer_index)
        hidden = hidden_by_layer[str(layer_index)]["input"][:8]
        parent = hidden_by_layer[str(layer_index)]["output"][:8]
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        layer_mask = masks[config.layer_types[layer_index]]
        exact_layer = qat.PARENT.instantiate_layer(config, layer_index, state)

        def exact_forward() -> torch.Tensor:
            return qat.forward_with_grad(
                exact_layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
            )

        exact_output, teacher_mlp_input, teacher_down_input = (
            capture_exact_teacher_activations(
                exact_layer, exact_forward, calibration_rows
            )
        )
        exact_cache_nmse = qat.selected_nmse(exact_output, parent, calibration_rows)
        if exact_cache_nmse > 1e-12:
            raise ValueError("exact BF16 layer no longer reproduces the training cache")

        block = qat.build_trainable_block(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
            selected_sites={"attention_output"},
        )
        qat.set_child_module(
            block.layer,
            "mlp",
            direct_mlp(
                layer_index=layer_index,
                direct_root=args.direct_artifact_root,
                direct_layer_result=direct_by_layer[layer_index],
                rows=calibration_rows,
            ),
        )

        def candidate_forward() -> torch.Tensor:
            return qat.forward_with_grad(
                block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
            )

        direct_output, candidate_mlp_input = capture_candidate_mlp_input(
            block.layer, candidate_forward, calibration_rows
        )
        reproduced_direct_nmse = qat.selected_nmse(
            direct_output, parent, calibration_rows
        )
        frozen_direct_nmse = float(
            direct_by_layer[layer_index][
                "direct_mixed_calibration_complete_block_nmse"
            ]
        )
        if abs(reproduced_direct_nmse - frozen_direct_nmse) > 1e-8:
            raise ValueError("gate-0 direct mixed score did not reproduce")
        qat.set_child_module(block.layer, "mlp", torch.nn.Identity())
        del direct_output, exact_output, exact_layer
        gc.collect()

        tensor_root = args.artifact_root / "layers" / str(layer_index) / "cq3"
        targets = direct.mlp_targets(layer_index)
        descriptors: list[dict[str, Any]] = []
        for target in targets[:2]:
            descriptors.append(
                train_cq3_tensor(
                    source=state[target["state_name"]],
                    candidate_inputs=candidate_mlp_input,
                    teacher_inputs=teacher_mlp_input,
                    target=target,
                    layer_index=layer_index,
                    tensor_root=tensor_root,
                    repository_commit=repository_commit,
                    preregistration_sha256=preregistration_sha256,
                    training_amendment_sha256=training_amendment_sha256,
                    training_cache_sha256=training_cache_sha256,
                    expected_source_sha256={
                        item["state_tensor"]: item["source_sha256"]
                        for item in direct_by_layer[layer_index]["cq3_payloads"]
                    }[target["state_name"]],
                    config=cq3_config,
                    row_shard=int(training["mlp_cq3"]["row_shard"]),
                )
            )
        gate_module = load_cq3_linear(tensor_root, targets[0])
        up_module = load_cq3_linear(tensor_root, targets[1])
        with torch.no_grad():
            candidate_down_input = (
                torch.nn.functional.silu(gate_module(candidate_mlp_input).float())
                * up_module(candidate_mlp_input).float()
            ).to(torch.bfloat16)
        descriptors.append(
            train_cq3_tensor(
                source=state[targets[2]["state_name"]],
                candidate_inputs=candidate_down_input,
                teacher_inputs=teacher_down_input,
                target=targets[2],
                layer_index=layer_index,
                tensor_root=tensor_root,
                repository_commit=repository_commit,
                preregistration_sha256=preregistration_sha256,
                training_amendment_sha256=training_amendment_sha256,
                training_cache_sha256=training_cache_sha256,
                expected_source_sha256={
                    item["state_tensor"]: item["source_sha256"]
                    for item in direct_by_layer[layer_index]["cq3_payloads"]
                }[targets[2]["state_name"]],
                config=cq3_config,
                row_shard=int(training["mlp_cq3"]["row_shard"]),
            )
        )
        down_module = load_cq3_linear(tensor_root, targets[2])
        qat.set_child_module(
            block.layer,
            "mlp",
            direct.SelectedRowsCq3Mlp(
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

        selected_parent = selected_rows(parent, calibration_rows).float()
        parent_energy = selected_parent.square().sum().detach()
        if float(parent_energy) <= 0:
            raise ValueError("mixed training parent block energy is zero")

        def complete_block_calibration_loss() -> torch.Tensor:
            output = candidate_forward()
            selected = selected_rows(output, calibration_rows).float()
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
                        "event": "mixed_aff3_outer_step_terminal",
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
            final_output, parent, calibration_rows
        )
        if abs(
            final_calibration_nmse
            - aff3_training["final_hard_forward_calibration_nmse"]
        ) > 1e-8:
            raise AssertionError("final AFF3 calibration score changed before export")
        site_manifests, aff3_storage = qat.export_trained_block(
            block=block,
            artifact_root=args.artifact_root,
            layer_index=layer_index,
            repository_commit=repository_commit,
            preregistration_sha256=preregistration_sha256,
            parent_receipt_sha256=direct_receipt_sha256,
        )
        cq3_storage = sum(item["payload"]["bytes"] for item in descriptors)
        relative_calibration_improvement = (
            (frozen_direct_nmse - final_calibration_nmse)
            / max(frozen_direct_nmse, 1e-30)
        )
        maximum_churn = max(
            item["aggregate_calibration_metrics"][
                "weighted_selected_code_churn_fraction"
            ]
            for item in descriptors
        )
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
            "preregistration_sha256": preregistration_sha256,
            "pre_execution_amendment_sha256": pre_execution_amendment_sha256,
            "training_amendment_sha256": training_amendment_sha256,
            "direct_receipt_sha256": direct_receipt_sha256,
            "parent_receipt_sha256": direct_receipt_sha256,
            "training_cache_sha256": training_cache_sha256,
            "layer": layer_index,
            "type": config.layer_types[layer_index],
            "calibration_rows": calibration_rows.numel(),
            "exact_bf16_cache_reproduction_nmse": exact_cache_nmse,
            "frozen_direct_calibration_complete_block_nmse": frozen_direct_nmse,
            "reproduced_direct_calibration_complete_block_nmse": reproduced_direct_nmse,
            "final_trained_calibration_complete_block_nmse": final_calibration_nmse,
            "relative_calibration_nmse_improvement_vs_direct": relative_calibration_improvement,
            "maximum_tensor_weighted_code_churn_fraction": maximum_churn,
            "cq3_descriptors": [
                str(
                    Path("layers")
                    / str(layer_index)
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
            "passed": layer_passed,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        atomic_json(result, terminal_path)
        results.append(result)
        print(
            json.dumps(
                {"event": "mixed_trained_layer_terminal", **result},
                sort_keys=True,
            ),
            flush=True,
        )
        del final_output, block, state, descriptors, gate_module, up_module, down_module
        gc.collect()

    mean_relative_improvement = sum(
        item["relative_calibration_nmse_improvement_vs_direct"] for item in results
    ) / len(results)
    gate_config = prereg["sequential_gates"]["gate_1_trained_three_block_numeric"]
    total_storage = sum(item["actual_persistent_payload_bytes"] for item in results)
    gate = {
        "layers": len(results),
        "all_layers_terminal": len(results) == len(TARGET_LAYERS)
        and all(item["terminal"] for item in results),
        "all_layers_calibration_not_regressed_vs_direct": all(
            item["calibration_not_regressed"] for item in results
        ),
        "mean_relative_calibration_nmse_reduction_vs_direct": mean_relative_improvement,
        "minimum_mean_relative_calibration_nmse_reduction_vs_direct_mixed": float(
            gate_config[
                "minimum_mean_relative_calibration_nmse_reduction_vs_direct_mixed"
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
        "pilot_persistent_payload_bytes": total_storage,
        "maximum_complete_compute_image_bytes": int(
            gate_config["maximum_complete_compute_image_bytes"]
        ),
        "projected_complete_compute_image_bytes_before_alignment": int(
            prereg["complete_runtime_storage_accounting"][
                "projected_complete_compute_image_bytes_before_alignment"
            ]
        ),
    }
    gate["storage_projection_within_maximum"] = bool(
        gate["projected_complete_compute_image_bytes_before_alignment"]
        <= gate["maximum_complete_compute_image_bytes"]
    )
    gate["passed"] = bool(
        gate["all_layers_terminal"]
        and gate["all_layers_calibration_not_regressed_vs_direct"]
        and gate["mean_relative_calibration_nmse_reduction_vs_direct"]
        >= gate["minimum_mean_relative_calibration_nmse_reduction_vs_direct_mixed"]
        and gate["all_code_churn_within_maximum"]
        and gate["all_payload_roundtrips_exact"]
        and gate["all_values_finite"]
        and gate["storage_projection_within_maximum"]
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "pre_execution_amendment_sha256": pre_execution_amendment_sha256,
        "training_amendment_sha256": training_amendment_sha256,
        "direct_receipt_sha256": direct_receipt_sha256,
        "training_cache_sha256": training_cache_sha256,
        "layers": results,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "claim_boundary": "terminal calibration-only three-block training subgate; fresh confirmation remains sealed and there is no complete-model quality, runtime, energy, or breakthrough result",
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
