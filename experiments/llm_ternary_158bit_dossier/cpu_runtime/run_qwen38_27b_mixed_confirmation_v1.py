#!/usr/bin/env python3
"""Score the two frozen Qwen3.8-27B mixed candidates on sealed confirmation."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUNTIME = Path(__file__).resolve().parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_mixed_direct_screen_v1 as direct
import run_qwen38_27b_mixed_train_blocks_v1 as train

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
from clark_quantize.qwen38_mixed_cq3_runtime import StreamedCq3Linear


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
SCHEMA = "qwen38_27b_calibration_ranked_mixed_confirmation_receipt_v1"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    direct.atomic_json(payload, path)


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
    qat.resource_guard(workspace)
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: mixed confirmation disk floor failed")


def load_trained_mlp(
    *,
    layer_index: int,
    trained_artifact_root: Path,
    trained_layer_result: dict[str, Any],
    rows: torch.Tensor,
) -> direct.SelectedRowsCq3Mlp:
    summaries = {
        item["state_tensor"]: item for item in trained_layer_result["cq3_training"]
    }
    modules = []
    for target in direct.mlp_targets(layer_index):
        descriptor_path = (
            trained_artifact_root
            / "layers"
            / str(layer_index)
            / "cq3"
            / f"{target['gguf_name']}.json"
        )
        payload_path = descriptor_path.with_suffix(".cq3")
        descriptor = json.loads(descriptor_path.read_text())
        summary = summaries[target["state_name"]]
        if (
            descriptor.get("schema") != train.TENSOR_SCHEMA
            or descriptor.get("state_tensor") != target["state_name"]
            or descriptor.get("gguf_tensor") != target["gguf_name"]
            or tuple(descriptor.get("shape", [])) != target["shape"]
            or descriptor.get("payload") != summary["payload"]
            or payload_path.stat().st_size != descriptor["payload"]["bytes"]
            or sha256_file(payload_path) != descriptor["payload"]["sha256"]
            or descriptor.get("all_pack_unpack_mappings_exact") is not True
            or descriptor.get("complete") is not True
            or descriptor.get("terminal") is not True
        ):
            raise ValueError("trained confirmation CQ3 payload identity changed")
        modules.append(
            StreamedCq3Linear.from_payload(
                payload_path,
                rows=target["shape"][0],
                columns=target["shape"][1],
                row_shard=128,
                output_dtype=torch.bfloat16,
            )
        )
    return direct.SelectedRowsCq3Mlp(*modules, selected_rows=rows)


def install_trained_attention_output(
    *,
    block: qat.TrainableBlock,
    layer_index: int,
    layer_type: str,
    parent_artifact_root: Path,
    trained_artifact_root: Path,
    repository_commit: str,
    preregistration_sha256: str,
    direct_receipt_sha256: str,
) -> None:
    parent_sites, _ = qat.parent_site_manifests(
        parent_artifact_root, layer_index, layer_type
    )
    parent_manifest = parent_sites["attention_output"]
    manifest_path = (
        trained_artifact_root
        / "layers"
        / str(layer_index)
        / "sites"
        / "attention_output"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "qwen38_affine_butterfly_qat_site_v1"
        or manifest.get("repository_commit") != repository_commit
        or manifest.get("preregistration_sha256") != preregistration_sha256
        or manifest.get("parent_receipt_sha256") != direct_receipt_sha256
        or manifest.get("site") != "attention_output"
        or manifest.get("tensor_names") != parent_manifest["tensor_names"]
        or manifest.get("complete") is not True
        or manifest.get("terminal") is not True
    ):
        raise ValueError("trained confirmation AFF3 manifest identity changed")
    angles = qat.PARENT.read_angles(trained_artifact_root, manifest["angles"])
    pairings = deterministic_givens_pairings(
        int(parent_manifest["input_dimension"]),
        stages=angles.shape[0],
        seed=int(parent_manifest["pairing_seed"]),
    )
    rotation = TrainableStoredRotation(angles, pairings)
    payloads: dict[str, TrainableAffinePayload] = {}
    for name, descriptor in zip(manifest["tensor_names"], manifest["tensors"]):
        verify_descriptor(trained_artifact_root, descriptor)
        payload = TrainableAffinePayload(
            load_compressed_affine_tensor(trained_artifact_root, descriptor)
        )
        payloads[name] = payload
        module_name, parameter_name = name.rsplit(".", 1)
        if parameter_name != "weight":
            raise ValueError("trained confirmation AFF3 target changed")
        qat.set_child_module(
            block.layer, module_name, StoredAffineLinear(payload, rotation)
        )
    block.payloads = payloads
    block.rotations = {"attention_output": rotation}
    block.names_by_site = {
        "attention_output": tuple(manifest["tensor_names"])
    }
    for parameter in block.layer.parameters():
        parameter.requires_grad_(False)
    block.layer.eval()


def confirmation_gate(
    layers: list[dict[str, Any]],
    *,
    maximum_final_nmse: float,
    minimum_mean_relative_reduction: float,
) -> dict[str, Any]:
    if {int(item["layer"]) for item in layers} != set(TARGET_LAYERS):
        raise ValueError("confirmation layer denominator changed")
    relative = [
        float(item["relative_confirmation_nmse_reduction_vs_direct"])
        for item in layers
    ]
    mean_relative = sum(relative) / len(relative)
    gate = {
        "layers": len(layers),
        "all_values_finite": all(
            math.isfinite(float(item["direct_confirmation_complete_block_nmse"]))
            and math.isfinite(float(item["trained_confirmation_complete_block_nmse"]))
            for item in layers
        ),
        "all_trained_layers_not_regressed_vs_direct_mixed": all(
            float(item["trained_confirmation_complete_block_nmse"])
            <= float(item["direct_confirmation_complete_block_nmse"])
            for item in layers
        ),
        "maximum_final_fresh_confirmation_complete_block_nmse_per_layer": maximum_final_nmse,
        "all_final_layers_below_maximum": all(
            float(item["trained_confirmation_complete_block_nmse"])
            <= maximum_final_nmse
            for item in layers
        ),
        "mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed": mean_relative,
        "minimum_mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed": minimum_mean_relative_reduction,
        "exact_candidates_scored_per_layer": 2,
    }
    gate["passed"] = bool(
        gate["all_values_finite"]
        and gate["all_trained_layers_not_regressed_vs_direct_mixed"]
        and gate["all_final_layers_below_maximum"]
        and gate[
            "mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed"
        ]
        >= gate[
            "minimum_mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed"
        ]
        and gate["exact_candidates_scored_per_layer"] == 2
    )
    return gate


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
        "--training-receipt",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/training.json",
    )
    parser.add_argument(
        "--confirmation-cache-receipt",
        type=Path,
        default=RUNTIME
        / "receipts/qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/confirmation-cache.json",
    )
    parser.add_argument(
        "--confirmation-cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/confirmation-cache.pt"
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
        "--trained-artifact-root",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/trained"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-calibration-ranked-mixed-confirmation-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("the frozen mixed confirmation uses four CPU threads")
    torch.set_num_threads(args.threads)

    preregistration_sha256 = sha256_file(args.preregistration)
    pre_execution_amendment_sha256 = sha256_file(args.pre_execution_amendment)
    training_amendment_sha256 = sha256_file(args.training_amendment)
    direct_receipt_sha256 = sha256_file(args.direct_receipt)
    training_receipt_sha256 = sha256_file(args.training_receipt)
    confirmation_cache_receipt_sha256 = sha256_file(
        args.confirmation_cache_receipt
    )
    confirmation_cache_sha256 = sha256_file(args.confirmation_cache)
    prereg = json.loads(args.preregistration.read_text())
    pre_execution_amendment = json.loads(args.pre_execution_amendment.read_text())
    training_amendment = json.loads(args.training_amendment.read_text())
    direct_receipt = json.loads(args.direct_receipt.read_text())
    training_receipt = json.loads(args.training_receipt.read_text())
    cache_receipt = json.loads(args.confirmation_cache_receipt.read_text())
    resource_guard(
        Path("/workspace"), int(prereg["execution_policy"]["minimum_free_disk_bytes"])
    )
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    artifact_repository_commit = training_receipt["artifact_repository_commit"]
    index_path = args.model_dir / "model.safetensors.index.json"
    if (
        preregistration_sha256
        != training_amendment["parent_preregistration_sha256"]
        or pre_execution_amendment_sha256
        != training_amendment["parent_pre_execution_amendment_sha256"]
        or direct_receipt_sha256 != training_amendment["gate_0_receipt_sha256"]
        or training_receipt.get("schema") != train.SCHEMA
        or training_receipt.get("gate", {}).get("passed") is not True
        or training_receipt.get("complete") is not True
        or training_receipt.get("terminal") is not True
        or cache_receipt.get("cache", {}).get("sha256")
        != confirmation_cache_sha256
        or cache_receipt.get("cache", {}).get("bytes")
        != args.confirmation_cache.stat().st_size
        or cache_receipt.get("complete") is not True
        or cache_receipt.get("terminal") is not True
        or prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or tuple(prereg["scope"]["pilot_layers"]) != TARGET_LAYERS
        or prereg["scope"]["qwen_4b_compute_forbidden"] is not True
        or pre_execution_amendment["corrections"][0]["field"]
        != "data.fresh_confirmation.may_score_only_the_final_frozen_candidate"
        or training_amendment["state_at_freeze"][
            "fresh_confirmation_had_been_scored"
        ]
        is not False
        or sha256_file(index_path) != qat.INDEX_SHA256
    ):
        raise ValueError("mixed confirmation identity or sequential gate changed")
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("schema") != SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("training_receipt_sha256") != training_receipt_sha256
            or existing.get("confirmation_cache_sha256")
            != confirmation_cache_sha256
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing mixed confirmation receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        raise SystemExit(0 if existing["gate"]["passed"] else 2)

    cache = torch.load(args.confirmation_cache, map_location="cpu", weights_only=False)
    expected_texts = prereg["data"]["fresh_confirmation"]["texts"]
    rows = cache["confirmation_flat_rows"].long()
    if (
        cache.get("schema") != "qwen38_27b_mixed_confirmation_cache_v1"
        or cache.get("model") != MODEL_ID
        or cache.get("revision") != MODEL_REVISION
        or cache.get("preregistration_sha256") != preregistration_sha256
        or cache.get("training_cache_sha256")
        != prereg["data"]["training_cache"]["sha256"]
        or cache.get("confirmation_texts") != expected_texts
        or rows.numel()
        != prereg["data"]["fresh_confirmation"]["selected_rows_per_layer"]
        or int(rows.max()) >= 4 * 64
    ):
        raise ValueError("sealed confirmation cache contents changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    attention_mask = cache["attention_mask"].long()
    position_ids = cache["text_position_ids"].long()
    weight_map = json.loads(index_path.read_text())["weight_map"]
    direct_by_layer = {
        int(item["layer"]): item for item in direct_receipt["layers"]
    }
    trained_by_layer = {
        int(item["layer"]): item for item in training_receipt["layers"]
    }
    started = time.monotonic()
    results: list[dict[str, Any]] = []

    for layer_index in TARGET_LAYERS:
        layer_started = time.monotonic()
        print(
            json.dumps(
                {"event": "mixed_confirmation_layer_started", "layer": layer_index},
                sort_keys=True,
            ),
            flush=True,
        )
        trained_result = trained_by_layer[layer_index]
        train.validate_terminal_layer(
            trained_result,
            artifact_root=args.trained_artifact_root,
            repository_commit=artifact_repository_commit,
            preregistration_sha256=preregistration_sha256,
            training_amendment_sha256=training_amendment_sha256,
            direct_receipt_sha256=direct_receipt_sha256,
        )
        state = qat.layer_state(args.model_dir, weight_map, layer_index)
        hidden = cache["layers"][str(layer_index)]["input"]
        parent = cache["layers"][str(layer_index)]["output"]
        position_embeddings, masks = qat.make_context(
            config, hidden, attention_mask, position_ids
        )
        layer_type = config.layer_types[layer_index]
        layer_mask = masks[layer_type]

        exact_layer = qat.PARENT.instantiate_layer(config, layer_index, state)
        with torch.no_grad():
            exact_output = qat.forward_with_grad(
                exact_layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
            )
        exact_cache_nmse = qat.selected_nmse(exact_output, parent, rows)
        if exact_cache_nmse > 1e-12:
            raise ValueError("exact BF16 layer no longer reproduces confirmation cache")
        del exact_layer, exact_output

        direct_block = qat.build_trainable_block(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
            selected_sites={"attention_output"},
        )
        qat.set_child_module(
            direct_block.layer,
            "mlp",
            train.direct_mlp(
                layer_index=layer_index,
                direct_root=args.direct_artifact_root,
                direct_layer_result=direct_by_layer[layer_index],
                rows=rows,
            ),
        )
        with torch.no_grad():
            direct_output = qat.forward_with_grad(
                direct_block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
            )
        direct_nmse = qat.selected_nmse(direct_output, parent, rows)
        del direct_block, direct_output
        gc.collect()

        trained_block = qat.build_trainable_block(
            config=config,
            layer_index=layer_index,
            state=state,
            parent_artifact_root=args.parent_artifact_root,
            selected_sites={"attention_output"},
        )
        install_trained_attention_output(
            block=trained_block,
            layer_index=layer_index,
            layer_type=layer_type,
            parent_artifact_root=args.parent_artifact_root,
            trained_artifact_root=args.trained_artifact_root,
            repository_commit=artifact_repository_commit,
            preregistration_sha256=preregistration_sha256,
            direct_receipt_sha256=direct_receipt_sha256,
        )
        qat.set_child_module(
            trained_block.layer,
            "mlp",
            load_trained_mlp(
                layer_index=layer_index,
                trained_artifact_root=args.trained_artifact_root,
                trained_layer_result=trained_result,
                rows=rows,
            ),
        )
        with torch.no_grad():
            trained_output = qat.forward_with_grad(
                trained_block.layer,
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
            )
        trained_nmse = qat.selected_nmse(trained_output, parent, rows)
        relative = (direct_nmse - trained_nmse) / max(direct_nmse, 1e-30)
        result = {
            "layer": layer_index,
            "type": layer_type,
            "confirmation_rows": rows.numel(),
            "exact_bf16_cache_reproduction_nmse": exact_cache_nmse,
            "direct_confirmation_complete_block_nmse": direct_nmse,
            "trained_confirmation_complete_block_nmse": trained_nmse,
            "relative_confirmation_nmse_reduction_vs_direct": relative,
            "candidates_scored": ["step_zero_direct_mixed", "final_trained_mixed"],
            "checkpoint_or_hyperparameter_selection_performed": False,
            "elapsed_seconds": time.monotonic() - layer_started,
            "complete": True,
            "terminal": True,
        }
        results.append(result)
        print(
            json.dumps(
                {"event": "mixed_confirmation_layer_terminal", **result},
                sort_keys=True,
            ),
            flush=True,
        )
        del trained_block, trained_output, state
        gc.collect()

    gate_config = prereg["sequential_gates"]["gate_1_trained_three_block_numeric"]
    gate = confirmation_gate(
        results,
        maximum_final_nmse=float(
            gate_config[
                "maximum_final_fresh_confirmation_complete_block_nmse_per_layer"
            ]
        ),
        minimum_mean_relative_reduction=float(
            gate_config[
                "minimum_mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed"
            ]
        ),
    )
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "artifact_repository_commit": artifact_repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "pre_execution_amendment_sha256": pre_execution_amendment_sha256,
        "training_amendment_sha256": training_amendment_sha256,
        "direct_receipt_sha256": direct_receipt_sha256,
        "training_receipt_sha256": training_receipt_sha256,
        "confirmation_cache_receipt_sha256": confirmation_cache_receipt_sha256,
        "confirmation_cache_sha256": confirmation_cache_sha256,
        "layers": results,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "claim_boundary": "terminal fresh-confirmation three-block numeric gate only; no complete-model conversion, capability, native-kernel, runtime, energy, or breakthrough result",
    }
    atomic_json(receipt, args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
