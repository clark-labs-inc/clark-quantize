#!/usr/bin/env python3
"""Fit and confirm a deployable calibration-span residual on three Qwen blocks."""

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
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUNTIME = Path(__file__).resolve().parent

import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_mixed_confirmation_v1 as prior_confirmation
import run_qwen38_27b_mixed_train_blocks_v1 as prior_training

from clark_quantize.qwen38_calibration_span_residual import (
    Int8ResidualAdapter,
    fit_float_residual_adapter,
    load_int8_adapter,
    normalized_mse,
    quantize_residual_adapter,
    write_int8_adapter,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET_LAYERS = (0, 32, 63)
CALIBRATION_SCHEMA = "qwen38_27b_calibration_span_residual_fit_result_v36"
CONFIRMATION_SCHEMA = "qwen38_27b_calibration_span_residual_confirmation_result_v36"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def selected_rows(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1]).index_select(0, rows).float()


def resource_guard(workspace: Path, minimum_free_disk_bytes: int) -> None:
    qat.resource_guard(workspace)
    stat = os.statvfs(workspace)
    if stat.f_bavail * stat.f_frsize < minimum_free_disk_bytes:
        raise SystemExit("infrastructure-invalid: residual pilot disk floor failed")


def validate_common(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    prereg = json.loads(args.preregistration.read_text())
    training_receipt = json.loads(args.training_receipt.read_text())
    prereg_sha = sha256_file(args.preregistration)
    training_sha = sha256_file(args.training_receipt)
    if (
        prereg.get("schema")
        != "qwen38_27b_calibration_span_residual_preregistration_v36"
        or prereg.get("status")
        != "frozen_after_terminal_qat_before_mixed_execution"
        or prereg["scope"].get("active_model_only") != MODEL_ID
        or prereg["scope"].get("revision") != MODEL_REVISION
        or tuple(prereg["scope"].get("pilot_layers", [])) != TARGET_LAYERS
        or prereg["scope"].get("qwen_4b_compute_forbidden") is not True
        or training_sha
        != prereg["base_candidate"]["training_receipt_sha256"]
        or training_receipt.get("schema")
        != prior_training.SCHEMA
        or training_receipt.get("gate", {}).get("passed") is not True
        or training_receipt.get("complete") is not True
        or training_receipt.get("terminal") is not True
    ):
        raise ValueError("v36 preregistration or base candidate identity changed")
    resource_guard(
        Path("/workspace"),
        int(prereg["execution_policy"]["minimum_free_disk_bytes"]),
    )
    return prereg, training_receipt, prereg_sha, training_sha


def load_context(
    config: Any,
    cache: dict[str, Any],
    layer_index: int,
    *,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]:
    hidden = cache["layers"][str(layer_index)]["input"]
    parent = cache["layers"][str(layer_index)]["output"]
    if selected.numel() == 0 or int(selected.min()) < 0:
        raise ValueError("v36 selected rows are empty or negative")
    sequence_length = int(hidden.shape[1])
    required_sequences = int(selected.max()) // sequence_length + 1
    if required_sequences > int(hidden.shape[0]):
        raise ValueError("v36 selected rows exceed the cached layer")
    # The frozen training cache contains additional validation sequences.  As in
    # the parent mixed-training receipt, replay only sequences containing the
    # selected rows.  Attention is sequence-local, so this is numerically exact
    # while avoiding needless full-block work on rows the streamed CQ3 MLP does
    # not materialize.
    hidden = hidden[:required_sequences]
    parent = parent[:required_sequences]
    attention_mask = cache["attention_mask"].long()[:required_sequences]
    position_ids = cache["text_position_ids"].long()[:required_sequences]
    position_embeddings, masks = qat.make_context(
        config, hidden, attention_mask, position_ids
    )
    return (
        hidden,
        parent,
        position_embeddings,
        masks[config.layer_types[layer_index]],
        position_ids,
    )


def trained_candidate_output(
    *,
    config: Any,
    model_dir: Path,
    weight_map: dict[str, str],
    cache: dict[str, Any],
    selected: torch.Tensor,
    layer_index: int,
    training_receipt: dict[str, Any],
    training_receipt_sha: str,
    prereg_sha: str,
    parent_artifact_root: Path,
    trained_artifact_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    by_layer = {int(item["layer"]): item for item in training_receipt["layers"]}
    trained_result = by_layer[layer_index]
    artifact_repository_commit = training_receipt["artifact_repository_commit"]
    prior_training.validate_terminal_layer(
        trained_result,
        artifact_root=trained_artifact_root,
        repository_commit=artifact_repository_commit,
        preregistration_sha256=training_receipt["preregistration_sha256"],
        training_amendment_sha256=training_receipt["training_amendment_sha256"],
        direct_receipt_sha256=training_receipt["direct_receipt_sha256"],
    )
    state = qat.layer_state(model_dir, weight_map, layer_index)
    hidden, parent, position_embeddings, layer_mask, position_ids = load_context(
        config, cache, layer_index, selected=selected
    )

    exact_layer = qat.PARENT.instantiate_layer(config, layer_index, state)
    with torch.no_grad():
        exact_output = qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=position_ids,
        )
    exact_nmse = qat.selected_nmse(exact_output, parent, selected)
    if exact_nmse > 1e-12:
        raise ValueError("exact BF16 layer no longer reproduces the cache")
    del exact_layer, exact_output

    block = qat.build_trainable_block(
        config=config,
        layer_index=layer_index,
        state=state,
        parent_artifact_root=parent_artifact_root,
        selected_sites={"attention_output"},
    )
    prior_confirmation.install_trained_attention_output(
        block=block,
        layer_index=layer_index,
        layer_type=config.layer_types[layer_index],
        parent_artifact_root=parent_artifact_root,
        trained_artifact_root=trained_artifact_root,
        repository_commit=artifact_repository_commit,
        preregistration_sha256=training_receipt["preregistration_sha256"],
        direct_receipt_sha256=training_receipt["direct_receipt_sha256"],
    )
    qat.set_child_module(
        block.layer,
        "mlp",
        prior_confirmation.load_trained_mlp(
            layer_index=layer_index,
            trained_artifact_root=trained_artifact_root,
            trained_layer_result=trained_result,
            rows=selected,
        ),
    )
    with torch.no_grad():
        candidate = qat.forward_with_grad(
            block.layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=position_ids,
        )
    inputs = selected_rows(hidden, selected)
    candidates = selected_rows(candidate, selected)
    targets = selected_rows(parent, selected)
    base_nmse = normalized_mse(candidates, targets)
    del block, state, candidate
    gc.collect()
    return inputs, candidates, targets, base_nmse


def adapter_descriptor(
    path: Path,
    adapter: Int8ResidualAdapter,
    *,
    layer: int,
    prereg_sha: str,
    training_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "qwen38_27b_calibration_span_residual_payload_v36",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "layer": layer,
        "input_features": int(adapter.left_codes.shape[0]),
        "output_features": int(adapter.right_codes.shape[1]),
        "rank": adapter.rank,
        "ridge_fraction": adapter.ridge_fraction,
        "format": "left_INT8_per_column_scale_FP16_then_right_INT8_per_row_scale_FP16_then_bias_FP16",
        "payload": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "complete": True,
        "terminal": True,
    }


def run_calibration(args: argparse.Namespace) -> int:
    prereg, training_receipt, prereg_sha, training_sha = validate_common(args)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("schema") != CALIBRATION_SCHEMA
            or existing.get("preregistration_sha256") != prereg_sha
            or existing.get("training_receipt_sha256") != training_sha
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing v36 calibration receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing["gate"]["passed"] else 2

    if sha256_file(args.training_cache) != prereg["data"]["training_cache"]["sha256"]:
        raise ValueError("v36 training cache identity changed")
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=False)
    calibration_rows = cache["calibration_flat_rows"].long()
    if calibration_rows.numel() != 64 or int(calibration_rows.max()) >= 8 * 64:
        raise ValueError("v36 calibration row denominator changed")
    fit_ordinals = torch.tensor(
        [index for index in range(64) if index % 4 != 3], dtype=torch.long
    )
    validation_ordinals = torch.tensor(
        [index for index in range(64) if index % 4 == 3], dtype=torch.long
    )
    if fit_ordinals.numel() != 48 or validation_ordinals.numel() != 16:
        raise AssertionError("v36 internal split changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    layer_data: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    for layer in TARGET_LAYERS:
        inputs, candidates, targets, base_nmse = trained_candidate_output(
            config=config,
            model_dir=args.model_dir,
            weight_map=weight_map,
            cache=cache,
            selected=calibration_rows,
            layer_index=layer,
            training_receipt=training_receipt,
            training_receipt_sha=training_sha,
            prereg_sha=prereg_sha,
            parent_artifact_root=args.parent_artifact_root,
            trained_artifact_root=args.trained_artifact_root,
        )
        layer_data[layer] = {
            "inputs": inputs,
            "candidates": candidates,
            "targets": targets,
            "base_full_nmse": base_nmse,
            "base_validation_nmse": normalized_mse(
                candidates.index_select(0, validation_ordinals),
                targets.index_select(0, validation_ordinals),
            ),
        }
        print(
            json.dumps(
                {
                    "event": "v36_base_layer_ready",
                    "layer": layer,
                    "base_full_nmse": base_nmse,
                    "base_validation_nmse": layer_data[layer]["base_validation_nmse"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    grid: list[dict[str, Any]] = []
    for rank in prereg["training"]["candidate_ranks"]:
        for ridge in prereg["training"]["candidate_ridge_fractions"]:
            layers = []
            for layer in TARGET_LAYERS:
                data = layer_data[layer]
                adapter = quantize_residual_adapter(
                    fit_float_residual_adapter(
                        data["inputs"].index_select(0, fit_ordinals),
                        data["candidates"].index_select(0, fit_ordinals),
                        data["targets"].index_select(0, fit_ordinals),
                        rank=int(rank),
                        ridge_fraction=float(ridge),
                    )
                )
                validation_inputs = data["inputs"].index_select(
                    0, validation_ordinals
                )
                validation_candidates = data["candidates"].index_select(
                    0, validation_ordinals
                )
                validation_targets = data["targets"].index_select(
                    0, validation_ordinals
                )
                corrected_nmse = normalized_mse(
                    validation_candidates + adapter.correction(validation_inputs),
                    validation_targets,
                )
                base_nmse = float(data["base_validation_nmse"])
                layers.append(
                    {
                        "layer": layer,
                        "base_validation_nmse": base_nmse,
                        "corrected_validation_nmse": corrected_nmse,
                        "relative_reduction": (base_nmse - corrected_nmse)
                        / max(base_nmse, 1e-30),
                    }
                )
            mean_nmse = sum(item["corrected_validation_nmse"] for item in layers) / 3
            no_regression = all(
                item["corrected_validation_nmse"] <= item["base_validation_nmse"]
                for item in layers
            )
            grid.append(
                {
                    "rank": rank,
                    "ridge_fraction": ridge,
                    "layers": layers,
                    "mean_validation_nmse": mean_nmse,
                    "all_layers_not_regressed": no_regression,
                }
            )
            print(
                json.dumps(
                    {"event": "v36_grid_candidate", **grid[-1]}, sort_keys=True
                ),
                flush=True,
            )

    admitted = [row for row in grid if row["all_layers_not_regressed"]]
    selected = min(
        admitted,
        key=lambda row: (
            row["mean_validation_nmse"],
            row["rank"],
            -row["ridge_fraction"],
        ),
    ) if admitted else None
    if selected is None:
        gate = {
            "passed": False,
            "reason": "no_global_rank_ridge_pair_avoids_validation_regression",
        }
        receipt = {
            "schema": CALIBRATION_SCHEMA,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "preregistration_sha256": prereg_sha,
            "training_receipt_sha256": training_sha,
            "grid": grid,
            "selected": None,
            "gate": gate,
            "elapsed_seconds": time.monotonic() - started,
            "complete": True,
            "terminal": True,
            "claim_boundary": prereg["claim_boundary"],
        }
        atomic_json(args.out, receipt)
        return 2

    validation_relative = [item["relative_reduction"] for item in selected["layers"]]
    validation_gate = prereg["gates"]["internal_validation"]
    validation_checks = {
        "all_three_layers_not_regressed": selected["all_layers_not_regressed"],
        "mean_relative_reduction_at_least_0_1": sum(validation_relative) / 3
        >= float(validation_gate["minimum_mean_relative_nmse_reduction"]),
        "layer63_relative_reduction_at_least_0_15": next(
            item["relative_reduction"]
            for item in selected["layers"]
            if item["layer"] == 63
        )
        >= float(validation_gate["minimum_layer63_relative_nmse_reduction"]),
    }

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    refit_layers = []
    payload_roundtrips = True
    for layer in TARGET_LAYERS:
        data = layer_data[layer]
        adapter = quantize_residual_adapter(
            fit_float_residual_adapter(
                data["inputs"],
                data["candidates"],
                data["targets"],
                rank=int(selected["rank"]),
                ridge_fraction=float(selected["ridge_fraction"]),
            )
        )
        payload_path = args.artifact_root / f"layer-{layer}.csr"
        write_int8_adapter(payload_path, adapter)
        restored = load_int8_adapter(
            payload_path,
            input_features=data["inputs"].shape[1],
            output_features=data["targets"].shape[1],
            rank=int(selected["rank"]),
            ridge_fraction=float(selected["ridge_fraction"]),
        )
        payload_roundtrips = payload_roundtrips and all(
            torch.equal(getattr(adapter, field), getattr(restored, field))
            for field in (
                "left_codes",
                "left_scales",
                "right_codes",
                "right_scales",
                "bias",
            )
        )
        corrected_nmse = normalized_mse(
            data["candidates"] + restored.correction(data["inputs"]),
            data["targets"],
        )
        base_nmse = float(data["base_full_nmse"])
        descriptor = adapter_descriptor(
            payload_path,
            restored,
            layer=layer,
            prereg_sha=prereg_sha,
            training_sha=training_sha,
        )
        descriptor_path = args.artifact_root / f"layer-{layer}.json"
        atomic_json(descriptor_path, descriptor)
        refit_layers.append(
            {
                "layer": layer,
                "base_calibration_nmse": base_nmse,
                "corrected_calibration_nmse": corrected_nmse,
                "relative_reduction": (base_nmse - corrected_nmse)
                / max(base_nmse, 1e-30),
                "descriptor": str(descriptor_path),
                "descriptor_sha256": sha256_file(descriptor_path),
                "payload": descriptor["payload"],
            }
        )

    rank = int(selected["rank"])
    per_layer_bytes = int(refit_layers[0]["payload"]["bytes"])
    projected_bytes = 64 * per_layer_bytes
    projected_macs = 64 * 2 * int(prereg["scope"]["hidden_size"]) * rank
    base_image = int(
        prereg["base_candidate"]["projected_complete_compute_image_bytes_before_residual"]
    )
    calibration_relative = [item["relative_reduction"] for item in refit_layers]
    deployment_gate = prereg["gates"]["deployment"]
    calibration_gate = prereg["gates"]["refit_calibration"]
    refit_checks = {
        "all_three_layers_not_regressed": all(
            item["corrected_calibration_nmse"] <= item["base_calibration_nmse"]
            for item in refit_layers
        ),
        "mean_relative_reduction_at_least_0_2": sum(calibration_relative) / 3
        >= float(calibration_gate["minimum_mean_relative_nmse_reduction"]),
        "all_payload_roundtrips_exact": payload_roundtrips,
    }
    deployment_checks = {
        "residual_bytes_at_most_41m": projected_bytes
        <= int(deployment_gate["maximum_residual_bytes_complete_64_layer_projection"]),
        "complete_compute_image_at_most_12_44b": base_image + projected_bytes
        <= int(deployment_gate["maximum_complete_compute_image_bytes"]),
        "residual_weight_fraction_at_most_0_004": projected_bytes / base_image
        <= float(deployment_gate["maximum_residual_weight_traffic_fraction"]),
        "residual_macs_at_most_32m": projected_macs
        <= int(
            deployment_gate[
                "maximum_residual_macs_per_token_complete_64_layer_projection"
            ]
        ),
    }
    all_checks = {
        **validation_checks,
        **refit_checks,
        **deployment_checks,
    }
    gate = {
        "checks": all_checks,
        "passed": all(all_checks.values()),
    }
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "training_cache_sha256": sha256_file(args.training_cache),
        "split": {
            "fit_ordinals": fit_ordinals.tolist(),
            "validation_ordinals": validation_ordinals.tolist(),
        },
        "grid": grid,
        "selected": selected,
        "refit_layers": refit_layers,
        "projection": {
            "rank": rank,
            "residual_bytes_complete_64_layers": projected_bytes,
            "complete_compute_image_bytes": base_image + projected_bytes,
            "residual_weight_traffic_fraction": projected_bytes / base_image,
            "residual_macs_per_token_complete_64_layers": projected_macs,
        },
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "next_stage": "collect_new_fresh_confirmation" if gate["passed"] else None,
        "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(args.out, receipt)
    print(json.dumps({"event": "v36_calibration_terminal", "gate": gate}, sort_keys=True))
    return 0 if gate["passed"] else 2


def run_confirmation(args: argparse.Namespace) -> int:
    prereg, training_receipt, prereg_sha, training_sha = validate_common(args)
    calibration = json.loads(args.calibration_receipt.read_text())
    cache_receipt = json.loads(args.confirmation_cache_receipt.read_text())
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("gate", {}).get("passed") is not True
        or calibration.get("terminal") is not True
        or calibration.get("preregistration_sha256") != prereg_sha
        or cache_receipt.get("schema")
        != "qwen38_27b_mixed_confirmation_cache_receipt_v1"
        or cache_receipt.get("preregistration_sha256") != prereg_sha
        or cache_receipt.get("cache", {}).get("sha256")
        != sha256_file(args.confirmation_cache)
        or cache_receipt.get("confirmation_rows") != 32
        or cache_receipt.get("complete") is not True
    ):
        raise ValueError("v36 calibration-to-confirmation handoff changed")
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("schema") != CONFIRMATION_SCHEMA
            or existing.get("calibration_receipt_sha256")
            != sha256_file(args.calibration_receipt)
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing v36 confirmation receipt changed")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing["gate"]["passed"] else 2

    cache = torch.load(args.confirmation_cache, map_location="cpu", weights_only=False)
    selected = cache["confirmation_flat_rows"].long()
    if selected.numel() != 32 or cache.get("confirmation_texts") != prereg["data"][
        "fresh_confirmation"
    ]["texts"]:
        raise ValueError("v36 fresh confirmation contents changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    rank = int(calibration["projection"]["rank"])
    ridge = float(calibration["selected"]["ridge_fraction"])
    layers = []
    started = time.monotonic()
    for layer in TARGET_LAYERS:
        inputs, candidates, targets, base_nmse = trained_candidate_output(
            config=config,
            model_dir=args.model_dir,
            weight_map=weight_map,
            cache=cache,
            selected=selected,
            layer_index=layer,
            training_receipt=training_receipt,
            training_receipt_sha=training_sha,
            prereg_sha=prereg_sha,
            parent_artifact_root=args.parent_artifact_root,
            trained_artifact_root=args.trained_artifact_root,
        )
        descriptor_path = args.artifact_root / f"layer-{layer}.json"
        descriptor = json.loads(descriptor_path.read_text())
        payload_path = args.artifact_root / f"layer-{layer}.csr"
        calibration_layer = next(
            item for item in calibration["refit_layers"] if item["layer"] == layer
        )
        if (
            descriptor.get("preregistration_sha256") != prereg_sha
            or descriptor.get("training_receipt_sha256") != training_sha
            or descriptor.get("rank") != rank
            or descriptor.get("ridge_fraction") != ridge
            or descriptor.get("payload") != calibration_layer["payload"]
            or sha256_file(descriptor_path)
            != calibration_layer["descriptor_sha256"]
            or sha256_file(payload_path) != descriptor["payload"]["sha256"]
        ):
            raise ValueError("v36 frozen adapter identity changed")
        adapter = load_int8_adapter(
            payload_path,
            input_features=inputs.shape[1],
            output_features=targets.shape[1],
            rank=rank,
            ridge_fraction=ridge,
        )
        corrected_nmse = normalized_mse(
            candidates + adapter.correction(inputs), targets
        )
        layers.append(
            {
                "layer": layer,
                "confirmation_rows": selected.numel(),
                "base_trained_mixed_nmse": base_nmse,
                "residual_corrected_nmse": corrected_nmse,
                "relative_reduction": (base_nmse - corrected_nmse)
                / max(base_nmse, 1e-30),
                "candidates_scored": [
                    "frozen_trained_mixed",
                    "frozen_trained_mixed_plus_int8_residual",
                ],
            }
        )
        print(
            json.dumps(
                {"event": "v36_confirmation_layer", **layers[-1]}, sort_keys=True
            ),
            flush=True,
        )

    gate_config = prereg["gates"]["fresh_confirmation"]
    relative = [item["relative_reduction"] for item in layers]
    checks = {
        "all_three_layers_not_regressed": all(
            item["residual_corrected_nmse"] <= item["base_trained_mixed_nmse"]
            for item in layers
        ),
        "mean_relative_reduction_at_least_0_1": sum(relative) / 3
        >= float(gate_config["minimum_mean_relative_nmse_reduction"]),
        "all_final_nmse_at_most_0_05": all(
            item["residual_corrected_nmse"]
            <= float(gate_config["maximum_final_complete_block_nmse_per_layer"])
            for item in layers
        ),
        "exactly_two_candidates_per_layer": all(
            len(item["candidates_scored"])
            == int(gate_config["exact_candidate_count_per_layer"])
            for item in layers
        ),
    }
    gate = {"checks": checks, "passed": all(checks.values())}
    receipt = {
        "schema": CONFIRMATION_SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "calibration_receipt_sha256": sha256_file(args.calibration_receipt),
        "confirmation_cache_receipt_sha256": sha256_file(
            args.confirmation_cache_receipt
        ),
        "confirmation_cache_sha256": sha256_file(args.confirmation_cache),
        "layers": layers,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "terminal": True,
        "next_stage": (
            "implement_and_benchmark_fused_avx2_skinny_residual_kernel"
            if gate["passed"]
            else None
        ),
        "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(args.out, receipt)
    print(json.dumps({"event": "v36_confirmation_terminal", "gate": gate}, sort_keys=True))
    return 0 if gate["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "confirmation"), required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt"),
    )
    parser.add_argument(
        "--parent-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact"),
    )
    parser.add_argument(
        "--trained-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/trained"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--confirmation-cache", type=Path)
    parser.add_argument("--confirmation-cache-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        parser.error("v36 is frozen to four CPU threads")
    if args.stage == "confirmation" and not all(
        (
            args.calibration_receipt,
            args.confirmation_cache,
            args.confirmation_cache_receipt,
        )
    ):
        parser.error("confirmation stage requires calibration and cache receipts")
    return args


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260819)
    return run_calibration(args) if args.stage == "calibration" else run_confirmation(args)


if __name__ == "__main__":
    raise SystemExit(main())
