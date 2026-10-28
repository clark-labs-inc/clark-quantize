#!/usr/bin/env python3
"""Run restartable row-sharded Qwen3.8-27B layer-0 activation-aware QAT."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import safe_open

from clark_quantize.qwen38_layer0_qat import (
    RowShardTrainingConfig,
    aggregate_shard_metrics,
    expected_cq3_payload_bytes,
    pack_cq3_rows,
    train_row_shard,
    unpack_cq3_rows,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGETS = (
    {
        "hf_name": "model.language_model.layers.0.mlp.gate_proj.weight",
        "gguf_name": "blk.0.ffn_gate.weight",
        "shape": (17408, 5120),
        "activation_prefix": "mlp_input",
    },
    {
        "hf_name": "model.language_model.layers.0.mlp.up_proj.weight",
        "gguf_name": "blk.0.ffn_up.weight",
        "shape": (17408, 5120),
        "activation_prefix": "mlp_input",
    },
    {
        "hf_name": "model.language_model.layers.0.mlp.down_proj.weight",
        "gguf_name": "blk.0.ffn_down.weight",
        "shape": (5120, 17408),
        "activation_prefix": "down_input",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_payload(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def load_source_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def row_directory(root: Path, start: int, stop: int) -> Path:
    return root / f"rows-{start:05d}-{stop:05d}"


def validate_completed_row(
    directory: Path,
    *,
    source_sha256: str,
    cache_sha256: str,
    preregistration_sha256: str,
    start: int,
    stop: int,
    columns: int,
) -> tuple[bytes, dict[str, Any]]:
    receipt_path = directory / "receipt.json"
    payload_path = directory / "payload.bin"
    if not directory.is_dir() or not receipt_path.is_file() or not payload_path.is_file():
        raise ValueError(f"incomplete committed row shard: {directory}")
    receipt = json.loads(receipt_path.read_text())
    payload = payload_path.read_bytes()
    expected = expected_cq3_payload_bytes(stop - start, columns)
    checks = (
        receipt.get("schema") == "qwen38_27b_layer0_mlp_qat_row_shard_v1"
        and receipt.get("row_start") == start
        and receipt.get("row_stop") == stop
        and receipt.get("columns") == columns
        and receipt.get("source_tensor_sha256") == source_sha256
        and receipt.get("activation_cache_sha256") == cache_sha256
        and receipt.get("preregistration_sha256") == preregistration_sha256
        and receipt.get("payload_bytes") == expected == len(payload)
        and receipt.get("payload_sha256") == hashlib.sha256(payload).hexdigest()
        and receipt.get("complete") is True
    )
    if not checks:
        raise ValueError(f"committed row shard identity mismatch: {directory}")
    codes, scales = unpack_cq3_rows(payload, rows=stop - start, columns=columns)
    if not bool(torch.isfinite(scales.float()).all()) or bool(((codes < -1) | (codes > 1)).any()):
        raise ValueError(f"committed row shard payload is invalid: {directory}")
    return payload, receipt["metrics"]


def commit_row(
    root: Path,
    *,
    start: int,
    stop: int,
    columns: int,
    payload: bytes,
    metrics: dict[str, Any],
    source_sha256: str,
    cache_sha256: str,
    preregistration_sha256: str,
) -> Path:
    final = row_directory(root, start, stop)
    if final.exists():
        raise FileExistsError(final)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".rows-{start:05d}-{stop:05d}.", dir=root))
    try:
        payload_path = temporary / "payload.bin"
        with payload_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {
            "schema": "qwen38_27b_layer0_mlp_qat_row_shard_v1",
            "row_start": start,
            "row_stop": stop,
            "columns": columns,
            "source_tensor_sha256": source_sha256,
            "activation_cache_sha256": cache_sha256,
            "preregistration_sha256": preregistration_sha256,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "metrics": metrics,
            "complete": True,
        }
        with (temporary / "receipt.json").open("w") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        return final
    except BaseException:
        # An uncommitted dot-directory is derived data and safe to retry after
        # inspection; it is intentionally not mistaken for a completed shard.
        raise


def train_tensor(
    target: dict[str, Any],
    *,
    source: torch.Tensor,
    calibration_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    teacher_calibration_inputs: torch.Tensor | None,
    teacher_validation_inputs: torch.Tensor | None,
    work_root: Path,
    payload_root: Path,
    row_shard: int,
    config: RowShardTrainingConfig,
    source_sha256: str,
    cache_sha256: str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    rows, columns = target["shape"]
    if tuple(source.shape) != (rows, columns):
        raise ValueError(f"source shape mismatch for {target['hf_name']}: {tuple(source.shape)}")
    tensor_root = work_root / target["gguf_name"]
    incomplete = sorted(path for path in tensor_root.glob(".rows-*") if path.is_dir())
    if incomplete:
        raise RuntimeError(f"incomplete row shard requires inspection: {incomplete[0]}")
    metrics: list[dict[str, Any]] = []
    payload_paths: list[Path] = []
    for start in range(0, rows, row_shard):
        stop = min(start + row_shard, rows)
        directory = row_directory(tensor_root, start, stop)
        if directory.exists():
            _, row_metrics = validate_completed_row(
                directory,
                source_sha256=source_sha256,
                cache_sha256=cache_sha256,
                preregistration_sha256=preregistration_sha256,
                start=start,
                stop=stop,
                columns=columns,
            )
        else:
            codes, scales, row_metrics = train_row_shard(
                source[start:stop],
                calibration_inputs,
                validation_inputs,
                config=config,
                teacher_calibration_inputs=teacher_calibration_inputs,
                teacher_validation_inputs=teacher_validation_inputs,
            )
            payload = pack_cq3_rows(codes, scales)
            checked_codes, checked_scales = unpack_cq3_rows(
                payload, rows=stop - start, columns=columns
            )
            if not torch.equal(checked_codes, codes) or not torch.equal(checked_scales, scales):
                raise AssertionError("CQ3_0 row payload did not round-trip exactly")
            directory = commit_row(
                tensor_root,
                start=start,
                stop=stop,
                columns=columns,
                payload=payload,
                metrics=row_metrics,
                source_sha256=source_sha256,
                cache_sha256=cache_sha256,
                preregistration_sha256=preregistration_sha256,
            )
            print(
                json.dumps(
                    {
                        "tensor": target["hf_name"],
                        "row_start": start,
                        "row_stop": stop,
                        "accepted": row_metrics["accepted"],
                        "relative_validation_improvement": row_metrics[
                            "relative_validation_improvement"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        metrics.append(row_metrics)
        payload_paths.append(directory / "payload.bin")

    aggregate = aggregate_shard_metrics(metrics)
    aggregate["tensor"] = target["hf_name"]
    aggregate["gguf_tensor"] = target["gguf_name"]
    output = payload_root / f"{target['gguf_name']}.cq3"
    if output.exists():
        if output.stat().st_size != expected_cq3_payload_bytes(rows, columns):
            raise ValueError(f"existing complete tensor payload has wrong size: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for path in payload_paths:
                    with path.open("rb") as source_handle:
                        shutil.copyfileobj(source_handle, destination, length=8 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            if temporary.stat().st_size != expected_cq3_payload_bytes(rows, columns):
                raise ValueError("assembled tensor payload byte count mismatch")
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    aggregate["payload"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    atomic_json(aggregate, output.with_suffix(".json"))
    return aggregate


@torch.no_grad()
def linear_from_cq3_payload(
    payload: Path,
    inputs: torch.Tensor,
    *,
    rows: int,
    columns: int,
    row_shard: int,
) -> torch.Tensor:
    if inputs.ndim != 2 or inputs.shape[1] != columns:
        raise ValueError("CQ3 linear input shape mismatch")
    row_bytes = expected_cq3_payload_bytes(1, columns)
    if payload.stat().st_size != row_bytes * rows:
        raise ValueError("CQ3 linear payload byte count mismatch")
    result = torch.empty(inputs.shape[0], rows, dtype=torch.float32)
    with payload.open("rb") as handle:
        for start in range(0, rows, row_shard):
            stop = min(start + row_shard, rows)
            raw = handle.read((stop - start) * row_bytes)
            codes, scales = unpack_cq3_rows(raw, rows=stop - start, columns=columns)
            weight = (
                codes.reshape(stop - start, -1, 128).float()
                * scales.float().unsqueeze(-1)
            ).reshape(stop - start, columns)
            result[:, start:stop] = functional.linear(inputs.float(), weight)
        consumed = handle.tell()
    if consumed != payload.stat().st_size or not bool(torch.isfinite(result).all()):
        raise ValueError("CQ3 streamed linear did not consume a finite complete payload")
    return result


def candidate_down_inputs(
    *,
    payload_root: Path,
    work_root: Path,
    calibration_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    cache_sha256: str,
    row_shard: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_path = payload_root / "blk.0.ffn_gate.weight.cq3"
    up_path = payload_root / "blk.0.ffn_up.weight.cq3"
    cache_path = work_root / "candidate-down-inputs.pt"
    receipt_path = work_root / "candidate-down-inputs.json"
    parents = {"gate": sha256_file(gate_path), "up": sha256_file(up_path)}
    if cache_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            receipt.get("schema") != "qwen38_27b_layer0_candidate_down_inputs_v1"
            or receipt.get("parents") != parents
            or receipt.get("activation_cache_sha256") != cache_sha256
            or receipt.get("cache_sha256") != sha256_file(cache_path)
            or cached.get("schema") != "qwen38_27b_layer0_candidate_down_inputs_v1"
        ):
            raise ValueError("candidate down-input cache identity mismatch")
        return cached["calibration"].float(), cached["validation"].float()
    if cache_path.exists() or receipt_path.exists():
        raise RuntimeError("partial candidate down-input cache requires inspection")

    gate_calibration = linear_from_cq3_payload(
        gate_path, calibration_inputs, rows=17408, columns=5120, row_shard=row_shard
    )
    up_calibration = linear_from_cq3_payload(
        up_path, calibration_inputs, rows=17408, columns=5120, row_shard=row_shard
    )
    gate_validation = linear_from_cq3_payload(
        gate_path, validation_inputs, rows=17408, columns=5120, row_shard=row_shard
    )
    up_validation = linear_from_cq3_payload(
        up_path, validation_inputs, rows=17408, columns=5120, row_shard=row_shard
    )
    calibration = functional.silu(gate_calibration) * up_calibration
    validation = functional.silu(gate_validation) * up_validation
    del gate_calibration, up_calibration, gate_validation, up_validation
    calibration_cached = calibration.to(torch.float16)
    validation_cached = validation.to(torch.float16)
    cached = {
        "schema": "qwen38_27b_layer0_candidate_down_inputs_v1",
        "calibration": calibration_cached,
        "validation": validation_cached,
    }
    atomic_torch_save(cached, cache_path)
    atomic_json(
        {
            "schema": "qwen38_27b_layer0_candidate_down_inputs_v1",
            "parents": parents,
            "activation_cache_sha256": cache_sha256,
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "calibration_shape": list(calibration.shape),
            "validation_shape": list(validation.shape),
            "finite": bool(torch.isfinite(calibration).all() and torch.isfinite(validation).all()),
        },
        receipt_path,
    )
    return calibration_cached.float(), validation_cached.float()


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
        "--cache-receipt", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json"),
    )
    parser.add_argument(
        "--work-root", type=Path,
        default=Path("/workspace/quant-runs/qwen38-27b-layer0-mlp-activation-qat-v1/row-shards"),
    )
    parser.add_argument(
        "--payload-root", type=Path,
        default=Path("/workspace/quant-runs/qwen38-27b-layer0-mlp-activation-qat-v1/payloads"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-v1.json"),
    )
    args = parser.parse_args()

    preregistration_sha256 = sha256_file(args.preregistration)
    prereg = json.loads(args.preregistration.read_text())
    training = prereg["training"]
    numeric_gate = prereg["numeric_promotion_gate"]
    if (
        prereg["scope"]["active_model_only"] != MODEL_ID
        or prereg["scope"]["revision"] != MODEL_REVISION
        or {item["hf_name"] for item in TARGETS} != set(prereg["scope"]["target_tensors"])
    ):
        raise ValueError("Qwen3.8-27B training preregistration identity mismatch")
    available = shutil.disk_usage(args.payload_root.parent).free
    minimum_free = int(training["resource_limits"]["minimum_free_disk_bytes"])
    if available < minimum_free:
        raise RuntimeError(f"training disk gate requires {minimum_free}, found {available}")

    cache_receipt = json.loads(args.cache_receipt.read_text())
    cache_sha256 = sha256_file(args.cache)
    if (
        cache_receipt.get("complete") is not True
        or cache_receipt.get("finite") is not True
        or cache_receipt.get("model_id") != MODEL_ID
        or cache_receipt.get("model_revision") != MODEL_REVISION
        or cache_receipt.get("preregistration", {}).get("sha256") != preregistration_sha256
        or cache_receipt.get("cache", {}).get("sha256") != cache_sha256
    ):
        raise ValueError("activation cache receipt is invalid")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if (
        cache.get("schema") != "qwen38_27b_layer0_mlp_activation_cache_v1"
        or cache.get("model_id") != MODEL_ID
        or cache.get("model_revision") != MODEL_REVISION
        or cache.get("preregistration_sha256") != preregistration_sha256
    ):
        raise ValueError("activation cache identity mismatch")
    activations: dict[str, torch.Tensor] = {}
    for prefix, columns in (("mlp_input", 5120), ("down_input", 17408)):
        for split in ("calibration", "validation"):
            key = f"{prefix}_{split}"
            tensor = cache.get(key)
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != 2
                or tensor.shape[1] != columns
                or not bool(torch.isfinite(tensor.float()).all())
            ):
                raise ValueError(f"invalid cached activation tensor: {key}")
            activations[key] = tensor.float().contiguous()

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    torch.set_num_threads(int(training["resource_limits"]["cpu_threads"]))
    torch.manual_seed(int(training["seed"]))
    config = RowShardTrainingConfig(
        steps=int(training["steps_per_row_shard"]),
        learning_rate=float(training["learning_rate"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        weight_reconstruction_weight=float(training["weight_reconstruction_weight"]),
        group_size=int(training["group_size"]),
    )
    summaries: list[dict[str, Any]] = []
    source_receipts: dict[str, Any] = {}
    candidate_down_calibration: torch.Tensor | None = None
    candidate_down_validation: torch.Tensor | None = None
    for target in TARGETS:
        if target["activation_prefix"] == "down_input" and candidate_down_calibration is None:
            candidate_down_calibration, candidate_down_validation = candidate_down_inputs(
                payload_root=args.payload_root,
                work_root=args.work_root,
                calibration_inputs=activations["mlp_input_calibration"],
                validation_inputs=activations["mlp_input_validation"],
                cache_sha256=cache_sha256,
                row_shard=int(training["row_shard"]),
            )
        source = load_source_tensor(args.model_dir, weight_map, target["hf_name"])
        source_sha256 = tensor_sha256(source)
        cached_source = cache_receipt.get("source_tensors", {}).get(target["hf_name"])
        if cached_source is not None and cached_source.get("sha256") != source_sha256:
            raise ValueError(f"source tensor changed after activation collection: {target['hf_name']}")
        source_receipts[target["hf_name"]] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "sha256": source_sha256,
            "shard": weight_map[target["hf_name"]],
        }
        prefix = target["activation_prefix"]
        candidate_calibration = (
            candidate_down_calibration
            if prefix == "down_input"
            else activations[f"{prefix}_calibration"]
        )
        candidate_validation = (
            candidate_down_validation
            if prefix == "down_input"
            else activations[f"{prefix}_validation"]
        )
        if candidate_calibration is None or candidate_validation is None:
            raise AssertionError("candidate activation selection failed")
        summary = train_tensor(
            target,
            source=source,
            calibration_inputs=candidate_calibration,
            validation_inputs=candidate_validation,
            teacher_calibration_inputs=(
                activations["down_input_calibration"] if prefix == "down_input" else None
            ),
            teacher_validation_inputs=(
                activations["down_input_validation"] if prefix == "down_input" else None
            ),
            work_root=args.work_root,
            payload_root=args.payload_root,
            row_shard=int(training["row_shard"]),
            config=config,
            source_sha256=source_sha256,
            cache_sha256=cache_sha256,
            preregistration_sha256=preregistration_sha256,
        )
        summaries.append(summary)
        del source
        gc.collect()

    initial_error = sum(item["initial_validation_squared_error_sum"] for item in summaries)
    selected_error = sum(item["selected_validation_squared_error_sum"] for item in summaries)
    target_energy = sum(item["validation_target_squared_sum"] for item in summaries)
    parameters = sum(math.prod(item["shape"]) for item in TARGETS)
    weighted_churn = sum(
        item["weighted_selected_code_churn_fraction"] * math.prod(target["shape"])
        for item, target in zip(summaries, TARGETS)
    ) / parameters
    relative_improvement = (initial_error - selected_error) / max(initial_error, 1e-12)
    checks = {
        "cache_receipts_valid": True,
        "all_shards_finite": all(item["all_finite"] for item in summaries),
        "minimum_global_relative_validation_nmse_improvement": relative_improvement
        >= float(numeric_gate["minimum_global_relative_validation_nmse_improvement"]),
        "no_tensor_validation_nmse_regression": all(
            item["weighted_selected_validation_nmse"]
            <= item["weighted_initial_validation_nmse"]
            for item in summaries
        ),
        "maximum_weighted_code_churn_fraction": weighted_churn
        <= float(numeric_gate["maximum_weighted_code_churn_fraction"]),
        "exact_payload_bytes": all(
            item["payload"]["bytes"] == expected_cq3_payload_bytes(*target["shape"])
            for item, target in zip(summaries, TARGETS)
        ),
    }
    receipt = {
        "schema": "qwen38_27b_layer0_mlp_activation_qat_receipt_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": preregistration_sha256,
        },
        "activation_cache": {
            "path": str(args.cache),
            "sha256": cache_sha256,
            "receipt": str(args.cache_receipt),
            "receipt_sha256": sha256_file(args.cache_receipt),
        },
        "source_tensors": source_receipts,
        "training_config": training,
        "tensors": summaries,
        "aggregate": {
            "parameters": parameters,
            "initial_validation_squared_error_sum": initial_error,
            "selected_validation_squared_error_sum": selected_error,
            "validation_target_squared_sum": target_energy,
            "initial_validation_nmse": initial_error / max(target_energy, 1e-12),
            "selected_validation_nmse": selected_error / max(target_energy, 1e-12),
            "relative_validation_nmse_improvement": relative_improvement,
            "weighted_selected_code_churn_fraction": weighted_churn,
        },
        "checks": checks,
        "numeric_promotion_pass": all(checks.values()),
        "complete": True,
        "claim_boundary": "numeric layerwise QAT receipt only; CUTE, MATH, code, MMLU, WikiText, and paired throughput remain mandatory",
    }
    atomic_json(receipt, args.out)
    marker = args.out.parent / (
        "QWEN38_LAYER0_MLP_ACTIVATION_QAT_NUMERIC_PASS"
        if receipt["numeric_promotion_pass"]
        else "QWEN38_LAYER0_MLP_ACTIVATION_QAT_NUMERIC_REJECTED"
    )
    atomic_payload(("pass\n" if receipt["numeric_promotion_pass"] else "rejected\n").encode(), marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["numeric_promotion_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
