#!/usr/bin/env python3
"""Train a storage-neutral SwiGLU diagonal-gauge CQ3 pilot on layer 59.

For an MLP ``down(silu(gate(x)) * up(x))`` and positive diagonal ``D``, the
unquantized transformation ``up' = D up`` and ``down' = down D^-1`` is exact.
This pilot folds D into the existing trained up-projection CQ3 row scales and
re-trains a CQ3 representation of ``down D^-1`` against the resulting mixed
topology.  No runtime sidecar or additional inference operation is introduced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as functional


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUNTIME))

import collect_qwen38_27b_affine_butterfly_block_cache_v1 as cache_v1
import collect_qwen38_27b_full_context_block_cache_v42 as cache_v42
import run_qwen38_27b_affine_butterfly_block_gate_v1 as block_parent
import run_qwen38_27b_affine_butterfly_block_qat_v1 as qat
import run_qwen38_27b_all_layer_mixed_direct_v43 as direct_v43
import run_qwen38_27b_mixed_train_blocks_v1 as mixed_training

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_affine_artifact import sha256_file
from clark_quantize.qwen38_layer0_qat import (
    RowShardTrainingConfig,
    pack_cq3_rows,
    unpack_cq3_rows,
)
from clark_quantize.qwen38_mixed_cq3_qat import normalized_squared_error
from clark_quantize.qwen38_mixed_cq3_runtime import (
    StreamedCq3Linear,
    tensor_sha256,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYER = 59
ROWS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
PAYLOAD_BYTES = 23674880
SCHEMA = "qwen38_27b_swiglu_diagonal_gauge_pilot_receipt_v71"


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def tensor_nmse(candidate: torch.Tensor, target: torch.Tensor) -> float:
    return float(normalized_squared_error(candidate.float(), target.float()))


def derive_column_rms_gauge(
    down_weight: torch.Tensor, *, epsilon: float, minimum: float, maximum: float
) -> torch.Tensor:
    if (
        down_weight.ndim != 2
        or not 0 < minimum <= 1 <= maximum
        or epsilon <= 0
    ):
        raise ValueError("invalid diagonal-gauge inputs")
    column_rms = down_weight.float().square().mean(dim=0).add(epsilon).sqrt()
    normalization = column_rms.median()
    if not bool(torch.isfinite(normalization)) or float(normalization) <= 0:
        raise ValueError("invalid diagonal-gauge normalization")
    return (column_rms / normalization).clamp(min=minimum, max=maximum)


def load_cq3(path: Path, *, rows: int, columns: int) -> StreamedCq3Linear:
    return StreamedCq3Linear.from_payload(
        path,
        rows=rows,
        columns=columns,
        row_shard=128,
        output_dtype=torch.bfloat16,
    )


def descriptor_by_tensor(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layer = receipt["layers"][LAYER]
    if layer.get("layer") != LAYER or len(layer.get("cq3_training", [])) != 3:
        raise ValueError("v44 layer-59 training denominator changed")
    return {item["gguf_tensor"]: item for item in layer["cq3_training"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--full-context-receipt", required=True, type=Path)
    parser.add_argument(
        "--full-context-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-full-context-block-cache-v42"),
    )
    parser.add_argument("--v44-receipt", required=True, type=Path)
    parser.add_argument(
        "--v44-artifact-root",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"),
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads != 4:
        raise ValueError("v71 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v71 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v71 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = file_sha256(args.preregistration)
    full_context_sha = file_sha256(args.full_context_receipt)
    v44_sha = file_sha256(args.v44_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        prereg.get("schema")
        != "qwen38_27b_swiglu_diagonal_gauge_pilot_preregistration_v71"
        or prereg.get("status") != "frozen_before_execution"
        or prereg.get("model", {}).get("id") != MODEL_ID
        or prereg.get("model", {}).get("revision") != MODEL_REVISION
        or prereg.get("scope", {}).get("layer") != LAYER
        or prereg.get("authenticated_inputs", {}).get("full_context_receipt_sha256")
        != full_context_sha
        or prereg.get("authenticated_inputs", {}).get("v44_training_receipt_sha256")
        != v44_sha
        or context_receipt.get("gate", {}).get("passed") is not True
        or context_receipt.get("complete") is not True
        or context_receipt.get("terminal") is not True
        or v44.get("gate", {}).get("passed") is not True
        or v44.get("complete") is not True
        or v44.get("terminal") is not True
    ):
        raise ValueError("v71 frozen protocol or authenticated parent changed")

    index_path = args.model_dir / "model.safetensors.index.json"
    if file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("Qwen3.8-27B model index changed")
    layer_record = context_receipt["layers"][LAYER]
    shard_path = args.full_context_root / "layers" / f"{LAYER}.pt"
    context_path = args.full_context_root / "context.pt"
    if (
        file_sha256(shard_path) != layer_record["shard"]["sha256"]
        or shard_path.stat().st_size != layer_record["shard"]["bytes"]
        or file_sha256(context_path) != context_receipt["context"]["sha256"]
    ):
        raise ValueError("v71 full-context artifacts changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    if config.num_hidden_layers != 64 or config.hidden_size != HIDDEN:
        raise ValueError("Qwen3.8-27B architecture changed")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    expected_sources = prereg["authenticated_inputs"]["source_tensor_sha256"]
    for name, expected in expected_sources.items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v71 source tensor changed: {name}")

    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    calibration_rows = context["calibration_flat_rows"].long()
    attention_mask = context["attention_mask"].long()[:8]
    position_ids = context["text_position_ids"].long()[:8]
    hidden = shard["input"][:8].contiguous()
    teacher_layer_output = shard["output"][:8].contiguous()
    exact_layer = block_parent.instantiate_layer(config, LAYER, state)
    position_embeddings, masks = qat.make_context(
        config, hidden, attention_mask, position_ids
    )
    layer_type = config.layer_types[LAYER]

    def exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=position_ids,
        )

    exact_output, mlp_input, teacher_down_input = (
        mixed_training.capture_exact_teacher_activations(
            exact_layer, exact_forward, calibration_rows
        )
    )
    if not torch.equal(exact_output, teacher_layer_output):
        raise ValueError("v71 exact layer replay changed")
    del exact_layer, exact_output, teacher_layer_output, hidden, shard
    gc.collect()

    trained = descriptor_by_tensor(v44)
    payload_root = args.v44_artifact_root / "layers" / str(LAYER) / "cq3"
    payloads = {
        role: payload_root / f"blk.{LAYER}.ffn_{role}.weight.cq3"
        for role in ("gate", "up", "down")
    }
    for role, path in payloads.items():
        record = trained[f"blk.{LAYER}.ffn_{role}.weight"]["payload"]
        if (
            not path.is_file()
            or path.stat().st_size != PAYLOAD_BYTES
            or record["bytes"] != PAYLOAD_BYTES
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(f"v71 trained {role} payload changed")

    gate = load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
    source_down = state["mlp.down_proj.weight"]
    with torch.no_grad():
        exact_mlp_output = functional.linear(
            teacher_down_input.float(), source_down.float()
        )
        gate_output = gate(mlp_input).float()
        up_output = up(mlp_input).float()
        baseline_down_input = (
            functional.silu(gate_output) * up_output
        ).to(torch.bfloat16)
        baseline_mlp_output = down(baseline_down_input).float()
        baseline_nmse = tensor_nmse(baseline_mlp_output, exact_mlp_output)

    gauge_spec = prereg["gauge"]
    with torch.no_grad():
        gauge = derive_column_rms_gauge(
            source_down,
            epsilon=float(gauge_spec["epsilon"]),
            minimum=float(gauge_spec["minimum"]),
            maximum=float(gauge_spec["maximum"]),
        )
    if (
        gauge.shape != (INTERMEDIATE,)
        or not bool(torch.isfinite(gauge).all())
        or float(gauge.min()) < float(gauge_spec["minimum"])
        or float(gauge.max()) > float(gauge_spec["maximum"])
    ):
        raise ValueError("v71 analytic gauge is invalid")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    gauge_path = args.artifact_root / "layer59-gauge-f32.pt"
    if not gauge_path.exists():
        temporary = gauge_path.with_suffix(".pt.tmp")
        torch.save(gauge.contiguous(), temporary)
        os.replace(temporary, gauge_path)
    else:
        stored_gauge = torch.load(gauge_path, map_location="cpu", weights_only=True)
        if not torch.equal(stored_gauge, gauge):
            raise ValueError("existing v71 gauge changed")

    scaled_up_path = args.artifact_root / "blk.59.ffn_up.gauged.weight.cq3"
    up_codes, up_scales = unpack_cq3_rows(
        payloads["up"].read_bytes(), rows=INTERMEDIATE, columns=HIDDEN
    )
    gauged_up_scales = (up_scales.float() * gauge[:, None]).to(torch.float16)
    if not scaled_up_path.exists():
        atomic_bytes(scaled_up_path, pack_cq3_rows(up_codes, gauged_up_scales))
    checked_codes, checked_scales = unpack_cq3_rows(
        scaled_up_path.read_bytes(), rows=INTERMEDIATE, columns=HIDDEN
    )
    up_codes_unchanged = torch.equal(checked_codes, up_codes)
    up_scales_exact = torch.equal(checked_scales, gauged_up_scales)
    if not up_codes_unchanged or not up_scales_exact:
        raise AssertionError("v71 gauged-up payload did not roundtrip exactly")
    del up_codes, up_scales, checked_codes, checked_scales, gauged_up_scales
    gc.collect()

    gauged_up = load_cq3(scaled_up_path, rows=INTERMEDIATE, columns=HIDDEN)
    with torch.no_grad():
        gauged_up_output = gauged_up(mlp_input).float()
        gauged_candidate_input = (
            functional.silu(gate_output) * gauged_up_output
        ).to(torch.bfloat16)
        ideal_scaled_candidate_input = (
            baseline_down_input.float() * gauge[None, :]
        )
        up_fold_nmse = tensor_nmse(
            gauged_candidate_input.float(), ideal_scaled_candidate_input
        )
        gauged_teacher_input = (
            teacher_down_input.float() * gauge[None, :]
        ).to(torch.bfloat16)
        gauged_source = (
            source_down.float() / gauge[None, :]
        ).to(torch.bfloat16)
        gauged_exact_output = functional.linear(
            gauged_teacher_input.float(), gauged_source.float()
        )
        gauge_identity_nmse = tensor_nmse(gauged_exact_output, exact_mlp_output)
    del gauged_up, gauged_up_output, ideal_scaled_candidate_input
    gc.collect()

    target = {
        "state_name": "mlp.down_proj.weight",
        "gguf_name": "blk.59.ffn_down_gauged.weight",
        "shape": (HIDDEN, INTERMEDIATE),
    }
    training_spec = prereg["training"]
    training_config = RowShardTrainingConfig(
        steps=int(training_spec["steps_per_row_shard"]),
        learning_rate=float(training_spec["learning_rate"]),
        gradient_clip_norm=float(training_spec["gradient_clip_norm"]),
        weight_reconstruction_weight=float(
            training_spec["weight_reconstruction_weight"]
        ),
        group_size=int(training_spec["group_size"]),
    )
    started = time.monotonic()
    descriptor = mixed_training.train_cq3_tensor(
        source=gauged_source,
        candidate_inputs=gauged_candidate_input,
        teacher_inputs=gauged_teacher_input,
        target=target,
        layer_index=LAYER,
        tensor_root=args.artifact_root,
        repository_commit=repository_commit,
        preregistration_sha256=prereg_sha,
        training_amendment_sha256=prereg_sha,
        training_cache_sha256=full_context_sha,
        expected_source_sha256=tensor_sha256(gauged_source),
        config=training_config,
        row_shard=int(training_spec["row_shard"]),
    )
    gauge_down_path = args.artifact_root / f"{target['gguf_name']}.cq3"
    gauge_down = load_cq3(
        gauge_down_path, rows=HIDDEN, columns=INTERMEDIATE
    )
    with torch.no_grad():
        gauge_mlp_output = gauge_down(gauged_candidate_input).float()
        gauge_candidate_nmse = tensor_nmse(gauge_mlp_output, exact_mlp_output)
    relative_improvement = (baseline_nmse - gauge_candidate_nmse) / max(
        baseline_nmse, 1e-30
    )

    gate_spec = prereg["pilot_gate"]
    checks = {
        "exact_layer_replay": True,
        "all_parent_artifacts_authenticated": True,
        "gauge_finite_and_bounded": bool(torch.isfinite(gauge).all()),
        "unquantized_gauge_identity_nmse_within_maximum": gauge_identity_nmse
        <= float(gate_spec["maximum_unquantized_gauge_identity_nmse"]),
        "up_scale_fold_nmse_within_maximum": up_fold_nmse
        <= float(gate_spec["maximum_up_scale_fold_nmse"]),
        "up_codes_unchanged": up_codes_unchanged,
        "up_payload_storage_neutral": scaled_up_path.stat().st_size == PAYLOAD_BYTES,
        "down_payload_storage_neutral": gauge_down_path.stat().st_size
        == PAYLOAD_BYTES,
        "training_all_finite": descriptor["aggregate_calibration_metrics"][
            "all_finite"
        ],
        "complete_mlp_nmse_not_regressed": gauge_candidate_nmse <= baseline_nmse,
        "minimum_relative_complete_mlp_nmse_improvement_met": relative_improvement
        >= float(gate_spec["minimum_relative_complete_mlp_nmse_improvement"]),
    }
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "authenticated_inputs": {
            "full_context_receipt_sha256": full_context_sha,
            "v44_training_receipt_sha256": v44_sha,
            "layer59_context_shard_sha256": file_sha256(shard_path),
            "source_tensor_sha256": expected_sources,
            "v44_payload_sha256": {
                role: file_sha256(path) for role, path in payloads.items()
            },
        },
        "gauge": {
            "algebra": "up_prime = D up; down_prime = down D^-1",
            "derivation": gauge_spec["derivation"],
            "minimum": float(gauge.min()),
            "maximum": float(gauge.max()),
            "median": float(gauge.median()),
            "mean": float(gauge.mean()),
            "sha256": file_sha256(gauge_path),
            "unquantized_identity_nmse": gauge_identity_nmse,
            "up_scale_fold_nmse": up_fold_nmse,
        },
        "training": {
            "topology": "v44 trained gate plus gauged v44 trained up plus newly trained gauged down",
            "calibration_rows": ROWS,
            "descriptor": str(
                gauge_down_path.with_suffix(".json").relative_to(args.artifact_root)
            ),
            "aggregate_calibration_metrics": descriptor[
                "aggregate_calibration_metrics"
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
        "storage": {
            "baseline_up_down_payload_bytes": 2 * PAYLOAD_BYTES,
            "gauged_up_down_payload_bytes": scaled_up_path.stat().st_size
            + gauge_down_path.stat().st_size,
            "runtime_sidecar_bytes": 0,
            "additional_runtime_operations": 0,
            "gauge_training_artifact_not_required_at_runtime": gauge_path.stat().st_size,
        },
        "metrics": {
            "v44_mixed_topology_complete_mlp_nmse": baseline_nmse,
            "gauged_topology_complete_mlp_nmse": gauge_candidate_nmse,
            "relative_complete_mlp_nmse_improvement": relative_improvement,
            "v44_recorded_down_projection_nmse": trained[
                "blk.59.ffn_down.weight"
            ]["aggregate_calibration_metrics"]["selected_calibration_nmse"],
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "decision": (
                "promote_gauge_to_multilayer_topology_conditioned_training"
                if all(checks.values())
                else "reject_fixed_column_rms_gauge_before_model_integration"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_swiglu_diagonal_gauge_pilot_marker_v71",
        "receipt_sha256": file_sha256(args.out),
        "passed": receipt["gate"]["passed"],
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
