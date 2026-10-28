#!/usr/bin/env python3
"""Learn a storage-neutral SwiGLU diagonal gauge from calibration error.

Unlike the rejected weight-only v71 gauge, this experiment optimizes the
shared hidden-channel gauge through the hard CQ3 forward path on a frozen half
of layer-59 calibration rows.  It then freezes D, folds it into the existing up
CQ3 scales, trains ``down D^-1`` on only those rows, and evaluates the untouched
confirmation half.  The runtime representation remains ordinary CQ3 bytes.
"""

from __future__ import annotations

import argparse
import gc
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
import run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71 as v71

from clark_quantize.base3_chunks import atomic_bytes
from clark_quantize.qwen38_layer0_qat import (
    RowShardTrainingConfig,
    pack_cq3_rows,
    unpack_cq3_rows,
)
from clark_quantize.qwen38_mixed_cq3_runtime import tensor_sha256
from clark_quantize.qwen_qat import hard_ternary_ste
from clark_quantize.ternary import ternary_codes_scales


MODEL_ID = v71.MODEL_ID
MODEL_REVISION = v71.MODEL_REVISION
LAYER = v71.LAYER
ROWS = v71.ROWS
HIDDEN = v71.HIDDEN
INTERMEDIATE = v71.INTERMEDIATE
PAYLOAD_BYTES = v71.PAYLOAD_BYTES
SCHEMA = "qwen38_27b_activation_learned_swiglu_gauge_receipt_v72"


def hard_ternary_weight(source: torch.Tensor, *, group_size: int) -> torch.Tensor:
    codes, scales = ternary_codes_scales(
        source.detach().float(), group=group_size, method="optimal"
    )
    return (
        codes.reshape(codes.shape[0], -1, group_size).float()
        * scales.float().unsqueeze(-1)
    ).reshape_as(codes)


def evaluate_gauge_nmse(
    source: torch.Tensor,
    candidate_inputs: torch.Tensor,
    teacher_inputs: torch.Tensor,
    gauge: torch.Tensor,
    *,
    row_shard: int,
    group_size: int,
) -> float:
    error = 0.0
    energy = 0.0
    scaled_inputs = candidate_inputs.float() * gauge[None, :]
    with torch.no_grad():
        for start in range(0, source.shape[0], row_shard):
            stop = min(start + row_shard, source.shape[0])
            shard = source[start:stop].float()
            packed = hard_ternary_weight(
                shard / gauge[None, :], group_size=group_size
            )
            candidate = functional.linear(scaled_inputs, packed)
            target = functional.linear(teacher_inputs.float(), shard)
            difference = candidate - target
            error += float(difference.square().sum())
            energy += float(target.square().sum())
    if energy <= 0:
        raise ValueError("activation-gauge target energy is zero")
    return error / energy


def learn_activation_gauge(
    source: torch.Tensor,
    candidate_inputs: torch.Tensor,
    teacher_inputs: torch.Tensor,
    *,
    row_shard: int,
    group_size: int,
    outer_steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    log_l2_weight: float,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if (
        source.ndim != 2
        or source.shape[1] != candidate_inputs.shape[1]
        or candidate_inputs.shape != teacher_inputs.shape
        or source.shape[0] % row_shard
        or source.shape[1] % group_size
        or not 0 < minimum <= 1 <= maximum
    ):
        raise ValueError("invalid activation-gauge training inputs")
    log_gauge = torch.nn.Parameter(torch.zeros(source.shape[1], dtype=torch.float32))
    optimizer = torch.optim.AdamW([log_gauge], lr=learning_rate, weight_decay=0.0)
    lower = math.log(minimum)
    upper = math.log(maximum)
    history: list[dict[str, Any]] = []
    initial = torch.ones_like(log_gauge)
    best_gauge = initial.clone()
    best_nmse = evaluate_gauge_nmse(
        source,
        candidate_inputs,
        teacher_inputs,
        best_gauge,
        row_shard=row_shard,
        group_size=group_size,
    )
    history.append(
        {
            "step": 0,
            "hard_calibration_nmse": best_nmse,
            "selected_as_best": True,
            "gradient_norm_before_clip": None,
            "minimum_gauge": 1.0,
            "maximum_gauge": 1.0,
            "log_gauge_std": 0.0,
        }
    )

    with torch.no_grad():
        target_energy = 0.0
        for start in range(0, source.shape[0], row_shard):
            target = functional.linear(
                teacher_inputs.float(), source[start : start + row_shard].float()
            )
            target_energy += float(target.square().sum())
    if target_energy <= 0:
        raise ValueError("activation-gauge target energy is zero")

    for step in range(1, outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, source.shape[0], row_shard):
            # Recreate the gauge graph for each shard so each backward can free
            # its hard-weight graph while accumulating into the shared log D.
            gauge = log_gauge.exp()
            shard = source[start : start + row_shard].float()
            hard = hard_ternary_ste(
                shard / gauge[None, :],
                group_size=group_size,
                row_batch=row_shard,
            )
            candidate = functional.linear(
                candidate_inputs.float() * gauge[None, :], hard
            )
            with torch.no_grad():
                target = functional.linear(teacher_inputs.float(), shard)
            data_loss = (candidate - target).square().sum() / target_energy
            regularizer = (
                log_l2_weight
                * log_gauge.square().mean()
                / (source.shape[0] // row_shard)
            )
            (data_loss + regularizer).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [log_gauge], gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise ValueError("activation-gauge gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            # Remove the unidentifiable global scale, then enforce the frozen
            # positive gauge interval.
            log_gauge.sub_(log_gauge.median()).clamp_(min=lower, max=upper)
            gauge = log_gauge.exp().clone()
        hard_nmse = evaluate_gauge_nmse(
            source,
            candidate_inputs,
            teacher_inputs,
            gauge,
            row_shard=row_shard,
            group_size=group_size,
        )
        selected = hard_nmse < best_nmse
        if selected:
            best_nmse = hard_nmse
            best_gauge = gauge.clone()
        history.append(
            {
                "step": step,
                "hard_calibration_nmse": hard_nmse,
                "selected_as_best": selected,
                "gradient_norm_before_clip": float(gradient_norm),
                "minimum_gauge": float(gauge.min()),
                "maximum_gauge": float(gauge.max()),
                "log_gauge_std": float(log_gauge.detach().std()),
            }
        )
    return best_gauge, history


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
    parser.add_argument("--v71-receipt", required=True, type=Path)
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
        raise ValueError("v72 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v72 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
            or marker.get("receipt_sha256") != v71.file_sha256(args.out)
            or marker.get("terminal") is not True
        ):
            raise ValueError("existing v72 terminal receipt changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = v71.file_sha256(args.preregistration)
    full_context_sha = v71.file_sha256(args.full_context_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v71_sha = v71.file_sha256(args.v71_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    prior = json.loads(args.v71_receipt.read_text())
    repository_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    authenticated = prereg["authenticated_inputs"]
    if (
        prereg.get("schema")
        != "qwen38_27b_activation_learned_swiglu_gauge_preregistration_v72"
        or prereg.get("status") != "frozen_before_execution"
        or prereg.get("scope", {}).get("layer") != LAYER
        or authenticated.get("full_context_receipt_sha256") != full_context_sha
        or authenticated.get("v44_training_receipt_sha256") != v44_sha
        or authenticated.get("v71_negative_receipt_sha256") != v71_sha
        or prior.get("gate", {}).get("passed") is not False
        or prior.get("gate", {}).get("decision")
        != "reject_fixed_column_rms_gauge_before_model_integration"
        or context_receipt.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("v72 frozen protocol or authenticated parents changed")

    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("Qwen3.8-27B model index changed")
    layer_record = context_receipt["layers"][LAYER]
    shard_path = args.full_context_root / "layers" / f"{LAYER}.pt"
    context_path = args.full_context_root / "context.pt"
    if (
        v71.file_sha256(shard_path) != layer_record["shard"]["sha256"]
        or v71.file_sha256(context_path) != context_receipt["context"]["sha256"]
    ):
        raise ValueError("v72 full-context artifacts changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state, _ = cache_v1.layer_state(args.model_dir, weight_map, LAYER)
    for name, expected in authenticated["source_tensor_sha256"].items():
        if tensor_sha256(state[name]) != expected:
            raise ValueError(f"v72 source tensor changed: {name}")

    context = torch.load(context_path, map_location="cpu", weights_only=False)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        LAYER,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    calibration_rows = context["calibration_flat_rows"].long()
    hidden = shard["input"][:8].contiguous()
    teacher_layer_output = shard["output"][:8].contiguous()
    attention_mask = context["attention_mask"].long()[:8]
    position_ids = context["text_position_ids"].long()[:8]
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
        raise ValueError("v72 exact layer replay changed")
    del exact_layer, exact_output, teacher_layer_output, hidden, shard
    gc.collect()

    trained = v71.descriptor_by_tensor(v44)
    payload_root = args.v44_artifact_root / "layers" / str(LAYER) / "cq3"
    payloads = {
        role: payload_root / f"blk.{LAYER}.ffn_{role}.weight.cq3"
        for role in ("gate", "up", "down")
    }
    for role, path in payloads.items():
        expected = authenticated["v44_payload_sha256"][role]
        if path.stat().st_size != PAYLOAD_BYTES or v71.file_sha256(path) != expected:
            raise ValueError(f"v72 trained {role} payload changed")

    gate = v71.load_cq3(payloads["gate"], rows=INTERMEDIATE, columns=HIDDEN)
    up = v71.load_cq3(payloads["up"], rows=INTERMEDIATE, columns=HIDDEN)
    down = v71.load_cq3(payloads["down"], rows=HIDDEN, columns=INTERMEDIATE)
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

    split = prereg["row_split"]
    train_index = torch.arange(int(split["train_start"]), ROWS, int(split["stride"]))
    confirmation_index = torch.arange(
        int(split["confirmation_start"]), ROWS, int(split["stride"])
    )
    if train_index.numel() != 32 or confirmation_index.numel() != 32:
        raise ValueError("v72 frozen row split changed")
    train_candidate_input = baseline_down_input.index_select(0, train_index)
    train_teacher_input = teacher_down_input.index_select(0, train_index)

    optimization = prereg["gauge_optimization"]
    started = time.monotonic()
    gauge, history = learn_activation_gauge(
        source_down,
        train_candidate_input,
        train_teacher_input,
        row_shard=int(optimization["row_shard"]),
        group_size=int(optimization["group_size"]),
        outer_steps=int(optimization["outer_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        log_l2_weight=float(optimization["log_l2_weight"]),
        minimum=float(optimization["minimum"]),
        maximum=float(optimization["maximum"]),
    )
    selected_step = min(history, key=lambda item: item["hard_calibration_nmse"])[
        "step"
    ]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    gauge_path = args.artifact_root / "layer59-activation-gauge-f32.pt"
    if not gauge_path.exists():
        temporary = gauge_path.with_suffix(".pt.tmp")
        torch.save(gauge.contiguous(), temporary)
        os.replace(temporary, gauge_path)
    else:
        stored = torch.load(gauge_path, map_location="cpu", weights_only=True)
        if not torch.equal(stored, gauge):
            raise ValueError("existing v72 gauge changed")

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
    del up_codes, up_scales, checked_codes, checked_scales, gauged_up_scales
    gc.collect()

    gauged_up = v71.load_cq3(
        scaled_up_path, rows=INTERMEDIATE, columns=HIDDEN
    )
    with torch.no_grad():
        gauged_up_output = gauged_up(mlp_input).float()
        gauged_candidate_input = (
            functional.silu(gate_output) * gauged_up_output
        ).to(torch.bfloat16)
        ideal_scaled_input = baseline_down_input.float() * gauge[None, :]
        up_fold_nmse = v71.tensor_nmse(
            gauged_candidate_input.float(), ideal_scaled_input
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
        gauge_identity_nmse = v71.tensor_nmse(
            gauged_exact_output, exact_mlp_output
        )
    del gauged_up, gauged_up_output, ideal_scaled_input
    gc.collect()

    target = {
        "state_name": "mlp.down_proj.weight",
        "gguf_name": "blk.59.ffn_down_activation_gauged.weight",
        "shape": (HIDDEN, INTERMEDIATE),
    }
    training = prereg["down_training"]
    training_config = RowShardTrainingConfig(
        steps=int(training["steps_per_row_shard"]),
        learning_rate=float(training["learning_rate"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        weight_reconstruction_weight=float(training["weight_reconstruction_weight"]),
        group_size=int(training["group_size"]),
    )
    descriptor = mixed_training.train_cq3_tensor(
        source=gauged_source,
        candidate_inputs=gauged_candidate_input.index_select(0, train_index),
        teacher_inputs=gauged_teacher_input.index_select(0, train_index),
        target=target,
        layer_index=LAYER,
        tensor_root=args.artifact_root,
        repository_commit=repository_commit,
        preregistration_sha256=prereg_sha,
        training_amendment_sha256=prereg_sha,
        training_cache_sha256=full_context_sha,
        expected_source_sha256=tensor_sha256(gauged_source),
        config=training_config,
        row_shard=int(training["row_shard"]),
    )
    gauge_down_path = args.artifact_root / f"{target['gguf_name']}.cq3"
    gauge_down = v71.load_cq3(
        gauge_down_path, rows=HIDDEN, columns=INTERMEDIATE
    )
    with torch.no_grad():
        gauge_mlp_output = gauge_down(gauged_candidate_input).float()

    baseline_train_nmse = v71.tensor_nmse(
        baseline_mlp_output.index_select(0, train_index),
        exact_mlp_output.index_select(0, train_index),
    )
    candidate_train_nmse = v71.tensor_nmse(
        gauge_mlp_output.index_select(0, train_index),
        exact_mlp_output.index_select(0, train_index),
    )
    baseline_confirmation_nmse = v71.tensor_nmse(
        baseline_mlp_output.index_select(0, confirmation_index),
        exact_mlp_output.index_select(0, confirmation_index),
    )
    candidate_confirmation_nmse = v71.tensor_nmse(
        gauge_mlp_output.index_select(0, confirmation_index),
        exact_mlp_output.index_select(0, confirmation_index),
    )
    train_improvement = (baseline_train_nmse - candidate_train_nmse) / max(
        baseline_train_nmse, 1e-30
    )
    confirmation_improvement = (
        baseline_confirmation_nmse - candidate_confirmation_nmse
    ) / max(baseline_confirmation_nmse, 1e-30)

    gate_spec = prereg["pilot_gate"]
    checks = {
        "exact_layer_replay": True,
        "all_parent_artifacts_authenticated": True,
        "gauge_finite_and_bounded": bool(torch.isfinite(gauge).all()),
        "gauge_selected_without_confirmation_rows": True,
        "unquantized_gauge_identity_nmse_within_maximum": gauge_identity_nmse
        <= float(gate_spec["maximum_unquantized_gauge_identity_nmse"]),
        "up_scale_fold_nmse_within_maximum": up_fold_nmse
        <= float(gate_spec["maximum_up_scale_fold_nmse"]),
        "up_codes_unchanged": up_codes_unchanged and up_scales_exact,
        "up_payload_storage_neutral": scaled_up_path.stat().st_size == PAYLOAD_BYTES,
        "down_payload_storage_neutral": gauge_down_path.stat().st_size
        == PAYLOAD_BYTES,
        "training_all_finite": descriptor["aggregate_calibration_metrics"][
            "all_finite"
        ],
        "minimum_train_improvement_met": train_improvement
        >= float(gate_spec["minimum_relative_train_mlp_nmse_improvement"]),
        "minimum_confirmation_improvement_met": confirmation_improvement
        >= float(gate_spec["minimum_relative_confirmation_mlp_nmse_improvement"]),
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
            "v71_negative_receipt_sha256": v71_sha,
            "layer59_context_shard_sha256": v71.file_sha256(shard_path),
            "source_tensor_sha256": authenticated["source_tensor_sha256"],
            "v44_payload_sha256": {
                role: v71.file_sha256(path) for role, path in payloads.items()
            },
        },
        "row_split": {
            "training_rows": train_index.tolist(),
            "confirmation_rows": confirmation_index.tolist(),
            "training_count": train_index.numel(),
            "confirmation_count": confirmation_index.numel(),
        },
        "gauge_optimization": {
            "method": "hard-CQ3 straight-through activation-error descent",
            "selected_step": selected_step,
            "history": history,
            "minimum": float(gauge.min()),
            "maximum": float(gauge.max()),
            "median": float(gauge.median()),
            "mean": float(gauge.mean()),
            "log_std": float(gauge.log().std()),
            "sha256": v71.file_sha256(gauge_path),
            "elapsed_seconds_including_down_training": time.monotonic() - started,
            "confirmation_rows_observed_during_selection": 0,
        },
        "exactness": {
            "unquantized_gauge_identity_nmse": gauge_identity_nmse,
            "up_scale_fold_nmse": up_fold_nmse,
            "up_codes_unchanged": up_codes_unchanged,
        },
        "down_training": {
            "descriptor": str(
                gauge_down_path.with_suffix(".json").relative_to(args.artifact_root)
            ),
            "aggregate_calibration_metrics": descriptor[
                "aggregate_calibration_metrics"
            ],
        },
        "metrics": {
            "baseline_train_complete_mlp_nmse": baseline_train_nmse,
            "candidate_train_complete_mlp_nmse": candidate_train_nmse,
            "relative_train_complete_mlp_nmse_improvement": train_improvement,
            "baseline_confirmation_complete_mlp_nmse": baseline_confirmation_nmse,
            "candidate_confirmation_complete_mlp_nmse": candidate_confirmation_nmse,
            "relative_confirmation_complete_mlp_nmse_improvement": confirmation_improvement,
            "v44_recorded_down_projection_nmse": trained[
                "blk.59.ffn_down.weight"
            ]["aggregate_calibration_metrics"]["selected_calibration_nmse"],
        },
        "storage": {
            "baseline_up_down_payload_bytes": 2 * PAYLOAD_BYTES,
            "gauged_up_down_payload_bytes": scaled_up_path.stat().st_size
            + gauge_down_path.stat().st_size,
            "runtime_sidecar_bytes": 0,
            "additional_runtime_operations": 0,
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "decision": (
                "promote_activation_gauge_to_multilayer_training"
                if all(checks.values())
                else "reject_activation_learned_diagonal_gauge_before_model_integration"
            ),
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(receipt, args.out)
    marker = {
        "schema": "qwen38_27b_activation_learned_swiglu_gauge_marker_v72",
        "receipt_sha256": v71.file_sha256(args.out),
        "passed": receipt["gate"]["passed"],
        "terminal": True,
    }
    atomic_json(marker, args.marker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["gate"]["passed"] else 2)


if __name__ == "__main__":
    main()
