#!/usr/bin/env python3
"""Screen exact activation-conditioned residual paging across Qwen3.8-27B."""

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
from typing import Any

import torch
from torch.nn import functional


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

from clark_quantize.qwen38_mixed_cq3_runtime import StreamedCq3Linear


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYERS = (1, 23, 45)
HIDDEN = 5120
INTERMEDIATE = 17408
GROUP_SIZE = 128
GROUP_FRACTION = 0.10
ROW_SHARD = 128
SCHEMA = "qwen38_27b_activation_conditioned_residual_paging_result_v82"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def tensor_nmse(candidate: torch.Tensor, target: torch.Tensor) -> float:
    difference = candidate.float() - target.float()
    energy = target.float().square().sum()
    if float(energy) <= 0:
        raise ValueError("target energy is zero")
    return float(difference.square().sum() / energy)


def reconstruct_rows(base: StreamedCq3Linear, start: int, stop: int) -> torch.Tensor:
    return (
        base.codes[start:stop]
        .reshape(stop - start, -1, GROUP_SIZE)
        .float()
        * base.scales[start:stop].float().unsqueeze(-1)
    ).reshape(stop - start, base.in_features)


def fixed_group_count(columns: int) -> int:
    if columns % GROUP_SIZE:
        raise ValueError("input width must be divisible by residual group size")
    return math.ceil(columns // GROUP_SIZE * GROUP_FRACTION)


def activation_conditioned_residual_linear(
    source: torch.Tensor,
    base: StreamedCq3Linear,
    inputs: torch.Tensor,
    *,
    row_shard: int = ROW_SHARD,
) -> dict[str, Any]:
    """Apply exact residual pages selected without reading residual values."""
    if (
        source.ndim != 2
        or source.shape != (base.out_features, base.in_features)
        or inputs.shape[-1] != base.in_features
        or base.in_features % GROUP_SIZE
        or row_shard <= 0
    ):
        raise ValueError("invalid activation-conditioned residual inputs")
    flat = inputs.reshape(-1, base.in_features).float()
    groups = base.in_features // GROUP_SIZE
    selected_groups = fixed_group_count(base.in_features)
    residual_squared_norm = torch.zeros(groups, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, base.out_features, row_shard):
            stop = min(start + row_shard, base.out_features)
            residual = source[start:stop].float() - reconstruct_rows(base, start, stop)
            residual_squared_norm.add_(
                residual.reshape(stop - start, groups, GROUP_SIZE)
                .double()
                .square()
                .sum(dim=(0, 2))
            )
        grouped_inputs = flat.reshape(-1, groups, GROUP_SIZE)
        input_norm = grouped_inputs.double().square().sum(dim=2).sqrt()
        bound_scores = input_norm * residual_squared_norm.sqrt().unsqueeze(0)
        selected = bound_scores.topk(selected_groups, dim=1, largest=True, sorted=True).indices
        mask = torch.zeros_like(bound_scores, dtype=torch.bool)
        mask.scatter_(1, selected, True)
        masked_inputs = grouped_inputs * mask.to(grouped_inputs.dtype).unsqueeze(-1)
        baseline = base(flat).reshape(flat.shape[0], base.out_features).float()
        exact = functional.linear(flat, source.float())
        correction = torch.empty_like(exact)
        for start in range(0, base.out_features, row_shard):
            stop = min(start + row_shard, base.out_features)
            residual = (
                source[start:stop].float() - reconstruct_rows(base, start, stop)
            ).reshape(stop - start, groups, GROUP_SIZE)
            correction[:, start:stop] = torch.einsum(
                "rgd,ngd->nr", residual, masked_inputs
            )
        candidate = baseline + correction
    baseline_nmse = tensor_nmse(baseline, exact)
    candidate_nmse = tensor_nmse(candidate, exact)
    reduction = (baseline_nmse - candidate_nmse) / max(baseline_nmse, 1e-30)
    selected_bound = bound_scores.gather(1, selected).sum(dim=1)
    total_bound = bound_scores.sum(dim=1).clamp_min(1e-30)
    return {
        "baseline": baseline,
        "exact": exact,
        "candidate": candidate,
        "selected": selected.cpu(),
        "metrics": {
            "input_rows": flat.shape[0],
            "input_groups": groups,
            "selected_groups_per_input": selected_groups,
            "selected_group_fraction": selected_groups / groups,
            "baseline_nmse": baseline_nmse,
            "candidate_nmse": candidate_nmse,
            "relative_nmse_reduction": reduction,
            "minimum_selected_bound_fraction": float((selected_bound / total_bound).min()),
            "median_selected_bound_fraction": float((selected_bound / total_bound).median()),
            "maximum_selected_bound_fraction": float((selected_bound / total_bound).max()),
        },
    }


def layer_training_records(v44: dict[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    matches = [record for record in v44["layers"] if int(record["layer"]) == layer]
    if len(matches) != 1 or len(matches[0].get("cq3_training", [])) != 3:
        raise ValueError(f"v44 layer denominator changed for layer {layer}")
    return {
        record["gguf_tensor"].rsplit("_", 1)[-1].split(".", 1)[0]: record
        for record in matches[0]["cq3_training"]
    }


def run_layer(
    layer: int,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    config: Any,
    context: dict[str, torch.Tensor],
    context_receipt: dict[str, Any],
    full_context_root: Path,
    v44: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    state, _ = cache_v1.layer_state(model_dir, weight_map, layer)
    shard_path = full_context_root / "layers" / f"{layer:02d}.pt"
    layer_record = context_receipt["layers"][layer]
    if v71.file_sha256(shard_path) != layer_record["shard"]["sha256"]:
        raise ValueError(f"layer-{layer} context shard changed")
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cache_v42.validate_shard_tensors(
        shard,
        layer,
        preregistration_sha256=context_receipt["preregistration_sha256"],
        producer_commit=context_receipt["repository_commit"],
    )
    selected_rows = torch.cat(
        (context["calibration_flat_rows"].long(), context["validation_flat_rows"].long())
    )
    hidden = shard["input"].contiguous()
    teacher_layer_output = shard["output"].contiguous()
    exact_layer = block_parent.instantiate_layer(config, layer, state)
    position_embeddings, masks = qat.make_context(
        config,
        hidden,
        context["attention_mask"].long(),
        context["text_position_ids"].long(),
    )
    layer_type = config.layer_types[layer]

    def exact_forward() -> torch.Tensor:
        return qat.forward_with_grad(
            exact_layer,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[layer_type],
            position_ids=context["text_position_ids"].long(),
        )

    exact_output, mlp_input, teacher_down_input = mixed_training.capture_exact_teacher_activations(
        exact_layer, exact_forward, selected_rows
    )
    exact_replay = torch.equal(exact_output, teacher_layer_output)
    if not exact_replay:
        raise ValueError(f"layer-{layer} exact replay changed")
    del exact_layer, exact_output, teacher_layer_output, hidden, shard, position_embeddings, masks
    gc.collect()

    training = layer_training_records(v44, layer)
    payload_root = artifact_root / "layers" / str(layer) / "cq3"
    payloads = {
        role: payload_root / f"blk.{layer}.ffn_{role}.weight.cq3"
        for role in ("gate", "up", "down")
    }
    dimensions = {
        "gate": (INTERMEDIATE, HIDDEN),
        "up": (INTERMEDIATE, HIDDEN),
        "down": (HIDDEN, INTERMEDIATE),
    }
    bases: dict[str, StreamedCq3Linear] = {}
    for role, path in payloads.items():
        descriptor = training[role]["payload"]
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["bytes"]
            or v71.file_sha256(path) != descriptor["sha256"]
        ):
            raise ValueError(f"layer-{layer} trained {role} payload changed")
        rows, columns = dimensions[role]
        bases[role] = v71.load_cq3(path, rows=rows, columns=columns)

    source_gate = state["mlp.gate_proj.weight"]
    source_up = state["mlp.up_proj.weight"]
    source_down = state["mlp.down_proj.weight"]
    gate = activation_conditioned_residual_linear(source_gate, bases["gate"], mlp_input)
    up = activation_conditioned_residual_linear(source_up, bases["up"], mlp_input)
    with torch.no_grad():
        baseline_down_input = (
            functional.silu(gate["baseline"]) * up["baseline"]
        ).to(torch.bfloat16)
        candidate_down_input = (
            functional.silu(gate["candidate"]) * up["candidate"]
        ).to(torch.bfloat16)
        exact_mlp_output = functional.linear(
            teacher_down_input.float(), source_down.float()
        )
        baseline_mlp_output = bases["down"](baseline_down_input).float()
    down = activation_conditioned_residual_linear(
        source_down, bases["down"], candidate_down_input
    )
    candidate_mlp_output = down["candidate"]
    baseline_mlp_nmse = tensor_nmse(baseline_mlp_output, exact_mlp_output)
    candidate_mlp_nmse = tensor_nmse(candidate_mlp_output, exact_mlp_output)
    complete_reduction = (
        baseline_mlp_nmse - candidate_mlp_nmse
    ) / max(baseline_mlp_nmse, 1e-30)
    all_finite = all(
        torch.isfinite(value).all()
        for value in (
            gate["candidate"], up["candidate"], down["candidate"],
            baseline_mlp_output, candidate_mlp_output, exact_mlp_output,
        )
    )
    result = {
        "layer": layer,
        "layer_type": layer_type,
        "rows": int(selected_rows.numel()),
        "exact_layer_replay": exact_replay,
        "operators": {
            "gate": gate["metrics"],
            "up": up["metrics"],
            "down_on_candidate_input": down["metrics"],
        },
        "complete_mlp": {
            "baseline_nmse": baseline_mlp_nmse,
            "candidate_nmse": candidate_mlp_nmse,
            "relative_nmse_reduction": complete_reduction,
        },
        "all_values_finite": bool(all_finite),
        "payload_sha256": {role: training[role]["payload"]["sha256"] for role in training},
    }
    del gate, up, down, bases, state, mlp_input, teacher_down_input
    del baseline_down_input, candidate_down_input, exact_mlp_output
    del baseline_mlp_output, candidate_mlp_output
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--full-context-receipt", required=True, type=Path)
    parser.add_argument("--v44-receipt", required=True, type=Path)
    parser.add_argument("--v81-receipt", required=True, type=Path)
    parser.add_argument("--full-context-root", type=Path, default=Path("/workspace/quant-runs/qwen38-full-context-block-cache-v42"))
    parser.add_argument("--v44-artifact-root", type=Path, default=Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("v82 is frozen to four physical CPU cores")
    torch.set_num_threads(args.threads)
    if args.out.exists() or args.marker.exists():
        if not args.out.is_file() or not args.marker.is_file():
            raise RuntimeError("partial v82 terminal state requires inspection")
        receipt = json.loads(args.out.read_text())
        marker = json.loads(args.marker.read_text())
        if marker.get("receipt_sha256") != v71.file_sha256(args.out):
            raise ValueError("existing v82 marker changed")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise SystemExit(0 if receipt["gate"]["passed"] else 2)

    prereg_sha = v71.file_sha256(args.preregistration)
    context_sha = v71.file_sha256(args.full_context_receipt)
    v44_sha = v71.file_sha256(args.v44_receipt)
    v81_sha = v71.file_sha256(args.v81_receipt)
    prereg = json.loads(args.preregistration.read_text())
    context_receipt = json.loads(args.full_context_receipt.read_text())
    v44 = json.loads(args.v44_receipt.read_text())
    v81 = json.loads(args.v81_receipt.read_text())
    if (
        prereg.get("status") != "frozen_before_execution"
        or prereg["authenticated_inputs"]["full_context_v42_receipt_sha256"] != context_sha
        or prereg["authenticated_inputs"]["trained_cq3_v44_receipt_sha256"] != v44_sha
        or prereg["authenticated_inputs"]["v81_quality_rejection_sha256"] != v81_sha
        or context_receipt.get("gate", {}).get("passed") is not True
        or v44.get("gate", {}).get("passed") is not True
        or v81.get("decision") != "reject_composition_before_kernel_integration"
    ):
        raise ValueError("v82 frozen protocol or authenticated parent changed")
    index_path = args.model_dir / "model.safetensors.index.json"
    if v71.file_sha256(index_path) != direct_v43.INDEX_SHA256:
        raise ValueError("Qwen3.8-27B model index changed")
    context_path = args.full_context_root / "context.pt"
    if v71.file_sha256(context_path) != context_receipt["context"]["sha256"]:
        raise ValueError("v42 context changed")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    ).text_config
    context = torch.load(context_path, map_location="cpu", weights_only=False)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    layers = [
        run_layer(
            layer,
            model_dir=args.model_dir,
            weight_map=weight_map,
            config=config,
            context=context,
            context_receipt=context_receipt,
            full_context_root=args.full_context_root,
            v44=v44,
            artifact_root=args.v44_artifact_root,
        )
        for layer in LAYERS
    ]
    operator_reductions = [
        record["operators"][role]["relative_nmse_reduction"]
        for record in layers
        for role in ("gate", "up", "down_on_candidate_input")
    ]
    complete_reductions = [record["complete_mlp"]["relative_nmse_reduction"] for record in layers]
    median_complete = float(torch.tensor(complete_reductions).median())
    gate_spec = prereg["gate"]
    checks = {
        "all_three_layers_present": [record["layer"] for record in layers] == list(LAYERS),
        "all_nine_operators_present": len(operator_reductions) == 9,
        "all_exact_layer_replays": all(record["exact_layer_replay"] for record in layers),
        "all_values_finite": all(record["all_values_finite"] for record in layers),
        "minimum_operator_recovery_met": min(operator_reductions) >= float(gate_spec["minimum_operator_nmse_reduction_each"]),
        "minimum_complete_mlp_recovery_each_met": min(complete_reductions) >= float(gate_spec["minimum_complete_mlp_nmse_reduction_each_layer"]),
        "median_complete_mlp_recovery_met": median_complete >= float(gate_spec["minimum_median_complete_mlp_nmse_reduction"]),
        "residual_traffic_budget_met": all(
            metric["selected_group_fraction"] <= float(gate_spec["maximum_residual_group_fraction_each"])
            for record in layers
            for metric in record["operators"].values()
        ),
    }
    passed = all(checks.values())
    receipt = {
        "schema": SCHEMA,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": prereg_sha,
        "authenticated_inputs": {
            "full_context_v42_receipt_sha256": context_sha,
            "trained_cq3_v44_receipt_sha256": v44_sha,
            "v81_quality_rejection_sha256": v81_sha,
        },
        "upper_bound": {
            "exact_bf16_residual_values": True,
            "selector_reads_residual_weights": False,
            "fixed_group_fraction": GROUP_FRACTION,
            "deployable_quantized_residual_exists": False,
        },
        "layers": layers,
        "aggregate": {
            "minimum_operator_relative_nmse_reduction": min(operator_reductions),
            "complete_mlp_relative_nmse_reductions": complete_reductions,
            "minimum_complete_mlp_relative_nmse_reduction": min(complete_reductions),
            "median_complete_mlp_relative_nmse_reduction": median_complete,
        },
        "checks": checks,
        "gate": {
            "passed": passed,
            "decision": "admit_to_fresh_prompt_exact_residual_replication" if passed else "reject_input_group_residual_paging_at_ten_percent_traffic",
        },
        "claim_boundary": prereg["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(args.out, receipt)
    atomic_json(
        args.marker,
        {
            "schema": "qwen38_27b_activation_conditioned_residual_paging_marker_v82",
            "receipt_sha256": v71.file_sha256(args.out),
            "passed": passed,
            "terminal": True,
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
