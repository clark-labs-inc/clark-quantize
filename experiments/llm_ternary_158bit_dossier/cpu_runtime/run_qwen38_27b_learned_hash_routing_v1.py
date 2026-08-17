#!/usr/bin/env python3
"""Disjoint-row screen for learned Clark-Hash routing on Qwen3.8-27B."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    fit_hash_ternary_residual,
    quantize_hash_coefficients_int4,
    sketch_rows,
)
from clark_quantize.qwen38_learned_hash_residual import (
    LearnedHashProjection,
    expand_learned_hash,
    fit_fixed_learned_hash_ternary_residual,
    learn_hash_ternary_residual,
    learned_hash_storage_projection,
    learned_sketch_rows,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen learned-hash source changed: {label}")


def load_span(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    tensor: dict,
    span_name: str,
) -> torch.Tensor:
    name = tensor["name"]
    shard = weight_map.get(name)
    if shard != tensor["shard"]:
        raise RuntimeError(f"Qwen3.8 learned-hash tensor shard changed: {name}")
    start, stop = tensor[span_name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        if list(view.get_shape()) != tensor["shape"]:
            raise RuntimeError(f"Qwen3.8 learned-hash tensor shape changed: {name}")
        return view[start:stop].contiguous().reshape(stop - start, -1).float()


def route_digest(projection: LearnedHashProjection) -> str:
    digest = hashlib.sha256()
    digest.update(projection.buckets.cpu().numpy().astype("<u2", copy=False).tobytes())
    digest.update(projection.signs.cpu().numpy().astype("i1", copy=False).tobytes())
    return digest.hexdigest()


def exported_learned_metrics(
    source: torch.Tensor,
    projection: LearnedHashProjection,
    *,
    group: int,
    support: int,
    iterations: int,
    probe_seed: int,
) -> dict:
    fit = fit_fixed_learned_hash_ternary_residual(
        source,
        projection,
        group=group,
        nonzero_per_group=support,
        iterations=iterations,
    )
    _, _, coefficients = quantize_hash_coefficients_int4(fit.hash_coefficients)
    scales = fit.ternary_scales.to(torch.float16).to(torch.float32)
    ternary = (
        fit.ternary_codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * scales[:, :, None]
    ).reshape_as(source)
    reconstructed = ternary + expand_learned_hash(coefficients, projection)
    squared_error = float(
        (source - reconstructed).square().sum(dtype=torch.float64).item()
    )
    if not math.isclose(
        squared_error, fit.exported_squared_error, rel_tol=2e-6, abs_tol=1e-7
    ):
        raise RuntimeError("learned-hash export error ledger changed")
    generator = torch.Generator().manual_seed(probe_seed)
    probes = torch.randn(8, source.shape[1], generator=generator)
    implicit = probes @ ternary.T + learned_sketch_rows(
        probes, projection
    ) @ coefficients.T
    dense = probes @ reconstructed.T
    identity_error = float((implicit - dense).abs().max().item())
    if identity_error > 5e-4:
        raise RuntimeError(f"learned-hash runtime identity failed: {identity_error}")
    return {
        "squared_error": squared_error,
        "relative_rmse": math.sqrt(
            squared_error / float(source.square().sum(dtype=torch.float64).item())
        ),
        "fp32_objective_history": list(fit.fp32_objective_history),
        "kernel_identity_max_abs_error": identity_error,
    }


def exported_random_control_metrics(
    source: torch.Tensor,
    *,
    sketch_dim: int,
    group: int,
    support: int,
    iterations: int,
    hashes_per_input: int,
    seed: int,
    probe_seed: int,
) -> dict:
    fit = fit_hash_ternary_residual(
        source,
        sketch_dim=sketch_dim,
        hashes_per_input=hashes_per_input,
        seed=seed,
        group=group,
        nonzero_per_group=support,
        iterations=iterations,
    )
    _, _, coefficients = quantize_hash_coefficients_int4(fit.hash_coefficients)
    scales = fit.ternary_scales.to(torch.float16).to(torch.float32)
    ternary = (
        fit.ternary_codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * scales[:, :, None]
    ).reshape_as(source)
    reconstructed = ternary + expand_sketch_rows(coefficients, fit.projection)
    squared_error = float(
        (source - reconstructed).square().sum(dtype=torch.float64).item()
    )
    generator = torch.Generator().manual_seed(probe_seed)
    probes = torch.randn(8, source.shape[1], generator=generator)
    implicit = probes @ ternary.T + sketch_rows(probes, fit.projection) @ coefficients.T
    identity_error = float((implicit - probes @ reconstructed.T).abs().max().item())
    if identity_error > 5e-4:
        raise RuntimeError(f"random-hash control identity failed: {identity_error}")
    return {
        "squared_error": squared_error,
        "relative_rmse": math.sqrt(
            squared_error / float(source.square().sum(dtype=torch.float64).item())
        ),
        "fp32_objective_history": list(fit.fp32_objective_history),
        "kernel_identity_max_abs_error": identity_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-learned-hash-routing-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_learned_hash_routing_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8-27B checkpoint identity changed")
    index_content = (args.model_dir / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_content).hexdigest() != INDEX_SHA256:
        raise RuntimeError("Qwen3.8-27B tensor index changed")
    weight_map = json.loads(index_content)["weight_map"]
    family = prereg["candidate"]
    group = int(family["group_size"])
    support = int(family["nonzero_per_group"])
    ratio = int(family["input_to_sketch_ratio"])

    tensors: dict[str, dict] = {}
    train: dict[str, torch.Tensor] = {}
    evaluation: dict[str, torch.Tensor] = {}
    for tensor in prereg["scope"]["tensors"]:
        train_start, train_stop = tensor["train_row_span"]
        eval_start, eval_stop = tensor["evaluation_row_span"]
        if max(train_start, eval_start) < min(train_stop, eval_stop):
            raise RuntimeError("learned-hash train/evaluation row spans overlap")
        name = tensor["name"]
        tensors[name] = tensor
        train[name] = load_span(
            model_dir=args.model_dir,
            weight_map=weight_map,
            tensor=tensor,
            span_name="train_row_span",
        )
        evaluation[name] = load_span(
            model_dir=args.model_dir,
            weight_map=weight_map,
            tensor=tensor,
            span_name="evaluation_row_span",
        )

    routing_groups: dict[str, LearnedHashProjection] = {}
    routing_training = {}
    for route_spec in prereg["scope"]["shared_routing_groups"]:
        members = route_spec["members"]
        inputs = {train[name].shape[1] for name in members}
        if len(inputs) != 1:
            raise RuntimeError("shared learned-hash route input widths changed")
        input_dim = inputs.pop()
        if input_dim % ratio:
            raise RuntimeError("learned-hash ratio no longer divides Qwen input width")
        source = torch.cat([train[name] for name in members], dim=0)
        learned = learn_hash_ternary_residual(
            source,
            sketch_dim=input_dim // ratio,
            group=group,
            nonzero_per_group=support,
            coordinate_iterations=int(family["coordinate_iterations"]),
            routing_iterations=int(family["routing_iterations"]),
        )
        projection = learned.projection
        routing_groups[route_spec["name"]] = projection
        counts = torch.bincount(
            projection.buckets, minlength=projection.sketch_dim
        )
        routing_training[route_spec["name"]] = {
            "members": members,
            "training_rows": source.shape[0],
            "input_dim": input_dim,
            "sketch_dim": projection.sketch_dim,
            "route_sha256": route_digest(projection),
            "occupied_buckets": int((counts > 0).sum().item()),
            "minimum_bucket_occupancy": int(counts.min().item()),
            "maximum_bucket_occupancy": int(counts.max().item()),
            "training_fp32_objective_history": list(
                learned.fp32_objective_history
            ),
            "training_exported_squared_error": learned.exported_squared_error,
        }

    route_by_tensor = {
        member: route["name"]
        for route in prereg["scope"]["shared_routing_groups"]
        for member in route["members"]
    }
    records = []
    learned_total = 0.0
    random_total = 0.0
    total_parameters = 0
    weighted_rate = 0.0
    maximum_regression = -math.inf
    maximum_identity_error = 0.0
    for index, (name, source) in enumerate(evaluation.items()):
        projection = routing_groups[route_by_tensor[name]]
        learned = exported_learned_metrics(
            source,
            projection,
            group=group,
            support=support,
            iterations=int(family["coordinate_iterations"]),
            probe_seed=int(family["probe_seed"]) + index * 2,
        )
        random = exported_random_control_metrics(
            source,
            sketch_dim=projection.sketch_dim,
            group=group,
            support=support,
            iterations=int(family["coordinate_iterations"]),
            hashes_per_input=int(family["random_control_hashes_per_input"]),
            seed=int(family["random_control_seed"]),
            probe_seed=int(family["probe_seed"]) + index * 2 + 1,
        )
        storage = learned_hash_storage_projection(
            input_dim=source.shape[1],
            output_rows=int(tensors[name]["shape"][0]),
            sketch_dim=projection.sketch_dim,
            nonzero_per_group=support,
            group=group,
        )
        parameters = source.numel()
        regression = learned["squared_error"] / random["squared_error"] - 1.0
        maximum_regression = max(maximum_regression, regression)
        maximum_identity_error = max(
            maximum_identity_error,
            learned["kernel_identity_max_abs_error"],
            random["kernel_identity_max_abs_error"],
        )
        learned_total += learned["squared_error"]
        random_total += random["squared_error"]
        total_parameters += parameters
        weighted_rate += float(storage["bits_per_weight"]) * parameters
        records.append(
            {
                "tensor": name,
                "route_group": route_by_tensor[name],
                "train_row_span": tensors[name]["train_row_span"],
                "evaluation_row_span": tensors[name]["evaluation_row_span"],
                "evaluation_parameters": parameters,
                "learned": learned,
                "random_control": random,
                "learned_error_change_vs_random": regression,
                "complete_component_storage_projection": storage,
            }
        )

    reduction = 1.0 - learned_total / random_total
    rate = weighted_rate / total_parameters
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_evaluation_spans": len(records),
        "observed_aggregate_error_reduction_vs_random": reduction,
        "observed_maximum_per_tensor_error_regression": maximum_regression,
        "observed_component_bits_per_weight": rate,
        "observed_maximum_kernel_identity_error": maximum_identity_error,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_evaluation_spans"])
        and reduction
        >= float(gate_spec["minimum_aggregate_error_reduction_vs_random"])
        and maximum_regression
        <= float(gate_spec["maximum_per_tensor_error_regression"])
        and rate <= float(gate_spec["maximum_component_bits_per_weight"])
        and maximum_identity_error
        <= float(gate_spec["maximum_kernel_identity_error"])
    )
    receipt = {
        "schema": "qwen38_27b_learned_hash_routing_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "routing_training": routing_training,
        "records": records,
        "aggregate": {
            "learned_squared_error": learned_total,
            "random_control_squared_error": random_total,
            "error_reduction_vs_random": reduction,
            "component_bits_per_weight": rate,
        },
        "gate": gate,
        "claim_boundary": {
            "real_Qwen38_27B_weights_measured": True,
            "routing_trained_on_disjoint_rows": True,
            "routing_evaluated_on_frozen_heldout_rows": True,
            "gate_and_up_share_one_route": True,
            "learned_routing_bytes_counted": True,
            "full_layers_or_model_fitted": False,
            "activations_or_capability_evaluated": False,
            "semi_lossless_compression_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
