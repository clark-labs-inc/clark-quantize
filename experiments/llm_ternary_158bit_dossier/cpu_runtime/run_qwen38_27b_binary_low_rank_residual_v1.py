#!/usr/bin/env python3
"""Full layer-0 gate/up screen for a shared binary-right residual factor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_binary_low_rank_residual import (
    binary_low_rank_outputs,
    binary_right_values,
    fit_binary_low_rank_residual,
    shared_binary_low_rank_storage_projection,
)
from clark_quantize.qwen38_gain_hash_residual import (
    gain_hash_storage_projection,
    learn_gain_hash_routing,
)
from clark_quantize.qwen38_hash_residual import fixed_support_ternary
from clark_quantize.qwen38_row_scale_codebook import (
    fit_row_normalized_log_codebook,
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


def validate_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen binary low-rank source changed: {label}")


def tensor_sha256(values: torch.Tensor, *, chunk_rows: int = 256) -> str:
    digest = hashlib.sha256()
    for start in range(0, values.shape[0], chunk_rows):
        block = values[start : start + chunk_rows].contiguous()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_tensor(
    model_dir: Path,
    weight_map: dict[str, str],
    specification: dict,
) -> tuple[torch.Tensor, str]:
    name = specification["name"]
    shard = weight_map.get(name)
    if shard != specification["shard"]:
        raise RuntimeError(f"Qwen binary low-rank tensor shard changed: {name}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        values = handle.get_tensor(name)
    if list(values.shape) != specification["shape"]:
        raise RuntimeError(f"Qwen binary low-rank tensor shape changed: {name}")
    return values.float().contiguous(), tensor_sha256(values)


def exported_ternary_residual(
    source: torch.Tensor,
    *,
    group: int,
    support: int,
) -> tuple[torch.Tensor, dict]:
    codes, scales, _ = fixed_support_ternary(
        source, group=group, nonzero_per_group=support
    )
    scale_export = fit_row_normalized_log_codebook(
        scales.numpy(), entries=16, iterations=50, chunk_rows=256
    )
    frozen_scales = torch.from_numpy(scale_export.reconstructed)
    reconstructed = (
        codes.to(torch.float32).reshape(source.shape[0], -1, group)
        * frozen_scales[:, :, None]
    ).reshape_as(source)
    residual = (source - reconstructed).contiguous()
    error = float(residual.square().sum(dtype=torch.float64).item())
    record = {
        "source_squared_sum": float(
            source.square().sum(dtype=torch.float64).item()
        ),
        "direct_ternary_squared_error": error,
        "direct_ternary_relative_rmse": math.sqrt(
            error / max(float(source.square().sum(dtype=torch.float64).item()), 1e-30)
        ),
        "scale_export": {
            "row_anchor_bytes": int(scale_export.anchors.nbytes),
            "ratio_codebook_bytes": int(scale_export.values.nbytes),
            "scale_index_count": int(scale_export.indices.size),
        },
    }
    return residual, record


def write_factor_artifact(
    path: Path,
    fit,
    tensor_names: list[str],
    split_rows: list[int],
) -> None:
    payload = {
        "schema": "qwen38_27b_binary_low_rank_factor_artifact_v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "tensor_names": tensor_names,
        "split_rows": split_rows,
        "right_signs": fit.right.signs.cpu(),
        "right_scales": fit.right.scales.cpu(),
        "left_codes": fit.left_codes.cpu(),
        "left_scales": fit.left_scales.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-binary-low-rank-v1.pt"),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_binary_low_rank_residual_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_sources(prereg)
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

    candidate = prereg["candidate"]
    group = int(candidate["group_size"])
    support = int(candidate["nonzero_per_group"])
    rank = int(candidate["rank"])
    residuals = []
    tensor_records = []
    tensor_names = []
    split_rows = []
    for specification in prereg["scope"]["tensors"]:
        source, source_sha256 = load_tensor(args.model_dir, weight_map, specification)
        residual, record = exported_ternary_residual(
            source, group=group, support=support
        )
        residuals.append(residual)
        tensor_names.append(specification["name"])
        split_rows.append(int(source.shape[0]))
        tensor_records.append(
            {
                "name": specification["name"],
                "shape": list(source.shape),
                "source_BF16_sha256": source_sha256,
                **record,
            }
        )
        del source

    stacked = torch.cat(residuals, dim=0)
    del residuals
    fit = fit_binary_low_rank_residual(
        stacked,
        rank=rank,
        iterations=int(candidate["alternating_iterations"]),
        pca_iterations=int(candidate["randomized_pca_iterations"]),
        seed=int(candidate["seed"]),
        relative_ridge=float(candidate["relative_ridge"]),
    )
    candidate_error = min(fit.objective_history)
    direct_error = fit.initial_squared_error

    gain = learn_gain_hash_routing(
        stacked,
        sketch_dim=rank,
        gain_bits=int(prereg["control"]["gain_bits"]),
        iterations=int(prereg["control"]["routing_iterations"]),
    )
    gain_error = gain.exported_squared_error
    storage = shared_binary_low_rank_storage_projection(
        [(record["shape"][0], record["shape"][1]) for record in tensor_records],
        rank=rank,
        group=group,
        nonzero_per_group=support,
    )
    gain_storage_records = [
        gain_hash_storage_projection(
            input_dim=record["shape"][1],
            output_rows=record["shape"][0],
            sketch_dim=rank,
            gain_bits=int(prereg["control"]["gain_bits"]),
            nonzero_per_group=support,
            group=group,
        )
        for record in tensor_records
    ]
    gain_storage_bpw = (
        sum(record["total_bytes"] for record in gain_storage_records) * 8
        / sum(record["shape"][0] * record["shape"][1] for record in tensor_records)
    )

    generator = torch.Generator().manual_seed(int(candidate["probe_seed"]))
    probes = torch.randn(8, stacked.shape[1], generator=generator)
    observed = binary_low_rank_outputs(
        probes, fit.left_coefficients[:32], fit.right
    )
    expanded = probes @ (
        fit.left_coefficients[:32] @ binary_right_values(fit.right)
    ).T
    identity_error = float((observed - expanded).abs().max().item())
    del stacked

    write_factor_artifact(args.artifact, fit, tensor_names, split_rows)
    artifact_sha256 = sha256_file(args.artifact)
    reduction_vs_gain = 1.0 - candidate_error / max(gain_error, 1e-30)
    reduction_vs_direct = 1.0 - candidate_error / max(direct_error, 1e-30)
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_full_tensors": len(tensor_records),
        "observed_binary_low_rank_squared_error": candidate_error,
        "observed_gain_hash_squared_error": gain_error,
        "observed_direct_ternary_squared_error": direct_error,
        "observed_error_reduction_vs_gain_hash": reduction_vs_gain,
        "observed_error_reduction_vs_direct_ternary": reduction_vs_direct,
        "observed_component_bits_per_weight": storage["bits_per_weight"],
        "observed_kernel_identity_max_abs_error": identity_error,
    }
    gate["passed"] = bool(
        len(tensor_records) == int(gate_spec["required_full_tensors"])
        and reduction_vs_gain
        >= float(gate_spec["minimum_error_reduction_vs_gain_hash"])
        and reduction_vs_direct
        >= float(gate_spec["minimum_error_reduction_vs_direct_ternary"])
        and storage["bits_per_weight"]
        <= float(gate_spec["maximum_component_bits_per_weight"])
        and identity_error
        <= float(gate_spec["maximum_kernel_identity_error"])
    )
    receipt = {
        "schema": "qwen38_27b_binary_low_rank_residual_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True,
        "terminal": True,
        "model": MODEL_ID,
        "revision": REVISION,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "scope": tensor_records,
        "candidate": {
            "rank": rank,
            "best_iteration": fit.best_iteration,
            "objective_history": list(fit.objective_history),
            "storage": storage,
            "factor_artifact": str(args.artifact),
            "factor_artifact_bytes": args.artifact.stat().st_size,
            "factor_artifact_sha256": artifact_sha256,
        },
        "gain_hash_control": {
            "objective_history": list(gain.objective_history),
            "continuous_squared_error": gain.continuous_squared_error,
            "exported_squared_error": gain_error,
            "per_tensor_storage": gain_storage_records,
            "aggregate_bits_per_weight": gain_storage_bpw,
        },
        "gate": gate,
        "claim_boundary": {
            "full_real_layer0_gate_and_up_weights": True,
            "complete_layer_including_down_projection": False,
            "activation_or_capability_preservation_proven": False,
            "end_to_end_CPU_throughput_proven": False,
            "whole_model_1_58_bit_artifact_proven": False,
            "scientific_novelty_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
