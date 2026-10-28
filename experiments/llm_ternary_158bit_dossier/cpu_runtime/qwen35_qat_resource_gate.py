#!/usr/bin/env python3
"""Fail-closed memory/disk admission for the Qwen3.5 global ternary-QAT recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from safetensors import safe_open


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes


TERNARIZE_SUFFIXES = (
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
)
EXCLUDE_SUBSTRINGS = ("visual", "vision", "mtp.", "in_proj_a", "in_proj_b", "conv1d")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_inventory(root: Path, *, hash_weights: bool) -> tuple[list[dict], list[dict]]:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards under {root}")
    tensors: list[dict] = []
    files: list[dict] = []
    names: set[str] = set()
    for shard in shards:
        files.append({
            "path": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": sha256_file(shard) if hash_weights else None,
        })
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in names:
                    raise ValueError(f"duplicate tensor in model shards: {name}")
                names.add(name)
                tensor = handle.get_slice(name)
                shape = list(tensor.get_shape())
                dtype = str(tensor.get_dtype())
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype}: {name}")
                numel = math.prod(shape)
                tensors.append({
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                    "bytes": numel * DTYPE_BYTES[dtype],
                })
    return tensors, files


def is_ternary_master(name: str, shape: list[int]) -> bool:
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    module = name[: -len(".weight")]
    return (
        not any(fragment in module for fragment in EXCLUDE_SUBSTRINGS)
        and any(module.endswith(suffix) for suffix in TERNARIZE_SUFFIXES)
    )


def estimate(tensors: list[dict], rules: dict) -> dict:
    total_numel = sum(item["numel"] for item in tensors)
    source_bytes = sum(item["bytes"] for item in tensors)
    ternary_numel = sum(
        item["numel"] for item in tensors
        if is_ternary_master(item["name"], item["shape"])
    )
    norm_numel = sum(
        item["numel"] for item in tensors
        if len(item["shape"]) == 1
        and "norm" in item["name"].lower()
        and not any(fragment in item["name"].lower() for fragment in ("visual", "vision", "mtp."))
    )
    retained_source_bytes = sum(
        item["bytes"] for item in tensors
        if not is_ternary_master(item["name"], item["shape"])
    )
    master = ternary_numel * rules["ternary_master_bytes_per_parameter"]
    cache = ternary_numel * rules["hard_forward_cache_bytes_per_parameter"]
    gradient = ternary_numel * rules["gradient_bytes_per_parameter"]
    momentum = ternary_numel * rules["muon_momentum_bytes_per_parameter"]
    norm_training = norm_numel * 8
    student_lower_bound = (
        master + cache + gradient + momentum + retained_source_bytes + norm_training
    )
    teacher = total_numel * rules["teacher_bytes_per_parameter"]
    full_lower_bound = student_lower_bound + teacher
    checkpoint_bytes = master + retained_source_bytes + momentum + norm_numel * 6
    return {
        "total_parameters": total_numel,
        "source_model_bytes": source_bytes,
        "ternary_master_parameters": ternary_numel,
        "trainable_norm_parameters": norm_numel,
        "retained_source_bytes": retained_source_bytes,
        "components": {
            "fp32_masters": master,
            "bf16_hard_forward_cache": cache,
            "fp32_gradients": gradient,
            "bf16_muon_momentum": momentum,
            "trainable_norm_state": norm_training,
            "bf16_teacher": teacher,
        },
        "teacher_cache_persistent_lower_bound_bytes": student_lower_bound,
        "full_teacher_persistent_lower_bound_bytes": full_lower_bound,
        "one_checkpoint_estimated_tensor_bytes": checkpoint_bytes,
    }


def available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=Path(__file__).with_name("qwen35_4b_qat_resource_preregistration_v1.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()

    prereg_content = args.preregistration.read_bytes()
    prereg = json.loads(prereg_content)
    if prereg.get("schema") != "clark_qwen35_4b_qat_resource_preregistration_v1":
        raise ValueError("QAT resource preregistration schema mismatch")
    if prereg["admission"]["require_complete_model_file_sha256"] and not args.hash_weights:
        raise ValueError("preregistration requires --hash-weights")
    model_dir = args.model_dir.resolve()
    tensors, files = model_inventory(model_dir, hash_weights=args.hash_weights)
    accounting = estimate(tensors, prereg["memory_accounting"])
    available_memory = available_memory_bytes()
    disk = shutil.disk_usage(args.receipt.parent)
    headroom = 1.0 + prereg["memory_accounting"]["minimum_headroom_fraction"]
    full_required = int(accounting["full_teacher_persistent_lower_bound_bytes"] * headroom)
    cached_required = int(accounting["teacher_cache_persistent_lower_bound_bytes"] * headroom)
    checkpoint = prereg["checkpoint_accounting"]
    checkpoint_peak = (
        accounting["one_checkpoint_estimated_tensor_bytes"]
        * (checkpoint["complete_checkpoints"] + checkpoint["maximum_hidden_in_progress_saves"])
        + checkpoint["reserve_bytes"]
    )
    checks = {
        "model_hashes_complete": all(item["sha256"] for item in files),
        "full_teacher_memory": available_memory >= full_required,
        "teacher_cache_memory": available_memory >= cached_required,
        "checkpoint_disk": disk.free >= checkpoint_peak,
    }
    passed = all(checks.values())
    report = {
        "schema": "clark_qwen35_4b_qat_resource_gate_v1",
        "created_at_unix": time.time(),
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(prereg_content).hexdigest(),
            "rules": prereg,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "available_memory_bytes": available_memory,
            "receipt_filesystem_free_bytes": disk.free,
        },
        "model": {
            "directory": str(model_dir),
            "config_sha256": sha256_file(model_dir / "config.json"),
            "index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
            "files": files,
            "tensor_count": len(tensors),
        },
        "accounting": {
            **accounting,
            "full_teacher_required_with_headroom_bytes": full_required,
            "teacher_cache_required_with_headroom_bytes": cached_required,
            "checkpoint_retention_peak_bytes": checkpoint_peak,
            "excluded_from_lower_bound": [
                "activations", "autograd saved tensors", "allocator fragmentation",
                "temporary model-surgery overlap", "kernel workspaces", "Python and framework overhead"
            ],
        },
        "checks": checks,
        "promotion": {
            "passed": passed,
            "decision": "admit_training" if passed else "reject_host_before_model_load",
            "scope": "resource admission only; no training or model-quality claim",
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_bytes(args.receipt, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
