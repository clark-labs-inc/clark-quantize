#!/usr/bin/env python3
"""Attribute one exact Qwen3.8-27B token across aligned2 projections."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_aligned2_cache import Qwen38Aligned2RuntimeImage
from clark_quantize.qwen38_cq3_runtime import Qwen38CQ3AlignedTensor
from clark_quantize.ternary_cpu import cpu_features, native_available
from benchmark_qwen38_27b_aligned2_full_text_v1 import (
    peak_rss_bytes,
    physical_cpu_ids,
    sha256_file,
    state_inventory,
    states_exact,
)
from benchmark_qwen38_27b_persistent_affinity_v1 import (
    persistent_pinned_executor,
)


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen Qwen3.8 profile input changed: {label}")


def projection_role(name: str) -> str:
    roles = (
        (".mlp.gate_proj.weight", "mlp_gate"),
        (".mlp.up_proj.weight", "mlp_up"),
        (".mlp.down_proj.weight", "mlp_down"),
        (".linear_attn.in_proj_qkv.weight", "linear_qkv"),
        (".linear_attn.in_proj_z.weight", "linear_z"),
        (".linear_attn.in_proj_a.weight", "linear_a"),
        (".linear_attn.in_proj_b.weight", "linear_b"),
        (".linear_attn.out_proj.weight", "linear_out"),
        (".self_attn.q_proj.weight", "attention_q"),
        (".self_attn.k_proj.weight", "attention_k"),
        (".self_attn.v_proj.weight", "attention_v"),
        (".self_attn.o_proj.weight", "attention_out"),
    )
    if name.startswith("lm_head.weight"):
        return "lm_head"
    for suffix, role in roles:
        if suffix in name:
            return role
    raise RuntimeError(f"unclassified Qwen3.8 projection {name!r}")


@contextmanager
def profile_aligned_gemv(records: list[dict]):
    """Time every aligned GEMV call without changing its inputs or result."""
    original = Qwen38CQ3AlignedTensor.gemv_q8

    def profiled(self, *args, **kwargs):
        started = time.perf_counter_ns()
        result = original(self, *args, **kwargs)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError("invalid Qwen3.8 projection timing")
        records.append(
            {
                "name": self.name,
                "role": projection_role(self.name),
                "rows": self.rows,
                "columns": self.columns,
                "aligned_weight_bytes": int(self.packed_weights.nbytes),
                "row_scale_bytes": int(
                    self.packed_scale_indices.nbytes
                    + self.row_anchors.nbytes
                    + self.ratio_codebook.nbytes
                ),
                "elapsed_seconds": elapsed,
            }
        )
        return result

    Qwen38CQ3AlignedTensor.gemv_q8 = profiled
    try:
        yield
    finally:
        Qwen38CQ3AlignedTensor.gemv_q8 = original


def run_token(model, token_id: int, preregistration: dict, executor):
    state = model.allocate_state(1)
    started = time.perf_counter_ns()
    values, indices = model.forward_token_topk(
        token_id,
        state,
        int(preregistration["input"]["streaming_topk_count"]),
        row_chunk=int(preregistration["input"]["streaming_row_chunk"]),
        gemv_implementation="avx2",
        elementwise_implementation="avx2",
        threads=int(preregistration["measurement"]["physical_threads"]),
        executor=executor,
        cpu_ids=None,
        validated_state=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("invalid Qwen3.8 full-graph timing")
    return elapsed, values, indices, state


def aggregate_records(records: list[dict]) -> dict:
    roles: dict[str, list[dict]] = {}
    for record in records:
        roles.setdefault(record["role"], []).append(record)
    aggregate = {}
    for role, values in sorted(roles.items()):
        seconds = [float(value["elapsed_seconds"]) for value in values]
        weight_bytes = sum(int(value["aligned_weight_bytes"]) for value in values)
        scale_bytes = sum(int(value["row_scale_bytes"]) for value in values)
        aggregate[role] = {
            "calls": len(values),
            "seconds": sum(seconds),
            "median_call_seconds": statistics.median(seconds),
            "aligned_weight_bytes": weight_bytes,
            "row_scale_bytes": scale_bytes,
            "effective_weight_and_scale_gib_per_second": (
                (weight_bytes + scale_bytes) / (1024**3) / sum(seconds)
            ),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-cq3-row-normalized-container-v1/"
            "Qwen3.8-27B-row-normalized.cq3"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2.manifest.json"
        ),
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-full-text-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-full-graph-profile-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_full_graph_profile_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if (
        not native_available()
        or not native_library
        or not Path(native_library).is_file()
    ):
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")

    threads = int(prereg["measurement"]["physical_threads"])
    cpu_ids = physical_cpu_ids()[:threads]
    if len(cpu_ids) != threads:
        raise SystemExit("infrastructure-invalid: physical CPU inventory changed")

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    source = parent.get("artifacts", {}).get("source_container", {})
    cache = parent.get("artifacts", {}).get("cache", {})
    if (
        parent.get("schema") != "qwen38_27b_aligned2_full_text_receipt_v1"
        or parent.get("gate", {}).get("passed") is not True
        or args.container.resolve() != Path(source["path"]).resolve()
        or args.container.stat().st_size != int(source["bytes"])
        or args.cache.resolve() != Path(cache["path"]).resolve()
        or args.cache.stat().st_size != int(cache["bytes"])
        or args.manifest.resolve() != Path(cache["manifest_path"]).resolve()
        or sha256_file(args.manifest) != cache["manifest_sha256"]
    ):
        raise RuntimeError("Qwen3.8 profile parent or artifacts changed")

    token_id = int(prereg["input"]["token_id"])
    records: list[dict] = []
    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        model = image.text_model()
        with persistent_pinned_executor(cpu_ids) as (executor, worker_affinity):
            control_seconds, control_values, control_indices, control_state = run_token(
                model, token_id, prereg, executor
            )
            faults_before = resource.getrusage(resource.RUSAGE_SELF)
            with profile_aligned_gemv(records):
                profile_seconds, profile_values, profile_indices, profile_state = run_token(
                    model, token_id, prereg, executor
                )
            faults_after = resource.getrusage(resource.RUSAGE_SELF)
        verified_tensors = len(image._verified_tensors)

    exact_values = bool(np.array_equal(control_values, profile_values))
    exact_indices = bool(np.array_equal(control_indices, profile_indices))
    exact_state = states_exact(control_state, profile_state)
    control_inventory = state_inventory(control_state)
    profile_inventory = state_inventory(profile_state)
    aggregate = aggregate_records(records)
    role_counts = {role: value["calls"] for role, value in aggregate.items()}
    attributed_seconds = sum(record["elapsed_seconds"] for record in records)
    unattributed_seconds = profile_seconds - attributed_seconds
    major_faults = int(faults_after.ru_majflt - faults_before.ru_majflt)
    minor_faults = int(faults_after.ru_minflt - faults_before.ru_minflt)
    peak_rss = peak_rss_bytes()
    finite = bool(
        exact_values
        and np.all(np.isfinite(control_values))
        and np.all(np.isfinite(profile_values))
        and control_inventory["finite"]
        and profile_inventory["finite"]
        and all(
            math.isfinite(record["elapsed_seconds"])
            and record["elapsed_seconds"] > 0.0
            for record in records
        )
        and math.isfinite(unattributed_seconds)
    )
    gate_spec = prereg["gate"]
    expected_counts = gate_spec["required_projection_role_call_counts"]
    gate = {
        **gate_spec,
        "observed_exact_topk_values": exact_values,
        "observed_exact_topk_indices": exact_indices,
        "observed_exact_full_state": exact_state,
        "observed_projection_role_call_counts": role_counts,
        "observed_projection_calls": len(records),
        "observed_authenticated_aligned_tensor_count": verified_tensors,
        "observed_profile_seconds": profile_seconds,
        "observed_attributed_projection_seconds": attributed_seconds,
        "observed_unattributed_seconds": unattributed_seconds,
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_major_page_faults": major_faults,
        "observed_all_outputs_states_and_timings_finite": finite,
    }
    gate["passed"] = bool(
        exact_values
        and exact_indices
        and exact_state
        and role_counts == expected_counts
        and len(records) == int(gate_spec["required_projection_calls"])
        and verified_tensors
        == int(gate_spec["required_authenticated_aligned_tensor_count"])
        and unattributed_seconds
        >= -float(gate_spec["maximum_timer_accounting_slack_seconds"])
        and peak_rss <= int(gate_spec["maximum_process_peak_rss_bytes"])
        and major_faults == 0
        and finite
    )

    receipt = {
        "schema": "qwen38_27b_full_graph_profile_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent": {
            "path": str(args.parent_receipt),
            "sha256": hashlib.sha256(parent_content).hexdigest(),
        },
        "artifacts": {
            "source_container": source,
            "cache": cache,
            "all_666_aligned_tensor_sha256_values_reverified": True,
        },
        "native_library": {
            "path": native_library,
            "sha256": sha256_file(Path(native_library)),
        },
        "platform": platform.platform(),
        "processor": platform.processor(),
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "physical_cpu_ids": cpu_ids,
        "worker_affinity": worker_affinity,
        "cpu_features": features,
        "input": prereg["input"],
        "control": {
            "seconds": control_seconds,
            "tokens_per_second": 1.0 / control_seconds,
            "topk_values": control_values.tolist(),
            "topk_indices": control_indices.tolist(),
            "state": control_inventory,
        },
        "profile": {
            "seconds": profile_seconds,
            "tokens_per_second": 1.0 / profile_seconds,
            "topk_values": profile_values.tolist(),
            "topk_indices": profile_indices.tolist(),
            "state": profile_inventory,
            "attributed_projection_seconds": attributed_seconds,
            "unattributed_seconds": unattributed_seconds,
            "projection_records": records,
            "aggregate_by_role": aggregate,
        },
        "measurement": {
            "minor_page_faults": minor_faults,
            "major_page_faults": major_faults,
            "process_peak_rss_bytes": peak_rss,
        },
        "gate": gate,
        "claim_boundary": {
            "exact_profiled_full_graph_parity_proven_if_passed": gate["passed"],
            "projection_time_attribution_diagnostic_if_passed": gate["passed"],
            "steady_state_throughput_proven": False,
            "capability_changed": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
