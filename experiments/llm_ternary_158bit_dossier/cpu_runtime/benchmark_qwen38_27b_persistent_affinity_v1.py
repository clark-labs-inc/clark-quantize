#!/usr/bin/env python3
"""Benchmark persistent worker affinity on the exact Qwen3.8-27B text graph."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from queue import SimpleQueue
import resource
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_aligned2_cache import Qwen38Aligned2RuntimeImage
from clark_quantize.ternary_cpu import cpu_features, native_available
from benchmark_qwen38_27b_aligned2_full_text_v1 import (
    peak_rss_bytes,
    physical_cpu_ids,
    sha256_file,
    state_inventory,
    states_exact,
)


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"frozen Qwen3.8 persistent-affinity input changed: {label}"
            )


@contextmanager
def persistent_pinned_executor(cpu_ids: list[int]):
    """Create every worker eagerly and pin it once for the pool lifetime."""
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Linux thread affinity is required")
    assignments: SimpleQueue[int] = SimpleQueue()
    for cpu_id in cpu_ids:
        assignments.put(cpu_id)

    def initialize() -> None:
        os.sched_setaffinity(0, {assignments.get_nowait()})

    executor = ThreadPoolExecutor(
        max_workers=len(cpu_ids),
        initializer=initialize,
        thread_name_prefix="qwen38-pinned",
    )
    barrier = threading.Barrier(len(cpu_ids) + 1)

    def prove_worker() -> dict:
        affinity = sorted(os.sched_getaffinity(0))
        record = {
            "native_thread_id": threading.get_native_id(),
            "affinity": affinity,
        }
        barrier.wait()
        return record

    try:
        futures = [executor.submit(prove_worker) for _ in cpu_ids]
        barrier.wait()
        records = [future.result() for future in futures]
        observed = sorted(record["affinity"][0] for record in records)
        if (
            any(len(record["affinity"]) != 1 for record in records)
            or observed != sorted(cpu_ids)
            or len({record["native_thread_id"] for record in records})
            != len(cpu_ids)
        ):
            raise RuntimeError(
                f"persistent worker affinity proof failed: {records!r}"
            )
        yield executor, records
    finally:
        executor.shutdown()


def run_token(
    model,
    token_id: int,
    *,
    threads: int,
    executor: ThreadPoolExecutor,
    cpu_ids: list[int] | None,
    topk_count: int,
    row_chunk: int,
):
    state = model.allocate_state(1)
    started = time.perf_counter_ns()
    values, indices = model.forward_token_topk(
        token_id,
        state,
        topk_count,
        row_chunk=row_chunk,
        gemv_implementation="avx2",
        elementwise_implementation="avx2",
        threads=threads,
        executor=executor,
        cpu_ids=cpu_ids,
        validated_state=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if (
        not math.isfinite(elapsed)
        or elapsed <= 0.0
        or not np.all(np.isfinite(values))
    ):
        raise RuntimeError("Qwen3.8 persistent-affinity timing is invalid")
    return elapsed, values, indices, state


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
            "/workspace/quant-state/"
            "qwen38-27b-persistent-affinity-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_persistent_affinity_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if (
        not native_available()
        or not native_library
        or not Path(native_library).is_file()
    ):
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise SystemExit("infrastructure-invalid: Linux thread affinity is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")

    threads = int(prereg["measurement"]["physical_threads"])
    cpu_ids = physical_cpu_ids()
    if len(cpu_ids) < threads:
        raise SystemExit(
            f"infrastructure-invalid: only {len(cpu_ids)} physical CPUs are available"
        )
    cpu_ids = cpu_ids[:threads]
    token_id = int(prereg["input"]["token_id"])
    topk_count = int(prereg["input"]["streaming_topk_count"])
    row_chunk = int(prereg["input"]["streaming_row_chunk"])

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema") != "qwen38_27b_aligned2_full_text_receipt_v1"
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("full-text parent gate did not pass")
    source = parent["artifacts"]["source_container"]
    cache = parent["artifacts"]["cache"]
    if (
        args.container.resolve() != Path(source["path"]).resolve()
        or args.container.stat().st_size != int(source["bytes"])
        or args.cache.resolve() != Path(cache["path"]).resolve()
        or args.cache.stat().st_size != int(cache["bytes"])
        or args.manifest.resolve() != Path(cache["manifest_path"]).resolve()
        or sha256_file(args.manifest) != cache["manifest_sha256"]
    ):
        raise RuntimeError("affinity artifacts differ from the full-text parent")

    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        model = image.text_model()
        with ThreadPoolExecutor(max_workers=threads) as control_executor:
            with persistent_pinned_executor(cpu_ids) as (
                candidate_executor,
                worker_affinity,
            ):
                common = {
                    "model": model,
                    "token_id": token_id,
                    "threads": threads,
                    "topk_count": topk_count,
                    "row_chunk": row_chunk,
                }
                run_token(
                    **common,
                    executor=control_executor,
                    cpu_ids=cpu_ids,
                )
                run_token(
                    **common,
                    executor=candidate_executor,
                    cpu_ids=None,
                )
                samples = {"control": [], "candidate": []}
                outputs = {}
                states = {}
                faults_before = resource.getrusage(resource.RUSAGE_SELF)
                repetitions = int(prereg["measurement"]["paired_repetitions"])
                for repetition in range(repetitions):
                    order = ("control", "candidate")
                    if repetition & 1:
                        order = tuple(reversed(order))
                    for arm in order:
                        executor = (
                            control_executor
                            if arm == "control"
                            else candidate_executor
                        )
                        affinity = cpu_ids if arm == "control" else None
                        elapsed, values, indices, state = run_token(
                            **common,
                            executor=executor,
                            cpu_ids=affinity,
                        )
                        samples[arm].append(elapsed)
                        outputs[arm] = (values, indices)
                        states[arm] = state
                faults_after = resource.getrusage(resource.RUSAGE_SELF)
        verified_tensors = len(image._verified_tensors)

    control_values, control_indices = outputs["control"]
    candidate_values, candidate_indices = outputs["candidate"]
    exact_values = bool(np.array_equal(control_values, candidate_values))
    exact_indices = bool(np.array_equal(control_indices, candidate_indices))
    exact_state = states_exact(states["control"], states["candidate"])
    control_inventory = state_inventory(states["control"])
    candidate_inventory = state_inventory(states["candidate"])
    control_median = statistics.median(samples["control"])
    candidate_median = statistics.median(samples["candidate"])
    speedup = control_median / candidate_median
    major_faults = int(faults_after.ru_majflt - faults_before.ru_majflt)
    minor_faults = int(faults_after.ru_minflt - faults_before.ru_minflt)
    peak_rss = peak_rss_bytes()
    finite = bool(
        exact_values
        and np.all(np.isfinite(control_values))
        and np.all(np.isfinite(candidate_values))
        and control_inventory["finite"]
        and candidate_inventory["finite"]
        and all(
            math.isfinite(value) and value > 0.0
            for values in samples.values()
            for value in values
        )
        and math.isfinite(speedup)
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_exact_topk_values": exact_values,
        "observed_exact_topk_indices": exact_indices,
        "observed_exact_full_state": exact_state,
        "observed_speedup": speedup,
        "observed_candidate_tokens_per_second": 1.0 / candidate_median,
        "observed_authenticated_aligned_tensor_count": verified_tensors,
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_major_page_faults": major_faults,
        "observed_all_outputs_states_and_timings_finite": finite,
    }
    gate["passed"] = bool(
        exact_values
        and exact_indices
        and exact_state
        and speedup >= float(gate_spec["minimum_speedup"])
        and verified_tensors
        == int(gate_spec["required_authenticated_aligned_tensor_count"])
        and peak_rss <= int(gate_spec["maximum_process_peak_rss_bytes"])
        and major_faults == 0
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_persistent_affinity_receipt_v1",
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
        "candidate_worker_affinity": worker_affinity,
        "cpu_features": features,
        "input": prereg["input"],
        "control": {
            "description": "set and restore worker affinity for every GEMV task",
            "samples_seconds": samples["control"],
            "median_seconds": control_median,
            "tokens_per_second": 1.0 / control_median,
            "topk_values": control_values.tolist(),
            "topk_indices": control_indices.tolist(),
            "state": control_inventory,
        },
        "candidate": {
            "description": "pin each persistent worker once and omit per-GEMV affinity syscalls",
            "samples_seconds": samples["candidate"],
            "median_seconds": candidate_median,
            "tokens_per_second": 1.0 / candidate_median,
            "topk_values": candidate_values.tolist(),
            "topk_indices": candidate_indices.tolist(),
            "state": candidate_inventory,
        },
        "parity": {
            "exact_topk_values": exact_values,
            "exact_topk_indices": exact_indices,
            "exact_full_state": exact_state,
        },
        "measurement": {
            "speedup": speedup,
            "minor_page_faults": minor_faults,
            "major_page_faults": major_faults,
            "process_peak_rss_bytes": peak_rss,
        },
        "gate": gate,
        "claim_boundary": {
            "exact_one_token_full_graph_parity_proven_if_passed": gate["passed"],
            "steady_batch_one_decode_speedup_proven_if_passed": gate["passed"],
            "multi_token_or_batched_speedup_proven": False,
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
