#!/usr/bin/env python3
"""Benchmark the exact-shape Qwen3.8-27B recurrent DeltaNet AVX2 kernel."""
from __future__ import annotations

import argparse
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
from clark_quantize.ternary_cpu import (
    cpu_features,
    native_available,
    recurrent_gated_delta_f32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    # Linux reports KiB; macOS reports bytes. The authoritative run is Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_one_layer(
    implementation: str,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    state = initial_state.copy()
    started = time.perf_counter_ns()
    output = recurrent_gated_delta_f32(
        query,
        key,
        value,
        log_decay,
        beta,
        state,
        implementation=implementation,
        validated=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("non-finite one-layer timing")
    return elapsed, output, state


def run_full_stack(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
) -> tuple[float, np.ndarray]:
    output = np.empty_like(value)
    started = time.perf_counter_ns()
    for layer in range(state.shape[0]):
        output[layer] = recurrent_gated_delta_f32(
            query[layer],
            key[layer],
            value[layer],
            log_decay[layer],
            beta[layer],
            state[layer],
            implementation="avx2",
            validated=True,
        )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("non-finite full-stack timing")
    return elapsed, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-recurrent-delta-avx2-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_recurrent_delta_avx2_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    scope = prereg["scope"]
    if scope["active_model_only"] != "Qwen/Qwen3.8-27B":
        raise ValueError("benchmark scope is not frozen to Qwen3.8-27B")
    if not native_available():
        raise SystemExit("infrastructure-invalid: native ternary CPU library is absent")
    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: native library path is not pinned")
    features = cpu_features()
    if not features["avx2"]:
        raise SystemExit("infrastructure-invalid: AVX2 is required")
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != 1:
        raise SystemExit("infrastructure-invalid: benchmark must be pinned to one CPU")

    layers = int(scope["deltanet_layers"])
    key_heads = int(scope["key_heads"])
    value_heads = int(scope["value_heads"])
    dimension = int(scope["head_dimension"])
    if value_heads != key_heads * int(scope["query_key_repeat_factor"]):
        raise ValueError("frozen q/k repeat factor does not compose")

    generator = np.random.default_rng(int(prereg["candidate"]["seed"]))
    query = generator.normal(size=(layers, key_heads, dimension)).astype(np.float32)
    key = generator.normal(size=(layers, key_heads, dimension)).astype(np.float32)
    query /= np.linalg.norm(query, axis=2, keepdims=True)
    key /= np.linalg.norm(key, axis=2, keepdims=True)
    value = generator.normal(size=(layers, value_heads, dimension)).astype(np.float32)
    log_decay = -generator.uniform(
        0.001, 2.0, size=(layers, value_heads)
    ).astype(np.float32)
    beta = generator.uniform(0.01, 0.99, size=(layers, value_heads)).astype(
        np.float32
    )
    initial_state = generator.normal(
        scale=0.05, size=(value_heads, dimension, dimension)
    ).astype(np.float32)

    measurement = prereg["measurement"]
    for _ in range(int(measurement["one_layer_warmups_per_implementation"])):
        run_one_layer(
            "scalar", query[0], key[0], value[0], log_decay[0], beta[0],
            initial_state,
        )
        run_one_layer(
            "avx2", query[0], key[0], value[0], log_decay[0], beta[0],
            initial_state,
        )

    scalar_samples: list[float] = []
    avx2_samples: list[float] = []
    scalar_output = scalar_state = avx2_output = avx2_state = None
    repetitions = int(measurement["one_layer_paired_repetitions"])
    for repetition in range(repetitions):
        order = ("scalar", "avx2")
        if repetition & 1:
            order = tuple(reversed(order))
        for implementation in order:
            elapsed, output, state = run_one_layer(
                implementation,
                query[0],
                key[0],
                value[0],
                log_decay[0],
                beta[0],
                initial_state,
            )
            if implementation == "scalar":
                scalar_samples.append(elapsed)
                scalar_output, scalar_state = output, state
            else:
                avx2_samples.append(elapsed)
                avx2_output, avx2_state = output, state
    assert all(item is not None for item in (
        scalar_output, scalar_state, avx2_output, avx2_state
    ))
    exact_output = bool(np.array_equal(scalar_output, avx2_output))
    exact_state = bool(np.array_equal(scalar_state, avx2_state))

    full_state = np.zeros(
        (layers, value_heads, dimension, dimension), dtype=np.float32
    )
    for _ in range(int(measurement["full_stack_warmups"])):
        run_full_stack(query, key, value, log_decay, beta, full_state)
    faults_before = resource.getrusage(resource.RUSAGE_SELF)
    full_stack_samples: list[float] = []
    full_output = None
    for _ in range(int(measurement["full_stack_repetitions"])):
        elapsed, full_output = run_full_stack(
            query, key, value, log_decay, beta, full_state
        )
        full_stack_samples.append(elapsed)
    faults_after = resource.getrusage(resource.RUSAGE_SELF)
    assert full_output is not None

    scalar_median = statistics.median(scalar_samples)
    avx2_median = statistics.median(avx2_samples)
    full_stack_median = statistics.median(full_stack_samples)
    speedup = scalar_median / avx2_median
    recurrence_only_tokens_per_second = 1.0 / full_stack_median
    peak_rss = peak_rss_bytes()
    minor_faults = int(faults_after.ru_minflt - faults_before.ru_minflt)
    major_faults = int(faults_after.ru_majflt - faults_before.ru_majflt)
    finite = bool(
        np.all(np.isfinite(scalar_output))
        and np.all(np.isfinite(scalar_state))
        and np.all(np.isfinite(avx2_output))
        and np.all(np.isfinite(avx2_state))
        and np.all(np.isfinite(full_output))
        and np.all(np.isfinite(full_state))
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_exact_scalar_avx2_output": exact_output,
        "observed_exact_scalar_avx2_final_state": exact_state,
        "observed_one_layer_avx2_speedup_over_scalar": speedup,
        "observed_full_stack_recurrence_only_tokens_per_second": (
            recurrence_only_tokens_per_second
        ),
        "observed_full_stack_median_seconds_per_token": full_stack_median,
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_all_outputs_and_states_finite": finite,
        "observed_major_page_faults_during_timed_full_stack": major_faults,
    }
    gate["passed"] = bool(
        exact_output
        and exact_state
        and speedup
        >= gate_spec["minimum_one_layer_avx2_speedup_over_scalar"]
        and recurrence_only_tokens_per_second
        >= gate_spec["minimum_full_stack_recurrence_only_tokens_per_second"]
        and full_stack_median
        <= gate_spec["maximum_full_stack_median_seconds_per_token"]
        and peak_rss <= gate_spec["maximum_process_peak_rss_bytes"]
        and finite
        and major_faults == 0
    )
    receipt = {
        "schema": "qwen38_27b_recurrent_delta_avx2_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "native_library": native_library,
        "native_library_sha256": sha256_file(Path(native_library)),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_features": features,
        "shape": {
            "layers": layers,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "head_dimension": dimension,
            "state_bytes_per_layer": int(initial_state.nbytes),
            "state_bytes_all_layers": int(full_state.nbytes),
        },
        "one_layer_paired": {
            "scalar_samples_seconds": scalar_samples,
            "avx2_samples_seconds": avx2_samples,
            "scalar_median_seconds": scalar_median,
            "avx2_median_seconds": avx2_median,
            "avx2_speedup_over_scalar": speedup,
            "exact_output": exact_output,
            "exact_final_state": exact_state,
        },
        "full_stack": {
            "avx2_samples_seconds_per_token": full_stack_samples,
            "median_seconds_per_token": full_stack_median,
            "recurrence_only_tokens_per_second": recurrence_only_tokens_per_second,
            "minor_page_faults_during_timed_region": minor_faults,
            "major_page_faults_during_timed_region": major_faults,
            "peak_rss_bytes": peak_rss,
            "finite": finite,
            "logical_state_stream_bytes_per_token_lower_bound": int(
                full_state.nbytes * 2
            ),
            "logical_kernel_state_access_bytes_per_token": int(
                full_state.nbytes * 4
            ),
        },
        "gate": gate,
        "claim_boundary": {
            "isolated_recurrent_update_proven_if_gate_passes": gate["passed"],
            "full_layer_parity_proven": false,
            "end_to_end_tokens_per_second_proven": false,
            "model_quality_proven": false,
            "scientific_breakthrough_proven": false,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
