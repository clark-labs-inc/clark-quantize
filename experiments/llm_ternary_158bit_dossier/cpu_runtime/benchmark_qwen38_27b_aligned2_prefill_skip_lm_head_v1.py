#!/usr/bin/env python3
"""Benchmark exact Qwen3.8-27B prompt advance without redundant LM heads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
            raise RuntimeError(f"frozen Qwen3.8 prefill input changed: {label}")


def run_reference(model, token_ids, *, threads, executor, cpu_ids, row_chunk):
    state = model.allocate_state(len(token_ids))
    values = indices = None
    started = time.perf_counter_ns()
    for token_id in token_ids:
        values, indices = model.forward_token_topk(
            token_id,
            state,
            1,
            row_chunk=row_chunk,
            gemv_implementation="avx2",
            elementwise_implementation="avx2",
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            validated_state=True,
        )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if values is None or indices is None or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("Qwen3.8 reference prefill timing is invalid")
    return elapsed, values, indices, state


def run_candidate(model, token_ids, *, threads, executor, cpu_ids, row_chunk):
    state = model.allocate_state(len(token_ids))
    started = time.perf_counter_ns()
    for token_id in token_ids[:-1]:
        model.advance_token(
            token_id,
            state,
            gemv_implementation="avx2",
            elementwise_implementation="avx2",
            threads=threads,
            executor=executor,
            cpu_ids=cpu_ids,
            validated_state=True,
        )
    values, indices = model.forward_token_topk(
        token_ids[-1],
        state,
        1,
        row_chunk=row_chunk,
        gemv_implementation="avx2",
        elementwise_implementation="avx2",
        threads=threads,
        executor=executor,
        cpu_ids=cpu_ids,
        validated_state=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("Qwen3.8 candidate prefill timing is invalid")
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
            "qwen38-27b-aligned2-prefill-skip-lm-head-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime
        / "qwen38_27b_aligned2_prefill_skip_lm_head_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_available() or not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
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
        raise RuntimeError("prefill artifacts differ from the full-text parent")

    token_ids = [int(value) for value in prereg["input"]["token_ids"]]
    row_chunk = int(prereg["candidate"]["streaming_row_chunk"])
    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        model = image.text_model()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            run_reference(
                model,
                token_ids,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
                row_chunk=row_chunk,
            )
            run_candidate(
                model,
                token_ids,
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
                row_chunk=row_chunk,
            )
            samples = {"reference": [], "candidate": []}
            outputs = {}
            states = {}
            faults_before = resource.getrusage(resource.RUSAGE_SELF)
            repetitions = int(prereg["measurement"]["paired_repetitions"])
            for repetition in range(repetitions):
                order = ("reference", "candidate")
                if repetition & 1:
                    order = tuple(reversed(order))
                for name in order:
                    function = (
                        run_reference if name == "reference" else run_candidate
                    )
                    elapsed, values, indices, state = function(
                        model,
                        token_ids,
                        threads=threads,
                        executor=executor,
                        cpu_ids=cpu_ids,
                        row_chunk=row_chunk,
                    )
                    samples[name].append(elapsed)
                    outputs[name] = (values, indices)
                    states[name] = state
            faults_after = resource.getrusage(resource.RUSAGE_SELF)
        verified_tensors = len(image._verified_tensors)

    reference_values, reference_indices = outputs["reference"]
    candidate_values, candidate_indices = outputs["candidate"]
    exact_values = bool(np.array_equal(reference_values, candidate_values))
    exact_indices = bool(np.array_equal(reference_indices, candidate_indices))
    exact_state = states_exact(states["reference"], states["candidate"])
    reference_inventory = state_inventory(states["reference"])
    candidate_inventory = state_inventory(states["candidate"])
    reference_median = statistics.median(samples["reference"])
    candidate_median = statistics.median(samples["candidate"])
    speedup = reference_median / candidate_median
    candidate_tokens_per_second = len(token_ids) / candidate_median
    major_faults = int(faults_after.ru_majflt - faults_before.ru_majflt)
    minor_faults = int(faults_after.ru_minflt - faults_before.ru_minflt)
    peak_rss = peak_rss_bytes()
    finite = bool(
        exact_values
        and np.all(np.isfinite(reference_values))
        and np.all(np.isfinite(candidate_values))
        and reference_inventory["finite"]
        and candidate_inventory["finite"]
        and all(
            math.isfinite(value) and value > 0.0
            for values in samples.values()
            for value in values
        )
        and math.isfinite(speedup)
        and math.isfinite(candidate_tokens_per_second)
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_exact_final_top1_values": exact_values,
        "observed_exact_final_top1_indices": exact_indices,
        "observed_exact_final_state": exact_state,
        "observed_speedup": speedup,
        "observed_candidate_tokens_per_second": candidate_tokens_per_second,
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
        "schema": "qwen38_27b_aligned2_prefill_skip_lm_head_receipt_v1",
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
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "physical_cpu_ids": cpu_ids,
        "cpu_features": features,
        "input": prereg["input"],
        "reference": {
            "description": "streaming LM-head top-1 after every prompt token",
            "samples_seconds": samples["reference"],
            "median_seconds": reference_median,
            "tokens_per_second": len(token_ids) / reference_median,
            "final_top1_values": reference_values.tolist(),
            "final_top1_indices": reference_indices.tolist(),
            "state": reference_inventory,
        },
        "candidate": {
            "description": "state-only advance except final prompt token",
            "samples_seconds": samples["candidate"],
            "median_seconds": candidate_median,
            "tokens_per_second": candidate_tokens_per_second,
            "final_top1_values": candidate_values.tolist(),
            "final_top1_indices": candidate_indices.tolist(),
            "state": candidate_inventory,
        },
        "parity": {
            "exact_final_top1_values": exact_values,
            "exact_final_top1_indices": exact_indices,
            "exact_final_state": exact_state,
        },
        "measurement": {
            "speedup": speedup,
            "minor_page_faults": minor_faults,
            "major_page_faults": major_faults,
            "process_peak_rss_bytes": peak_rss,
        },
        "gate": gate,
        "claim_boundary": {
            "exact_prompt_state_and_final_top1_parity_proven_if_passed": (
                gate["passed"]
            ),
            "sequential_batch_one_prefill_speedup_proven_if_passed": gate[
                "passed"
            ],
            "batched_or_parallel_prefill_proven": False,
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
