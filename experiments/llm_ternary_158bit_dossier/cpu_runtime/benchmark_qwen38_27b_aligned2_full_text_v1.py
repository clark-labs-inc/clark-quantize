#!/usr/bin/env python3
"""Run the authenticated 64-layer Qwen3.8-27B text graph for one token."""
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
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_aligned2_cache import Qwen38Aligned2RuntimeImage
from clark_quantize.qwen38_cq3_runtime import (
    Qwen38AttentionState,
    Qwen38TextState,
)
from clark_quantize.ternary_cpu import cpu_features, native_available


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def physical_cpu_ids() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("infrastructure-invalid: CPU affinity is unavailable")
    selected: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        if key not in seen:
            seen.add(key)
            selected.append(cpu)
    return selected


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen full-text input changed: {label}")


def state_arrays(state: Qwen38TextState):
    for layer_state in state.layer_states:
        if isinstance(layer_state, tuple):
            yield from layer_state
        elif isinstance(layer_state, Qwen38AttentionState):
            yield layer_state.key
            yield layer_state.value
        else:
            raise TypeError("Qwen3.8 text state type changed")


def state_inventory(state: Qwen38TextState) -> dict:
    digest = hashlib.sha256()
    arrays = 0
    total_bytes = 0
    finite = True
    for value in state_arrays(state):
        arrays += 1
        total_bytes += int(value.nbytes)
        finite = finite and bool(np.all(np.isfinite(value)))
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast("B"))
    attention_lengths = [
        layer_state.length
        for layer_state in state.layer_states
        if isinstance(layer_state, Qwen38AttentionState)
    ]
    return {
        "position": state.position,
        "arrays": arrays,
        "bytes": total_bytes,
        "finite": finite,
        "sha256": digest.hexdigest(),
        "attention_lengths": attention_lengths,
    }


def states_exact(left: Qwen38TextState, right: Qwen38TextState) -> bool:
    left_arrays = list(state_arrays(left))
    right_arrays = list(state_arrays(right))
    return bool(
        left.position == right.position
        and len(left_arrays) == len(right_arrays)
        and all(
            np.array_equal(left_value, right_value)
            for left_value, right_value in zip(left_arrays, right_arrays)
        )
        and [
            value.length
            for value in left.layer_states
            if isinstance(value, Qwen38AttentionState)
        ]
        == [
            value.length
            for value in right.layer_states
            if isinstance(value, Qwen38AttentionState)
        ]
    )


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
        "--cache-parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-full-cache-v1.json"
        ),
    )
    parser.add_argument(
        "--layer-parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-real-layers-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-full-text-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = (
        runtime / "qwen38_27b_aligned2_full_text_preregistration_v1.json"
    )
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    if prereg["scope"]["active_model_only"] != "Qwen/Qwen3.8-27B":
        raise ValueError("full-text gate is not frozen to Qwen3.8-27B")
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

    cache_parent_content = args.cache_parent_receipt.read_bytes()
    cache_parent = json.loads(cache_parent_content)
    layer_parent_content = args.layer_parent_receipt.read_bytes()
    layer_parent = json.loads(layer_parent_content)
    if (
        cache_parent.get("schema")
        != "qwen38_27b_aligned2_full_cache_receipt_v1"
        or cache_parent.get("gate", {}).get("passed") is not True
        or layer_parent.get("schema")
        != "qwen38_27b_aligned2_real_layers_receipt_v1"
        or layer_parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("full-text parent gates did not pass")
    source = cache_parent["source_container"]
    cache = cache_parent["cache"]
    if (
        args.container.resolve() != Path(source["path"]).resolve()
        or args.container.stat().st_size != int(source["bytes"])
        or args.cache.resolve() != Path(cache["path"]).resolve()
        or args.cache.stat().st_size != int(cache["bytes"])
        or args.manifest.resolve() != Path(cache["manifest_path"]).resolve()
        or sha256_file(args.manifest) != cache["manifest_sha256"]
        or layer_parent["artifacts"]["source_container"]["sha256"]
        != source["sha256"]
        or layer_parent["artifacts"]["cache"]["sha256"]
        != cache["sha256"]
    ):
        raise RuntimeError("full-text artifact identities differ from their parents")

    token_id = int(prereg["input"]["token_id"])
    topk_count = int(prereg["candidate"]["streaming_topk_count"])
    row_chunk = int(prereg["candidate"]["streaming_row_chunk"])
    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        model = image.text_model()
        if len(image._verified_tensors) != 666:
            raise RuntimeError("full-text graph did not authenticate 666 tensors")
        reference_state = model.allocate_state(1)
        candidate_state = model.allocate_state(1)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            reference_started = time.perf_counter_ns()
            logits = model.forward_token(
                token_id,
                reference_state,
                gemv_implementation="avx2",
                elementwise_implementation="avx2",
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
                validated_state=True,
            )
            reference_seconds = (
                time.perf_counter_ns() - reference_started
            ) / 1e9
            reference_values, reference_indices = model.topk(
                logits, topk_count
            )
            candidate_faults_before = resource.getrusage(resource.RUSAGE_SELF)
            candidate_started = time.perf_counter_ns()
            candidate_values, candidate_indices = model.forward_token_topk(
                token_id,
                candidate_state,
                topk_count,
                row_chunk=row_chunk,
                gemv_implementation="avx2",
                elementwise_implementation="avx2",
                threads=threads,
                executor=executor,
                cpu_ids=cpu_ids,
                validated_state=True,
            )
            candidate_seconds = (
                time.perf_counter_ns() - candidate_started
            ) / 1e9
            candidate_faults_after = resource.getrusage(resource.RUSAGE_SELF)
        verified_tensors = sorted(image._verified_tensors)

    reference_inventory = state_inventory(reference_state)
    candidate_inventory = state_inventory(candidate_state)
    exact_state = states_exact(reference_state, candidate_state)
    exact_topk_indices = bool(
        np.array_equal(reference_indices, candidate_indices)
    )
    exact_topk_values = bool(np.array_equal(reference_values, candidate_values))
    major_faults = int(
        candidate_faults_after.ru_majflt - candidate_faults_before.ru_majflt
    )
    minor_faults = int(
        candidate_faults_after.ru_minflt - candidate_faults_before.ru_minflt
    )
    peak_rss = peak_rss_bytes()
    finite = bool(
        math.isfinite(reference_seconds)
        and reference_seconds > 0.0
        and math.isfinite(candidate_seconds)
        and candidate_seconds > 0.0
        and np.all(np.isfinite(logits))
        and np.all(np.isfinite(reference_values))
        and np.all(np.isfinite(candidate_values))
        and reference_inventory["finite"]
        and candidate_inventory["finite"]
    )
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_exact_streaming_topk_indices": exact_topk_indices,
        "observed_exact_streaming_topk_values": exact_topk_values,
        "observed_exact_final_state": exact_state,
        "observed_authenticated_aligned_tensor_count": len(verified_tensors),
        "observed_candidate_seconds_per_token": candidate_seconds,
        "observed_candidate_tokens_per_second": 1.0 / candidate_seconds,
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_major_page_faults_during_candidate": major_faults,
        "observed_all_outputs_and_states_finite": finite,
    }
    gate["passed"] = bool(
        exact_topk_indices
        and exact_topk_values
        and exact_state
        and len(verified_tensors)
        == int(gate_spec["required_authenticated_aligned_tensor_count"])
        and candidate_seconds
        <= float(gate_spec["maximum_candidate_seconds_per_token"])
        and peak_rss <= int(gate_spec["maximum_process_peak_rss_bytes"])
        and major_faults == 0
        and finite
    )
    receipt = {
        "schema": "qwen38_27b_aligned2_full_text_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parents": {
            "cache": {
                "path": str(args.cache_parent_receipt),
                "sha256": hashlib.sha256(cache_parent_content).hexdigest(),
            },
            "real_layers": {
                "path": str(args.layer_parent_receipt),
                "sha256": hashlib.sha256(layer_parent_content).hexdigest(),
            },
        },
        "artifacts": {
            "source_container": source,
            "cache": cache,
            "full_source_and_cache_sha256_reused_from_parent": True,
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
        "reference_full_logits": {
            "seconds_per_token": reference_seconds,
            "tokens_per_second": 1.0 / reference_seconds,
            "logit_count": int(logits.size),
            "topk_values": reference_values.tolist(),
            "topk_indices": reference_indices.tolist(),
            "state": reference_inventory,
        },
        "candidate_streaming_topk": {
            "seconds_per_token": candidate_seconds,
            "tokens_per_second": 1.0 / candidate_seconds,
            "materialized_full_logits": False,
            "row_chunk": row_chunk,
            "topk_values": candidate_values.tolist(),
            "topk_indices": candidate_indices.tolist(),
            "state": candidate_inventory,
            "minor_page_faults": minor_faults,
            "major_page_faults": major_faults,
        },
        "parity": {
            "exact_topk_indices": exact_topk_indices,
            "exact_topk_values": exact_topk_values,
            "exact_final_state": exact_state,
        },
        "authenticated_aligned_tensors": verified_tensors,
        "measurement": {
            "process_peak_rss_bytes": peak_rss,
            "candidate_minor_page_faults": minor_faults,
            "candidate_major_page_faults": major_faults,
        },
        "gate": gate,
        "claim_boundary": {
            "actual_qwen38_27b_weights_used": True,
            "all_64_text_layers_executed_if_passed": gate["passed"],
            "one_token_streaming_topk_graph_proven_if_passed": gate["passed"],
            "teacher_or_reference_model_logit_parity_proven": False,
            "multi_token_generation_proven": False,
            "model_capability_proven": False,
            "steady_state_end_to_end_tokens_per_second_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
