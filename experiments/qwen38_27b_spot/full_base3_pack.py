#!/usr/bin/env python3
"""Bounded-memory export of every mapped Qwen3.8-27B matrix.

The export stores one independently decodable base-3 frame and one FP16 scale
block per deterministic tensor row batch.  A strict partial manifest makes the
long run resumable on a spot volume without accepting changed source weights.
Retained high-precision tensors are accounted for but intentionally remain in
the immutable source checkpoint during this research gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from clark_quantize.base3_chunks import (
    ChunkedBase3Store,
    atomic_bytes,
    encode_chunk,
    sha256_bytes,
    verify_chunk_file,
)
from clark_quantize.ternary import ternary_codes_scales
from preflight_quantization import category


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def mapped_sources(model_dir: Path) -> tuple[list[dict[str, Any]], int]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    mapped: list[dict[str, Any]] = []
    total = 0
    for shard in sorted(by_shard):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in sorted(by_shard[shard]):
                shape = list(handle.get_slice(name).get_shape())
                label = category(name, shape)
                if label is None:
                    continue
                trits = math.prod(shape)
                mapped.append({"name": name, "shard": shard, "shape": shape, "category": label, "trits": trits})
                total += trits
    return mapped, total


def source_batches(
    model_dir: Path, sources: list[dict[str, Any]], row_batch: int, device: torch.device
) -> Iterator[tuple[int, np.ndarray, np.ndarray, dict[str, Any]]]:
    index = 0
    current_shard = None
    context = None
    handle = None
    try:
        for source in sources:
            if source["shard"] != current_shard:
                if context is not None:
                    context.__exit__(None, None, None)
                context = safe_open(model_dir / source["shard"], framework="pt", device="cpu")
                handle = context.__enter__()
                current_shard = source["shard"]
            view = handle.get_slice(source["name"])
            rows = source["shape"][0]
            for row_start in range(0, rows, row_batch):
                row_stop = min(rows, row_start + row_batch)
                weight = view[row_start:row_stop, :].contiguous().to(device, dtype=torch.float32)
                codes, scales = ternary_codes_scales(weight, group=128, method="optimal")
                cpu_codes = codes.cpu().numpy().reshape(-1)
                cpu_scales = scales.cpu().numpy().astype("<f2", copy=False)
                descriptor = {
                    "tensor": source["name"],
                    "category": source["category"],
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "tensor_shape": source["shape"],
                    "scale_shape": list(cpu_scales.shape),
                }
                yield index, cpu_codes, cpu_scales, descriptor
                index += 1
                del weight, codes, scales
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full.json"))
    parser.add_argument("--progress", type=Path, default=Path("/workspace/quant-state/qwen38-base3-full-progress.json"))
    parser.add_argument("--row-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.row_batch <= 0 or args.workers <= 0:
        raise ValueError("row batch and worker count must be positive")
    max_in_flight = args.max_in_flight or args.workers * 2
    if max_in_flight < args.workers:
        raise ValueError("max-in-flight must be at least workers")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preflight = json.loads(args.preflight.read_text())
    sources, mapped_trits = mapped_sources(args.model_dir)
    ledger_expected = preflight["parameter_ledger"]
    if mapped_trits != ledger_expected["ternary_payload"]:
        raise RuntimeError(
            f"mapped source inventory has {mapped_trits} trits, preflight requires {ledger_expected['ternary_payload']}"
        )
    expected_batches = sum(math.ceil(source["shape"][0] / args.row_batch) for source in sources)
    identity = {
        "schema": "qwen38_full_base3_identity_v1",
        "model_revision": preflight["download_receipt"]["revision"],
        "index_sha256": preflight["download_receipt"]["index_sha256"],
        "quantizer": {"method": "optimal", "group_size": 128, "scale_dtype": "float16"},
        "row_batch": args.row_batch,
        "mapped_sources": sources,
    }
    code_store = ChunkedBase3Store(args.out / "codes.cq3d", identity)
    scales_dir = args.out / "scales"
    scales_dir.mkdir(parents=True, exist_ok=True)
    initial_batches = code_store.next_index
    code_source_hash = hashlib.sha256()
    scale_source_hash = hashlib.sha256()
    scale_bytes = 0
    reused = written = 0
    trits_seen = 0
    started = time.perf_counter()
    pending: dict[int, tuple[Future, Path]] = {}

    def publish_progress(last_descriptor: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        atomic_json(args.progress, {
            "schema": "clark_qwen38_base3_full_progress_v1",
            "model_revision": identity["model_revision"],
            "committed_batches": code_store.next_index,
            "expected_batches": expected_batches,
            "trits_regenerated": trits_seen,
            "expected_trits": mapped_trits,
            "reused_batches": reused,
            "written_batches": written,
            "last_descriptor": last_descriptor,
            "elapsed_seconds": elapsed,
            "device": device_name,
        })

    def commit_oldest() -> None:
        nonlocal written
        index = min(pending)
        future, scale_path = pending.pop(index)
        encoded = future.result()
        if not scale_path.exists():
            raise RuntimeError(f"scale block disappeared before code commit: {scale_path}")
        if code_store.commit_encoded(encoded) != "written":
            raise AssertionError("new full-pack task unexpectedly reused a committed batch")
        written += 1
        if written == 1 or written % 10 == 0:
            publish_progress(encoded["descriptor"])

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        for index, codes, scales, descriptor in source_batches(args.model_dir, sources, args.row_batch, device):
            code_raw = codes.tobytes()
            scale_raw = scales.tobytes(order="C")
            code_source_hash.update(code_raw)
            scale_source_hash.update(scale_raw)
            trits_seen += codes.size
            scale_bytes += len(scale_raw)
            scale_path = scales_dir / f"{index:08d}.f16"
            descriptor = {
                **descriptor,
                "scale_file": scale_path.name,
                "scale_bytes": len(scale_raw),
                "scale_sha256": sha256_bytes(scale_raw),
            }
            if code_store.reuse_if_committed(index, codes, descriptor):
                if not scale_path.exists() or sha256_bytes(scale_path.read_bytes()) != descriptor["scale_sha256"]:
                    raise ValueError(f"committed scale block {index} is missing or changed")
                reused += 1
                continue
            atomic_bytes(scale_path, scale_raw)
            pending[index] = (executor.submit(encode_chunk, index, codes, descriptor), scale_path)
            while len(pending) >= max_in_flight:
                commit_oldest()
        while pending:
            commit_oldest()

    if trits_seen != mapped_trits:
        raise RuntimeError(f"export regenerated {trits_seen} trits, expected {mapped_trits}")
    expected_scale_bytes = ledger_expected["fp16_scale_count"] * 2
    if scale_bytes != expected_scale_bytes:
        raise RuntimeError(f"export generated {scale_bytes} scale bytes, expected {expected_scale_bytes}")
    verification_started = time.perf_counter()
    records = code_store.manifest["chunks"]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as verifier:
        verified = list(verifier.map(
            verify_chunk_file,
            (code_store.chunk_path(record["index"]) for record in records),
            records,
            chunksize=1,
        ))
    independently_verified_trits = sum(item["trits"] for item in verified)
    if independently_verified_trits != mapped_trits:
        raise RuntimeError(
            f"parallel independent decode verified {independently_verified_trits} trits, expected {mapped_trits}"
        )
    verification_seconds = time.perf_counter() - verification_started
    manifest = code_store.finalize(
        expected_chunks=expected_batches,
        expected_trits=mapped_trits,
        source_sha256=code_source_hash.hexdigest(),
        extra={
            "scale_source_sha256": scale_source_hash.hexdigest(),
            "scale_bytes": scale_bytes,
            "scale_count": ledger_expected["fp16_scale_count"],
            "parallel_independently_verified_trits": independently_verified_trits,
        },
        full_decode=False,
    )
    elapsed = time.perf_counter() - started
    code_ledger = manifest["ledger"]
    retained_bytes = ledger_expected["retained_high_precision"] * 2
    complete_payload_bytes = code_ledger["frame_bytes"] + scale_bytes + retained_bytes
    artifact_storage_bytes = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    artifact_metadata_bytes = artifact_storage_bytes - code_ledger["frame_bytes"] - scale_bytes
    complete_transport_bytes = artifact_storage_bytes + retained_bytes
    receipt = {
        "schema": "clark_qwen38_full_base3_pack_v1",
        "hypothesis": (
            "Every mapped Qwen3.8-27B matrix can be optimal-MSE ternarized and exactly packed with bounded "
            "memory and strict spot resume while preserving the preflight bit ledger."
        ),
        "identity": identity,
        "artifact": str(args.out),
        "code_manifest_sha256": sha256_bytes((args.out / "codes.cq3d" / "manifest.json").read_bytes()),
        "code_source_sha256": code_source_hash.hexdigest(),
        "scale_source_sha256": scale_source_hash.hexdigest(),
        "resume": {"initial_batches": initial_batches, "reused_batches": reused, "written_batches": written},
        "code_ledger": {
            **code_ledger,
            "payload_bits_per_trit": code_ledger["payload_bits"] / mapped_trits,
            "frame_bits_per_trit": code_ledger["frame_bytes"] * 8 / mapped_trits,
            "delta_payload_bits_per_trit": code_ledger["payload_bits"] / mapped_trits - math.log2(3),
        },
        "complete_model_ledger": {
            "ternary_code_frame_bytes": code_ledger["frame_bytes"],
            "fp16_scale_bytes": scale_bytes,
            "retained_high_precision_bytes": retained_bytes,
            "complete_payload_bytes": complete_payload_bytes,
            "payload_effective_bits_per_total_parameter": complete_payload_bytes * 8 / ledger_expected["total"],
            "artifact_storage_bytes_excluding_retained_tensors": artifact_storage_bytes,
            "artifact_metadata_bytes": artifact_metadata_bytes,
            "complete_transport_bytes_including_referenced_retained_tensors": complete_transport_bytes,
            "transport_effective_bits_per_total_parameter": complete_transport_bytes * 8 / ledger_expected["total"],
            "preflight_ideal_payload_bytes": ledger_expected["base3_payload_bytes"],
            "payload_overhead_bytes_vs_preflight": complete_payload_bytes - ledger_expected["base3_payload_bytes"],
            "transport_overhead_bytes_vs_preflight": complete_transport_bytes - ledger_expected["base3_payload_bytes"],
            "retained_tensor_storage": "referenced from immutable source checkpoint; bytes included in ledger, not duplicated",
        },
        "performance": {
            "workers": args.workers,
            "max_in_flight": max_in_flight,
            "row_batch": args.row_batch,
            "elapsed_seconds_including_resume_validation_and_full_decode": elapsed,
            "parallel_independent_decode_seconds": verification_seconds,
            "parallel_independently_verified_trits": independently_verified_trits,
            "effective_mtrits_per_second": mapped_trits / elapsed / 1e6,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "device": device_name,
        },
        "protected_metrics": [
            "all mapped parameter count",
            "exact per-batch decode",
            "regenerated code and scale SHA-256",
            "preflight scale and retained-island ledger",
            "bounded row and process queues",
        ],
        "promotion": {
            "passed": (
                mapped_trits == ledger_expected["ternary_payload"]
                and scale_bytes == expected_scale_bytes
                and code_ledger["payload_bits"] / mapped_trits <= math.log2(3) + 0.001
            ),
            "scope": "all mapped weights and FP16 scales; behavioral quality and standalone retained-tensor export are separate gates",
        },
        "stop_condition": "All mapped tensors or first source, resume, integrity, count, rate, or memory-bound failure.",
    }
    atomic_json(args.receipt, receipt)
    args.progress.unlink(missing_ok=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["promotion"]["passed"]:
        raise RuntimeError("full base-3 pack failed preregistered promotion criteria")


if __name__ == "__main__":
    main()
