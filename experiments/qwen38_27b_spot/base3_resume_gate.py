#!/usr/bin/env python3
"""Preregistered parallel base-3 interruption/resume gate on real Qwen codes."""
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
import tempfile
import time
from typing import Iterator

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import ChunkedBase3Store, atomic_bytes, encode_chunk
from clark_quantize.ternary import ternary_codes_scales


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def selected_sources(preflight: dict, model_dir: Path) -> list[dict]:
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    seen = set()
    result = []
    for probe in preflight["probes"]:
        name = probe["name"]
        if name in seen:
            continue
        seen.add(name)
        result.append({"name": name, "shard": weight_map[name], "shape": probe["shape"]})
    return result


def real_code_chunks(
    model_dir: Path,
    sources: list[dict],
    max_trits: int,
    chunk_trits: int,
    row_batch: int,
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Yield fixed-size chunks from a deterministic Qwen tensor/row stream."""
    pieces: list[np.ndarray] = []
    buffered = 0
    emitted = 0
    chunk_index = 0
    for source in sources:
        if emitted + buffered >= max_trits:
            break
        name = source["name"]
        with safe_open(model_dir / source["shard"], framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            rows = view.get_shape()[0]
            for row_start in range(0, rows, row_batch):
                if emitted + buffered >= max_trits:
                    break
                row_stop = min(rows, row_start + row_batch)
                weight = view[row_start:row_stop, :].contiguous().to("cuda", dtype=torch.float32)
                codes, _ = ternary_codes_scales(weight, group=128, method="optimal")
                cpu_codes = codes.cpu().numpy().reshape(-1)
                del weight, codes
                position = 0
                while position < cpu_codes.size and emitted + buffered < max_trits:
                    take = min(cpu_codes.size - position, chunk_trits - buffered, max_trits - emitted - buffered)
                    pieces.append(cpu_codes[position : position + take].copy())
                    buffered += take
                    position += take
                    if buffered == chunk_trits or emitted + buffered == max_trits:
                        chunk = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                        descriptor = {"global_start": emitted, "global_stop": emitted + buffered}
                        yield chunk_index, chunk, descriptor
                        emitted += buffered
                        chunk_index += 1
                        pieces = []
                        buffered = 0
    if pieces:
        chunk = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        descriptor = {"global_start": emitted, "global_stop": emitted + buffered}
        yield chunk_index, chunk, descriptor
        emitted += buffered
    if emitted != max_trits:
        raise RuntimeError(f"selected real tensors yielded {emitted} of {max_trits} requested trits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--store", type=Path, default=Path("/workspace/quant-runs/qwen38-base3-resume/artifacts/codes.cq3d"))
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-base3-resume.json"))
    parser.add_argument("--max-trits", type=int, default=32_000_000)
    parser.add_argument("--chunk-trits", type=int, default=1_000_000)
    parser.add_argument("--row-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--commit-delay", type=float, default=0.0, help="test-only delay enabling deterministic forced termination")
    args = parser.parse_args()
    if args.max_trits <= 0 or args.chunk_trits <= 0 or args.workers <= 0:
        raise ValueError("trit counts and worker count must be positive")
    max_in_flight = args.max_in_flight or args.workers * 2
    if max_in_flight < args.workers:
        raise ValueError("max-in-flight must be at least workers")

    preflight = json.loads(args.preflight.read_text())
    sources = selected_sources(preflight, args.model_dir)
    identity = {
        "schema": "qwen38_base3_resume_gate_identity_v1",
        "model_revision": preflight["download_receipt"]["revision"],
        "quantizer": {"method": "optimal", "group_size": 128, "scale_dtype": "float16"},
        "max_trits": args.max_trits,
        "chunk_trits": args.chunk_trits,
        "source_order": sources,
    }
    store = ChunkedBase3Store(args.store, identity)
    initial_chunks = store.next_index
    started = time.perf_counter()
    source_hash = hashlib.sha256()
    reused = written = 0
    pending: dict[int, Future] = {}

    def commit_oldest() -> None:
        nonlocal written
        index = min(pending)
        encoded = pending.pop(index).result()
        status = store.commit_encoded(encoded)
        if status != "written":
            raise AssertionError("new parallel task unexpectedly reused an existing chunk")
        written += 1
        if args.commit_delay:
            time.sleep(args.commit_delay)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        for index, codes, descriptor in real_code_chunks(
            args.model_dir, sources, args.max_trits, args.chunk_trits, args.row_batch
        ):
            source_hash.update(codes.tobytes())
            if store.reuse_if_committed(index, codes, descriptor):
                reused += 1
                continue
            pending[index] = executor.submit(encode_chunk, index, codes, descriptor)
            while len(pending) >= max_in_flight:
                commit_oldest()
        while pending:
            commit_oldest()

    expected_chunks = math.ceil(args.max_trits / args.chunk_trits)
    manifest = store.finalize(
        expected_chunks=expected_chunks,
        expected_trits=args.max_trits,
        source_sha256=source_hash.hexdigest(),
        extra={"model_revision": identity["model_revision"], "quantizer": identity["quantizer"]},
    )
    elapsed = time.perf_counter() - started
    ledger = manifest["ledger"]
    ideal = math.log2(3)
    receipt = {
        "schema": "clark_qwen38_base3_resume_gate_v1",
        "hypothesis": (
            "A bounded parallel chunk container can resume after process termination with exact source identity, "
            "independent chunk decoding, and aggregate payload within 0.001 bits/trit of log2(3)."
        ),
        "identity": identity,
        "store": str(args.store),
        "manifest_sha256": hashlib.sha256((args.store / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": source_hash.hexdigest(),
        "initial_committed_chunks": initial_chunks,
        "reused_chunks": reused,
        "written_chunks": written,
        "ledger": {
            **ledger,
            "payload_bits_per_trit": ledger["payload_bits"] / ledger["trits"],
            "frame_bits_per_trit": ledger["frame_bytes"] * 8 / ledger["trits"],
            "delta_payload_bits_per_trit": ledger["payload_bits"] / ledger["trits"] - ideal,
        },
        "parallelism": {
            "workers": args.workers,
            "max_in_flight": max_in_flight,
            "elapsed_seconds_including_source_regeneration_and_full_decode": elapsed,
            "effective_mtrits_per_second": args.max_trits / elapsed / 1e6,
        },
        "protected_metrics": [
            "regenerated raw SHA-256 equality",
            "per-chunk frame and raw SHA-256",
            "per-frame CRC32",
            "atomic manifest prefix",
            "independent full decode",
        ],
        "promotion": {
            "exact_resume_observed": reused > 0 and initial_chunks > 0,
            "max_delta_bits_per_trit": 0.001,
            "passed": reused > 0 and initial_chunks > 0 and ledger["payload_bits"] / ledger["trits"] <= ideal + 0.001,
            "scope": "chunk container and interruption recovery; full checkpoint packing remains a separate gate",
        },
        "stop_condition": "32 million real Qwen trits or first identity, decode, integrity, or rate failure.",
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["promotion"]["passed"]:
        raise RuntimeError("resume gate did not satisfy its preregistered promotion criteria")


if __name__ == "__main__":
    main()
