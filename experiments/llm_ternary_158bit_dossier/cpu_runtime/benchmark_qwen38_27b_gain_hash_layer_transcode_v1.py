#!/usr/bin/env python3
"""Full-Qwen-shape direct-vs-dense gain-hash load-time transcode benchmark."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_fixed_support_codec import FixedSupportPayload
from clark_quantize.qwen38_fixed_support_cpu import (
    load_fixed_support_cpu_library,
    native_unpack_fixed_support_matrix,
)
from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    load_mask64_cpu_library,
    native_pack_mask64_matrix,
)
from clark_quantize.qwen38_gain_hash_layer_transcode import (
    load_gain_hash_layer_transcode_cpu_library,
    native_transcode_fixed_support_mask64,
)


ROWS = 17408
COLUMNS = 5120
GROUP_SIZE = 128
GROUP_WIDTH_BITS = 189
MASK64_GROUP_BYTES = 24
MINIMUM_RSS_SAVING_BYTES = 64 * 1024 * 1024
MAXIMUM_DIRECT_RSS_BYTES = 256 * 1024 * 1024
MAXIMUM_DIRECT_MEDIAN_SECONDS = 5.0
MAXIMUM_SLOWDOWN_RATIO = 1.75


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def canonical_full_shape_payload(rows: int, columns: int) -> tuple[bytes, bytes]:
    groups = rows * (columns // GROUP_SIZE)
    if groups % 8:
        raise ValueError("benchmark group count must make a byte-aligned template")
    sign_bits = (np.arange(64, dtype=np.uint8) & 1).astype(np.uint8)
    group_bits = np.zeros((8, GROUP_WIDTH_BITS), dtype=np.uint8)
    group_bits[:, :64] = sign_bits
    encoded_template = np.packbits(
        group_bits.reshape(-1), bitorder="little"
    ).tobytes()
    if len(encoded_template) != 8 * GROUP_WIDTH_BITS // 8:
        raise AssertionError("gain-hash benchmark input template changed")
    support_low = (1 << 64) - 1
    signs = sum(int(bit) << index for index, bit in enumerate(sign_bits))
    mask_group = (
        support_low.to_bytes(8, "little")
        + bytes(8)
        + signs.to_bytes(8, "little")
    )
    return encoded_template * (groups // 8), mask_group * groups


def worker(args: argparse.Namespace) -> None:
    payload = args.input.read_bytes()
    encoded = FixedSupportPayload(
        payload=payload,
        bit_length=args.rows * (args.columns // GROUP_SIZE) * GROUP_WIDTH_BITS,
        rows=args.rows,
        columns=args.columns,
        group_size=GROUP_SIZE,
        nonzero_per_group=64,
        group_width_bits=GROUP_WIDTH_BITS,
    )
    if args.worker == "direct":
        library = load_gain_hash_layer_transcode_cpu_library(args.library)
        if library is None:
            raise RuntimeError("direct transcode worker library did not load")

        def operation():
            return native_transcode_fixed_support_mask64(
                encoded, library=library
            )

    else:
        fixed = load_fixed_support_cpu_library(args.library)
        mask = load_mask64_cpu_library(args.library)
        if fixed is None or mask is None:
            raise RuntimeError("dense-control transcode worker library did not load")

        def operation():
            codes = native_unpack_fixed_support_matrix(encoded, library=fixed)
            return native_pack_mask64_matrix(codes, library=mask)

    warmup = operation()
    output_bytes = len(warmup.payload)
    output_sha256 = hashlib.sha256(warmup.payload).hexdigest()
    del warmup
    gc.collect()
    durations = []
    hashes = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        observed = operation()
        durations.append(time.perf_counter() - started)
        hashes.append(hashlib.sha256(observed.payload).hexdigest())
        del observed
        gc.collect()
    if len(set(hashes + [output_sha256])) != 1:
        raise RuntimeError("transcode worker output changed across repetitions")
    median = statistics.median(durations)
    record = {
        "schema": "qwen38_27b_gain_hash_layer_transcode_worker_v1",
        "mode": args.worker,
        "rows": args.rows,
        "columns": args.columns,
        "input_bytes": len(payload),
        "output_bytes": output_bytes,
        "repeats": args.repeats,
        "durations_seconds": durations,
        "median_seconds": median,
        "input_MiB_per_second": len(payload) / (1024 * 1024) / median,
        "output_sha256": output_sha256,
        "maximum_rss_bytes": maximum_rss_bytes(),
        "complete": True,
        "terminal": True,
    }
    atomic_bytes(args.out, canonical_json(record))


def parent(args: argparse.Namespace) -> None:
    if platform.machine() not in {"x86_64", "AMD64"} and args.require_x86:
        raise RuntimeError("authoritative transcode benchmark requires x86_64")
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("a C11 compiler is required")
    c_transcode = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode.c"
    c_fixed = REPO / "clark_quantize/qwen38_fixed_support_cpu.c"
    c_mask = REPO / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"
    binding = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode.py"
    benchmark = Path(__file__).resolve()
    verifier_receipt = REPO / (
        "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
        "qwen38-gain-hash-layer-transcode-v1.json"
    )
    expected_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    shared_flags = (
        ["-dynamiclib"]
        if platform.system() == "Darwin"
        else ["-shared", "-fPIC"]
    )
    payload, expected_mask64 = canonical_full_shape_payload(ROWS, COLUMNS)
    expected_output_sha256 = hashlib.sha256(expected_mask64).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        library = directory / f"gain-layer-transcode{extension}"
        compile_command = [
            compiler,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            *shared_flags,
            str(c_transcode),
            str(c_fixed),
            str(c_mask),
            "-o",
            str(library),
        ]
        subprocess.run(compile_command, check=True)
        input_path = directory / "fixed-support.bin"
        input_path.write_bytes(payload)
        worker_records = {}
        for mode in ("direct", "dense"):
            worker_out = directory / f"{mode}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--input",
                    str(input_path),
                    "--library",
                    str(library),
                    "--rows",
                    str(ROWS),
                    "--columns",
                    str(COLUMNS),
                    "--repeats",
                    str(args.repeats),
                    "--out",
                    str(worker_out),
                ],
                check=True,
            )
            worker_records[mode] = json.loads(worker_out.read_text())
    direct = worker_records["direct"]
    dense = worker_records["dense"]
    speedup = dense["median_seconds"] / direct["median_seconds"]
    slowdown = direct["median_seconds"] / dense["median_seconds"]
    rss_saving = dense["maximum_rss_bytes"] - direct["maximum_rss_bytes"]
    passed = bool(
        direct["output_sha256"] == expected_output_sha256
        and dense["output_sha256"] == expected_output_sha256
        and direct["median_seconds"] <= MAXIMUM_DIRECT_MEDIAN_SECONDS
        and slowdown <= MAXIMUM_SLOWDOWN_RATIO
        and rss_saving >= MINIMUM_RSS_SAVING_BYTES
        and direct["maximum_rss_bytes"] <= MAXIMUM_DIRECT_RSS_BYTES
    )
    receipt = {
        "schema": "qwen38_27b_gain_hash_layer_transcode_benchmark_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": expected_commit,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "sources": {
            "native_transcode": {
                "path": str(c_transcode.relative_to(REPO)),
                "sha256": sha256_file(c_transcode),
            },
            "binding": {
                "path": str(binding.relative_to(REPO)),
                "sha256": sha256_file(binding),
            },
            "benchmark": {
                "path": str(benchmark.relative_to(REPO)),
                "sha256": sha256_file(benchmark),
            },
            "fixed_support_control": {
                "path": str(c_fixed.relative_to(REPO)),
                "sha256": sha256_file(c_fixed),
            },
            "mask64_control": {
                "path": str(c_mask.relative_to(REPO)),
                "sha256": sha256_file(c_mask),
            },
            "correctness_receipt": {
                "path": str(verifier_receipt.relative_to(REPO)),
                "sha256": sha256_file(verifier_receipt),
            },
        },
        "shape": {
            "rows": ROWS,
            "columns": COLUMNS,
            "parameters": ROWS * COLUMNS,
            "also_covers_down_projection": [5120, 17408],
            "groups": ROWS * (COLUMNS // GROUP_SIZE),
            "input_bytes": len(payload),
            "output_bytes": len(expected_mask64),
            "dense_INT8_intermediate_bytes": ROWS * COLUMNS,
            "expected_output_sha256": expected_output_sha256,
        },
        "workers": worker_records,
        "comparison": {
            "direct_speedup_vs_dense_control": speedup,
            "direct_slowdown_vs_dense_control": slowdown,
            "maximum_rss_saving_bytes": rss_saving,
        },
        "gate": {
            "minimum_rss_saving_bytes": MINIMUM_RSS_SAVING_BYTES,
            "maximum_direct_rss_bytes": MAXIMUM_DIRECT_RSS_BYTES,
            "maximum_direct_median_seconds": MAXIMUM_DIRECT_MEDIAN_SECONDS,
            "maximum_slowdown_ratio": MAXIMUM_SLOWDOWN_RATIO,
            "require_both_output_hashes_exact": True,
            "passed": passed,
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "full_shape_synthetic_load_time_transcode_only": True,
            "real_weights_or_model_quality_proven": False,
            "token_throughput_proven": False,
            "semi_lossless_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("direct", "dense"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-gain-hash-layer-transcode-v1.json"
        ),
    )
    parser.add_argument("--require-x86", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        if parsed.input is None or parsed.library is None:
            raise SystemExit("worker mode requires --input and --library")
        worker(parsed)
    else:
        parent(parsed)
