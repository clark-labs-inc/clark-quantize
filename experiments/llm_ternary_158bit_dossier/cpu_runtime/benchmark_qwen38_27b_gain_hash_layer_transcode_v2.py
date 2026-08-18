#!/usr/bin/env python3
"""Benchmark the optimized direct enumerative-to-mask64 layer transcode."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json


ROWS = 17408
COLUMNS = 5120
GROUP_SIZE = 128
GROUP_WIDTH_BITS = 189
MINIMUM_SPEEDUP = 2.0
MINIMUM_RSS_SAVING_BYTES = 64 * 1024 * 1024
WORKER = Path(__file__).with_name(
    "benchmark_qwen38_27b_gain_hash_layer_transcode_v1.py"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1_benchmark():
    specification = importlib.util.spec_from_file_location(
        "qwen38_gain_hash_transcode_benchmark_v1", WORKER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the frozen v1 worker")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
            "qwen38-gain-hash-layer-transcode-v2-local.json"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("at least three repetitions are required")
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("a C11 compiler is required")

    benchmark_v1 = load_v1_benchmark()
    payload, expected_mask64 = benchmark_v1.canonical_full_shape_payload(
        ROWS, COLUMNS
    )
    expected_sha256 = hashlib.sha256(expected_mask64).hexdigest()
    native_source = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode_v2.c"
    fixed_source = REPO / "clark_quantize/qwen38_fixed_support_cpu.c"
    mask_source = REPO / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"
    binding = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode.py"
    test_source = REPO / "clark_quantize/test_qwen38_gain_hash_layer_transcode_v2.py"
    benchmark = Path(__file__).resolve()
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    shared_flags = (
        ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
    )

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        library = directory / f"gain-layer-transcode-v2{extension}"
        compile_command = [
            compiler,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            *shared_flags,
            str(native_source),
            str(fixed_source),
            str(mask_source),
            "-o",
            str(library),
        ]
        subprocess.run(compile_command, check=True)
        input_path = directory / "fixed-support.bin"
        input_path.write_bytes(payload)
        workers = {}
        for mode in ("direct", "dense"):
            worker_out = directory / f"{mode}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
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
            workers[mode] = json.loads(worker_out.read_text())

    direct = workers["direct"]
    dense = workers["dense"]
    speedup = dense["median_seconds"] / direct["median_seconds"]
    rss_saving = dense["maximum_rss_bytes"] - direct["maximum_rss_bytes"]
    exact = (
        direct["output_sha256"]
        == dense["output_sha256"]
        == expected_sha256
    )
    passed = bool(
        exact
        and speedup >= MINIMUM_SPEEDUP
        and rss_saving >= MINIMUM_RSS_SAVING_BYTES
    )
    sources = {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for name, path in (
            ("optimized_native_transcode", native_source),
            ("binding", binding),
            ("test", test_source),
            ("benchmark", benchmark),
            ("frozen_worker", WORKER),
            ("fixed_support_control", fixed_source),
            ("mask64_control", mask_source),
        )
    }
    receipt = {
        "schema": "qwen38_27b_gain_hash_layer_transcode_benchmark_v2",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "sources": sources,
        "shape": {
            "rows": ROWS,
            "columns": COLUMNS,
            "parameters": ROWS * COLUMNS,
            "also_covers_down_projection": [5120, 17408],
            "groups": ROWS * (COLUMNS // GROUP_SIZE),
            "input_bytes": len(payload),
            "output_bytes": len(expected_mask64),
            "dense_INT8_intermediate_bytes": ROWS * COLUMNS,
            "expected_output_sha256": expected_sha256,
        },
        "workers": workers,
        "comparison": {
            "direct_speedup_vs_dense_control": speedup,
            "maximum_rss_saving_bytes": rss_saving,
        },
        "gate": {
            "minimum_speedup": MINIMUM_SPEEDUP,
            "minimum_rss_saving_bytes": MINIMUM_RSS_SAVING_BYTES,
            "require_exact_output_hashes": True,
            "passed": passed,
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "synthetic_full_Qwen_shape_load_time_only": True,
            "real_weights_or_model_quality_proven": False,
            "token_throughput_proven": False,
            "semi_lossless_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
