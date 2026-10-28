#!/usr/bin/env python3
"""Build, execute, and decide the frozen v58 Milan CQ3 LUT benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
PREREGISTRATION = RUNTIME / (
    "qwen38_27b_cq3_exact_lut32_avx2_preregistration_v58.json"
)
BUILD_SCRIPT = REPO / "scripts/cpu/build_qwen38_cq3_exact_lut_benchmark_v58.sh"
VALIDATION_BUILD_SCRIPT = (
    REPO / "scripts/cpu/build_qwen38_cq3_exact_lut_validation_v58.sh"
)
SOURCES = {
    "preregistration": (
        PREREGISTRATION,
        "f767918d279fd673147057d607ee0bf15873dbb29eef6e0db1e3cba5ce1dc399",
    ),
    "native_source": (
        REPO / "clark_quantize/qwen38_cq3_exact_lut_cpu.c",
        "880386b55ce01c8457c258fa0d1c9b36e1503ca1c6541b0166a2206bf4b6d497",
    ),
    "native_header": (
        REPO / "clark_quantize/qwen38_cq3_exact_lut_cpu.h",
        "46f028717bdce7343131c554aed98a9600af55cdbdc05dd7c96a543d19673308",
    ),
    "benchmark_source": (
        RUNTIME / "kernels/benchmark_qwen38_cq3_exact_lut_native_v58.c",
        "adbc175d99e54d45281dcf7052f1053d137c83abf01af35c62895f543c46e97f",
    ),
    "validation_source": (
        RUNTIME / "kernels/validate_qwen38_cq3_exact_lut_native_v58.c",
        "cb55d393d18689c6844c9a6eb9da40d2d548d4e600ed7624d3205ecea5fcc70a",
    ),
    "validation_build_script": (
        VALIDATION_BUILD_SCRIPT,
        "120a4d176127c0380db063d3444c39c81d6533b297322d11d4caa2fdfc38e618",
    ),
}
SHAPES = ("gate_up", "down")
ALLOWED_UNIT = "quant-qwen38-27b-cq3-exact-lut32-v58.service"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def lscpu_fields() -> dict[str, str]:
    payload = json.loads(command("lscpu", "-J").stdout)
    return {
        record["field"].rstrip(":"): record.get("data", "")
        for record in payload["lscpu"]
    }


def assert_idle() -> list[str]:
    output = command(
        "systemctl",
        "list-units",
        "--type=service",
        "--state=running",
        "--no-legend",
    ).stdout
    active = [
        line.split()[0]
        for line in output.splitlines()
        if line.strip() and "quant-qwen38" in line
    ]
    conflicting = [unit for unit in active if unit != ALLOWED_UNIT]
    if conflicting:
        raise RuntimeError(
            "v58 refuses overlapping Qwen3.8 compute: " + ", ".join(conflicting)
        )
    return active


def median_absolute_deviation_fraction(values: list[float], center: float) -> float:
    return statistics.median(abs(value - center) for value in values) / center


def validate_shape(record: dict[str, Any], shape: str) -> dict[str, Any]:
    if (
        record.get("schema") != "qwen38_cq3_exact_lut_native_benchmark_v58"
        or record.get("shape") != shape
        or record.get("warmup_pairs") != 2
        or record.get("measured_pairs") != 21
        or record.get("cache_eviction_bytes") != 268_435_456
        or record.get("x8_lut32_bit_exact") is not True
        or float(record.get("maximum_scaled_delta_vs_ordinary", math.inf))
        > 0.00005
        or record.get("complete") is not True
        or record.get("terminal") is not True
        or record.get("source_bytes") != record.get("x8_bytes")
        or record.get("source_bytes") != record.get("lut32_bytes")
    ):
        raise ValueError(f"v58 {shape} identity, exactness, or storage gate changed")
    x8 = [float(value) for value in record.get("x8_over_lut32_ratios", [])]
    ordinary = [
        float(value) for value in record.get("ordinary_over_lut32_ratios", [])
    ]
    if (
        len(x8) != 21
        or len(ordinary) != 21
        or not all(math.isfinite(value) and value > 0.0 for value in x8 + ordinary)
    ):
        raise ValueError(f"v58 {shape} ratio denominator changed")
    x8_median = statistics.median(x8)
    ordinary_median = statistics.median(ordinary)
    x8_mad = median_absolute_deviation_fraction(x8, x8_median)
    ordinary_mad = median_absolute_deviation_fraction(ordinary, ordinary_median)
    for observed, reported, label in (
        (x8_median, float(record["x8_over_lut32_median"]), "x8 median"),
        (
            ordinary_median,
            float(record["ordinary_over_lut32_median"]),
            "ordinary median",
        ),
        (x8_mad, float(record["x8_ratio_mad_fraction"]), "x8 MAD"),
        (
            ordinary_mad,
            float(record["ordinary_ratio_mad_fraction"]),
            "ordinary MAD",
        ),
    ):
        if not math.isclose(observed, reported, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"v58 {shape} reported {label} changed")
    return {
        "shape": shape,
        "x8_over_lut32_median": x8_median,
        "ordinary_over_lut32_median": ordinary_median,
        "x8_ratio_mad_fraction": x8_mad,
        "ordinary_ratio_mad_fraction": ordinary_mad,
        "x8_shape_minimum_passed": x8_median >= 1.10,
        "stability_passed": x8_mad <= 0.03 and ordinary_mad <= 0.03,
    }


def decide(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shapes = [validate_shape(records[shape], shape) for shape in SHAPES]
    x8_geometric = math.sqrt(
        shapes[0]["x8_over_lut32_median"]
        * shapes[1]["x8_over_lut32_median"]
    )
    ordinary_geometric = math.sqrt(
        shapes[0]["ordinary_over_lut32_median"]
        * shapes[1]["ordinary_over_lut32_median"]
    )
    gates = {
        "both_shapes_present": set(records) == set(SHAPES),
        "integer_and_scaled_output_exact": all(
            records[shape]["x8_lut32_bit_exact"] for shape in SHAPES
        ),
        "persistent_storage_neutral": all(
            records[shape]["source_bytes"] == records[shape]["lut32_bytes"]
            for shape in SHAPES
        ),
        "each_shape_at_least_1_10x_x8": all(
            result["x8_shape_minimum_passed"] for result in shapes
        ),
        "x8_geometric_mean_at_least_1_15x": x8_geometric >= 1.15,
        "ordinary_geometric_mean_at_least_1_45x": ordinary_geometric >= 1.45,
        "all_ratio_mad_fractions_at_most_0_03": all(
            result["stability_passed"] for result in shapes
        ),
    }
    return {
        "shape_summaries": shapes,
        "geometric_mean_x8_over_lut32": x8_geometric,
        "geometric_mean_ordinary_over_lut32": ordinary_geometric,
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_native(record: dict[str, Any]) -> bool:
    return record == {
        "schema": "qwen38_cq3_exact_lut_native_validation_v58",
        "random_groups": 1024,
        "adversarial_groups": 8,
        "reserved_code_checks": 8,
        "direct_scalar_bit_exact": True,
        "scalar_avx2_bit_exact": True,
        "storage_neutral": True,
        "complete": True,
        "terminal": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-cq3-exact-lut32-avx2-v58"),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/workspace/build/qwen38-cq3-exact-lut32-avx2-v58"),
    )
    parser.add_argument("--cpu-set", default="0,2,4,6")
    args = parser.parse_args()
    receipt = args.state_dir / "terminal-decision.json"
    marker = args.state_dir / "EXPERIMENT_COMPLETE.json"
    if receipt.exists() or marker.exists():
        if not receipt.is_file() or not marker.is_file():
            raise RuntimeError("partial v58 terminal state requires inspection")
        existing = json.loads(receipt.read_text())
        if sha256(receipt) != json.loads(marker.read_text()).get("receipt_sha256"):
            raise RuntimeError("v58 terminal marker hash changed")
        print(json.dumps({"event": "v58_terminal_resumed", "passed": existing["passed"]}))
        return 0 if existing["passed"] else 1

    tracked = command(
        "git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"
    ).stdout.strip()
    if tracked:
        raise RuntimeError("v58 requires a clean tracked research worktree")
    source_hashes: dict[str, str] = {}
    for label, (path, expected) in SOURCES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"v58 frozen {label} hash changed")
        source_hashes[f"{label}_sha256"] = observed
    preregistration = json.loads(PREREGISTRATION.read_text())
    if (
        preregistration.get("schema")
        != "qwen38_27b_cq3_exact_lut32_avx2_preregistration_v58"
        or preregistration.get("status") != "frozen_before_native_implementation"
        or preregistration.get("model") != "Qwen/Qwen3.8-27B"
    ):
        raise RuntimeError("v58 preregistration identity changed")

    active_units = assert_idle()
    cpu = lscpu_fields()
    flags = set(cpu.get("Flags", "").split())
    if (
        cpu.get("Architecture") != "x86_64"
        or cpu.get("Vendor ID") != "AuthenticAMD"
        or "Milan" not in cpu.get("Model name", "")
        or not {"avx2", "fma", "f16c"}.issubset(flags)
        or cpu.get("Core(s) per socket") != "4"
    ):
        raise RuntimeError("v58 host is not the frozen four-core AMD Milan target")
    swaps = Path("/proc/swaps").read_text().splitlines()
    if len(swaps) != 1:
        raise RuntimeError("v58 refuses a host with active swap")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    binary = args.build_dir / "qwen38-cq3-exact-lut32-avx2-v58"
    validation_binary = args.build_dir / "validate-qwen38-cq3-exact-lut32-v58"
    command(str(BUILD_SCRIPT), str(binary))
    command(str(VALIDATION_BUILD_SCRIPT), str(validation_binary))
    binary_hash = sha256(binary)
    validation_binary_hash = sha256(validation_binary)
    records: dict[str, dict[str, Any]] = {}
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    validation_process = subprocess.run(
        ["taskset", "-c", args.cpu_set, str(validation_binary)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    validation_lines = [
        line for line in validation_process.stdout.splitlines() if line.strip()
    ]
    if len(validation_lines) != 1:
        raise RuntimeError("v58 native validation emitted a non-terminal transcript")
    validation = json.loads(validation_lines[0])
    if not validate_native(validation):
        raise RuntimeError("v58 frozen native validation gate failed")
    validation_path = args.state_dir / "native-validation.json"
    atomic_json(validation, validation_path)
    for shape in SHAPES:
        process = subprocess.run(
            ["taskset", "-c", args.cpu_set, str(binary), "--shape", shape],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"v58 {shape} emitted a non-terminal transcript")
        record = json.loads(lines[0])
        records[shape] = record
        atomic_json(record, args.state_dir / f"{shape}.json")

    decision = decide(records)
    decision["gates"]["frozen_native_validation_passed"] = True
    decision["passed"] = all(decision["gates"].values())
    repository_commit = command("git", "-C", str(REPO), "rev-parse", "HEAD").stdout.strip()
    final = {
        "schema": "qwen38_cq3_exact_lut_native_benchmark_decision_v58",
        "experiment_id": "qwen38_27b_cq3_exact_lut32_avx2_v58",
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "executed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": {
            "lscpu": cpu,
            "cpu_set": args.cpu_set,
            "threads": 1,
            "active_qwen38_units_at_start": active_units,
            "swap_bytes": 0,
        },
        "provenance": {
            "repository_commit": repository_commit,
            **source_hashes,
            "runner_sha256": sha256(Path(__file__)),
            "build_script_sha256": sha256(BUILD_SCRIPT),
            "binary_sha256": binary_hash,
            "validation_binary_sha256": validation_binary_hash,
            "compiler": command("cc", "--version").stdout.splitlines()[0],
        },
        "raw_receipts": {
            shape: {
                "path": str(args.state_dir / f"{shape}.json"),
                "sha256": sha256(args.state_dir / f"{shape}.json"),
                "result": records[shape],
            }
            for shape in SHAPES
        },
        "native_validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "result": validation,
        },
        **decision,
        "infrastructure_valid": True,
        "decision": "promote_to_v59_integration" if decision["passed"] else "retain_negative_stop_before_v59",
        "claim_boundary": preregistration["claim_boundary"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(final, receipt)
    atomic_json(
        {
            "schema": "qwen38_cq3_exact_lut_v58_completion_marker",
            "receipt_sha256": sha256(receipt),
            "passed": final["passed"],
            "terminal": True,
        },
        marker,
    )
    print(json.dumps({"event": "v58_terminal", "passed": final["passed"], "receipt": str(receipt)}))
    return 0 if final["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
