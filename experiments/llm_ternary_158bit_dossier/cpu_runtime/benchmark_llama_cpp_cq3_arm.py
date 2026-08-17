#!/usr/bin/env python3
"""Run the frozen paired llama.cpp CQ3 ARM graph-kernel experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time


HERE = Path(__file__).resolve().parent
PERF_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+us/run")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    check: bool = True,
) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    result = {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": sha256_bytes(completed.stderr.encode()),
    }
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return result


def git_output(source: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(source), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return completed.stdout


def validate_source(label: str, source: Path, frozen: dict) -> dict:
    if not source.is_dir():
        raise ValueError(f"missing {label} source directory: {source}")
    revision = git_output(source, "rev-parse", "HEAD").decode().strip()
    diff = git_output(source, "diff", "--no-ext-diff", "--full-index")
    diff_check = subprocess.run(
        ["git", "-C", str(source), "diff", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if revision != frozen["revision"]:
        raise ValueError(f"{label} revision differs from preregistration")
    if sha256_bytes(diff) != frozen["full_diff_sha256"]:
        raise ValueError(f"{label} source diff differs from preregistration")
    if diff_check.returncode != 0 or diff_check.stdout or diff_check.stderr:
        raise ValueError(f"{label} source fails git diff --check")
    actual_files = {
        relative: sha256_file(source / relative)
        for relative in frozen["file_sha256"]
    }
    if actual_files != frozen["file_sha256"]:
        raise ValueError(f"{label} frozen file hashes differ")
    return {
        "path": str(source),
        "revision": revision,
        "full_diff_sha256": sha256_bytes(diff),
        "file_sha256": actual_files,
    }


def build_env(build: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "LC_ALL": "C",
        "LANG": "C",
        "NO_COLOR": "1",
        "DYLD_LIBRARY_PATH": str(build / "bin"),
    })
    return env


def benchmark_once(build: Path, test_filter: str) -> dict:
    command = [
        str(build / "bin/test-backend-ops"),
        "perf",
        "-b", "CPU",
        "-o", "MUL_MAT",
        "-p", test_filter,
        "--output", "console",
    ]
    result = run(command, env=build_env(build), timeout=180)
    matches = PERF_RE.findall(result["stdout"] + result["stderr"])
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one performance result, found {len(matches)}\n"
            f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
        )
    result["microseconds_per_run"] = float(matches[0])
    return result


def sysctl(name: str) -> str | None:
    completed = subprocess.run(
        ["sysctl", "-n", name], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--control-build", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=HERE / "llama_cpp_cq3_arm_preregistration_v1.json",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if not args.receipt.is_absolute():
        parser.error("--receipt must be absolute")

    prereg_bytes = args.preregistration.read_bytes()
    prereg = json.loads(prereg_bytes)
    performance = prereg.get("performance", {})
    if (
        prereg.get("schema") != "clark_llama_cpp_cq3_arm_preregistration_v1"
        or prereg.get("status") != "frozen_before_authoritative_paired_run"
        or args.warmups != performance.get("warmups")
        or args.repetitions != performance.get("repetitions")
    ):
        raise ValueError("benchmark arguments differ from frozen preregistration")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the v1 authoritative run is scoped to Darwin arm64")
    if os.cpu_count() != performance["ggml_cpu_threads"]:
        raise RuntimeError("logical CPU count differs from frozen GGML thread count")

    base_patch = HERE / prereg["patches"]["base"]["path"]
    arm_patch = HERE / prereg["patches"]["arm_delta"]["path"]
    if (
        sha256_file(base_patch) != prereg["patches"]["base"]["sha256"]
        or sha256_file(arm_patch) != prereg["patches"]["arm_delta"]["sha256"]
    ):
        raise ValueError("versioned llama.cpp patch identity changed")
    if sha256_file(Path(__file__)) != prereg["benchmark_sha256"]:
        raise ValueError("benchmark source changed after preregistration")

    sources = {
        "candidate": validate_source("candidate", args.candidate_source, prereg["sources"]["candidate"]),
        "control": validate_source("control", args.control_source, prereg["sources"]["control"]),
    }
    for label, build in (("candidate", args.candidate_build), ("control", args.control_build)):
        if not (build / "CMakeCache.txt").is_file():
            raise ValueError(f"missing {label} CMake build: {build}")
        if sha256_file(build / "CMakeCache.txt") != prereg["builds"][label]["cmake_cache_sha256"]:
            raise ValueError(f"{label} CMake configuration changed")

    build_receipts = {
        "candidate": run([
            "cmake", "--build", str(args.candidate_build),
            "--target", "test-cq3", "test-backend-ops", "-j", "8",
        ], timeout=300),
        "control": run([
            "cmake", "--build", str(args.control_build),
            "--target", "test-backend-ops", "-j", "8",
        ], timeout=300),
    }

    candidate_cq3 = run(
        [str(args.candidate_build / "bin/test-cq3")],
        env=build_env(args.candidate_build), timeout=120,
    )
    backend_correctness = run([
        str(args.candidate_build / "bin/test-backend-ops"),
        "test", "-b", "CPU", "-o", "MUL_MAT",
        "-p", prereg["correctness"]["cq3_filter"],
        "--output", "console",
    ], env=build_env(args.candidate_build), timeout=180)
    correctness_text = backend_correctness["stdout"] + backend_correctness["stderr"]
    cq3_cases = correctness_text.count("MUL_MAT(type_a=cq3_0")
    if cq3_cases != prereg["correctness"]["expected_backend_cases"]:
        raise RuntimeError(f"expected {prereg['correctness']['expected_backend_cases']} CQ3 cases, found {cq3_cases}")
    if "FAIL" in correctness_text:
        raise RuntimeError("CQ3 backend correctness output contains FAIL")

    test_filter = performance["test_filter"]
    warmups: list[dict] = []
    for _ in range(args.warmups):
        warmups.append({
            "control": benchmark_once(args.control_build, test_filter),
            "candidate": benchmark_once(args.candidate_build, test_filter),
        })

    samples: list[dict] = []
    by_variant: dict[str, list[float]] = {"control": [], "candidate": []}
    for repetition in range(args.repetitions):
        order = ["control", "candidate"]
        if repetition % 2:
            order.reverse()
        pair: dict = {"repetition": repetition, "order": order, "results": {}}
        for label in order:
            build = args.control_build if label == "control" else args.candidate_build
            measurement = benchmark_once(build, test_filter)
            pair["results"][label] = measurement
            by_variant[label].append(measurement["microseconds_per_run"])
        pair["speedup"] = (
            pair["results"]["control"]["microseconds_per_run"]
            / pair["results"]["candidate"]["microseconds_per_run"]
        )
        samples.append(pair)

    medians = {
        label: statistics.median(values)
        for label, values in by_variant.items()
    }
    median_speedup = medians["control"] / medians["candidate"]
    paired_speedups = [pair["speedup"] for pair in samples]
    rules = prereg["promotion"]
    passed = (
        candidate_cq3["returncode"] == 0
        and backend_correctness["returncode"] == 0
        and cq3_cases == prereg["correctness"]["expected_backend_cases"]
        and median_speedup >= rules["minimum_median_speedup"]
        and medians["candidate"] <= rules["maximum_candidate_median_microseconds"]
    )

    binaries = {
        "candidate_test_cq3_sha256": sha256_file(args.candidate_build / "bin/test-cq3"),
        "candidate_test_backend_ops_sha256": sha256_file(args.candidate_build / "bin/test-backend-ops"),
        "control_test_backend_ops_sha256": sha256_file(args.control_build / "bin/test-backend-ops"),
    }
    report = {
        "schema": "clark_llama_cpp_cq3_arm_receipt_v1",
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_bytes(prereg_bytes),
            "rules": prereg,
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cpu_brand": sysctl("machdep.cpu.brand_string"),
            "physical_cpus": sysctl("hw.physicalcpu"),
            "os_version": sysctl("kern.osversion"),
        },
        "sources": sources,
        "builds": {
            "receipts": build_receipts,
            "binaries": binaries,
            "candidate_cmake_cache_sha256": sha256_file(args.candidate_build / "CMakeCache.txt"),
            "control_cmake_cache_sha256": sha256_file(args.control_build / "CMakeCache.txt"),
        },
        "correctness": {
            "test_cq3": candidate_cq3,
            "backend_ops": backend_correctness,
            "backend_case_count": cq3_cases,
            "passed": True,
        },
        "performance": {
            "warmups": warmups,
            "samples": samples,
            "microseconds_per_run": by_variant,
            "median_microseconds_per_run": medians,
            "median_speedup": median_speedup,
            "paired_speedup_median": statistics.median(paired_speedups),
            "paired_speedup_minimum": min(paired_speedups),
            "paired_speedup_maximum": max(paired_speedups),
        },
        "promotion": {
            "passed": passed,
            "rules": rules,
            "scope": "ARM CQ3_0 single-token GGML MUL_MAT dispatch only",
            "authorizes_model_or_breakthrough_claim": False,
            "capability_eval_required_after_model_graph_integration": True,
        },
    }
    atomic_json(args.receipt, report)
    print(json.dumps({
        "receipt": str(args.receipt),
        "receipt_sha256": sha256_file(args.receipt),
        "candidate_median_us": medians["candidate"],
        "control_median_us": medians["control"],
        "median_speedup": median_speedup,
        "promotion_passed": passed,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
