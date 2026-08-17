#!/usr/bin/env python3
"""Run the frozen same-binary CQ3 gate/up/SwiGLU ARM graph experiment."""
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
    if completed.returncode != 0:
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


def validate_source(source: Path, frozen: dict) -> dict:
    if not source.is_dir():
        raise ValueError(f"missing source directory: {source}")
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
        raise ValueError("source revision differs from preregistration")
    if sha256_bytes(diff) != frozen["full_diff_sha256"]:
        raise ValueError("source diff differs from preregistration")
    if diff_check.returncode != 0 or diff_check.stdout or diff_check.stderr:
        raise ValueError("source fails git diff --check")
    actual_files = {
        relative: sha256_file(source / relative)
        for relative in frozen["file_sha256"]
    }
    if actual_files != frozen["file_sha256"]:
        raise ValueError("frozen source file hashes differ")
    return {
        "path": str(source),
        "revision": revision,
        "full_diff_sha256": sha256_bytes(diff),
        "file_sha256": actual_files,
    }


def build_env(build: Path, *, fusion_enabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "LC_ALL": "C",
        "LANG": "C",
        "NO_COLOR": "1",
        "DYLD_LIBRARY_PATH": str(build / "bin"),
        "GGML_CPU_DISABLE_FUSION": "0" if fusion_enabled else "1",
    })
    return env


def benchmark_once(
    build: Path,
    test_filter: str,
    *,
    threads: int,
    fusion_enabled: bool,
) -> dict:
    result = run([
        str(build / "bin/test-backend-ops"),
        "perf", "-b", "CPU", "-t", str(threads),
        "-o", "MUL_MAT_VEC_FUSION",
        "-p", test_filter,
        "--output", "console",
    ], env=build_env(build, fusion_enabled=fusion_enabled), timeout=180)
    matches = PERF_RE.findall(result["stdout"] + result["stderr"])
    if len(matches) != 1:
        raise RuntimeError(f"expected one performance result, found {len(matches)}")
    result["microseconds_per_run"] = float(matches[0])
    return result


def sysctl(name: str) -> str | None:
    completed = subprocess.run(
        ["sysctl", "-n", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=HERE / "llama_cpp_cq3_fused_arm_preregistration_v1.json",
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
        prereg.get("schema") != "clark_llama_cpp_cq3_fused_arm_preregistration_v1"
        or prereg.get("status") != "frozen_before_authoritative_paired_run"
        or args.warmups != performance.get("warmups")
        or args.repetitions != performance.get("repetitions")
    ):
        raise ValueError("benchmark arguments differ from frozen preregistration")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the v1 authoritative run is scoped to Darwin arm64")
    if (os.cpu_count() or 0) < max(performance["thread_counts"]):
        raise RuntimeError("host exposes fewer CPUs than the frozen thread counts")

    for patch in prereg["patches"].values():
        if sha256_file(HERE / patch["path"]) != patch["sha256"]:
            raise ValueError(f"versioned patch identity changed: {patch['path']}")
    if sha256_file(Path(__file__)) != prereg["benchmark_sha256"]:
        raise ValueError("benchmark source changed after preregistration")

    source_receipt = validate_source(args.source, prereg["source"])
    cache = args.build / "CMakeCache.txt"
    if not cache.is_file() or sha256_file(cache) != prereg["build"]["cmake_cache_sha256"]:
        raise ValueError("CMake configuration differs from preregistration")

    build_receipt = run([
        "cmake", "--build", str(args.build),
        "--target", "test-cq3", "test-backend-ops", "-j", "8",
    ])
    binaries = {
        name: sha256_file(args.build / "bin" / name)
        for name in ("test-cq3", "test-backend-ops", "libggml-cpu.dylib")
    }
    cq3_test = run(
        [str(args.build / "bin/test-cq3")],
        env=build_env(args.build, fusion_enabled=True),
    )
    backend_test = run([
        str(args.build / "bin/test-backend-ops"),
        "test", "-b", "CPU", "-t", str(performance["correctness_threads"]),
        "-o", "MUL_MAT_VEC_FUSION",
        "-p", prereg["correctness"]["test_filter"],
        "--output", "console",
    ], env=build_env(args.build, fusion_enabled=True))
    correctness_text = backend_test["stdout"] + backend_test["stderr"]
    case_count = correctness_text.count("MUL_MAT_VEC_FUSION(type=cq3_0")
    correctness_passed = (
        case_count == prereg["correctness"]["expected_backend_cases"]
        and "FAIL" not in correctness_text
    )
    if not correctness_passed:
        raise RuntimeError(f"fused graph correctness failed: observed {case_count} cases")

    lanes: dict[str, dict] = {}
    for threads in performance["thread_counts"]:
        warmups: list[dict] = []
        for _ in range(args.warmups):
            warmups.append({
                "control": benchmark_once(
                    args.build, performance["test_filter"], threads=threads,
                    fusion_enabled=False,
                ),
                "candidate": benchmark_once(
                    args.build, performance["test_filter"], threads=threads,
                    fusion_enabled=True,
                ),
            })

        samples: list[dict] = []
        by_variant: dict[str, list[float]] = {"control": [], "candidate": []}
        for repetition in range(args.repetitions):
            order = (
                ("control", "candidate")
                if repetition % 2 == 0
                else ("candidate", "control")
            )
            pair: dict[str, object] = {"repetition": repetition, "order": list(order)}
            for variant in order:
                measured = benchmark_once(
                    args.build,
                    performance["test_filter"],
                    threads=threads,
                    fusion_enabled=variant == "candidate",
                )
                pair[variant] = measured
                by_variant[variant].append(measured["microseconds_per_run"])
            control_us = float(pair["control"]["microseconds_per_run"])
            candidate_us = float(pair["candidate"]["microseconds_per_run"])
            pair["control_over_candidate_speedup"] = control_us / candidate_us
            samples.append(pair)

        control_median = statistics.median(by_variant["control"])
        candidate_median = statistics.median(by_variant["candidate"])
        paired_speedups = [
            float(pair["control_over_candidate_speedup"]) for pair in samples
        ]
        summary = {
            "control_microseconds": {
                "median": control_median,
                "minimum": min(by_variant["control"]),
                "maximum": max(by_variant["control"]),
                "samples": by_variant["control"],
            },
            "candidate_microseconds": {
                "median": candidate_median,
                "minimum": min(by_variant["candidate"]),
                "maximum": max(by_variant["candidate"]),
                "samples": by_variant["candidate"],
            },
            "ratio_of_medians": control_median / candidate_median,
            "paired_speedup": {
                "median": statistics.median(paired_speedups),
                "minimum": min(paired_speedups),
                "maximum": max(paired_speedups),
            },
        }
        lane_rule = prereg["promotion"]["thread_rules"][str(threads)]
        lane_passed = (
            summary["ratio_of_medians"] >= lane_rule["minimum_ratio_of_medians"]
            and sum(value > 1.0 for value in paired_speedups)
            >= lane_rule["minimum_paired_wins"]
        )
        lanes[str(threads)] = {
            "warmups": warmups,
            "samples": samples,
            "summary": summary,
            "promotion_rule": lane_rule,
            "passed": lane_passed,
        }

    gate_passed = correctness_passed and all(lane["passed"] for lane in lanes.values())
    receipt = {
        "schema": "clark_llama_cpp_cq3_fused_arm_receipt_v1",
        "created_at_unix": time.time(),
        "status": "gate_passed" if gate_passed else "gate_failed",
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_bytes(prereg_bytes),
            "rules": prereg,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "brand": sysctl("machdep.cpu.brand_string"),
            "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        },
        "source": source_receipt,
        "build": {
            "path": str(args.build),
            "cmake_cache_sha256": sha256_file(cache),
            "rebuild": build_receipt,
            "binary_sha256": binaries,
        },
        "correctness": {
            "test_cq3": cq3_test,
            "backend_test": backend_test,
            "observed_backend_cases": case_count,
            "passed": correctness_passed,
        },
        "performance": {"lanes": lanes},
        "promotion": {
            "passed": gate_passed,
            "scope": prereg["promotion"]["scope"],
            "claim_boundary": prereg["promotion"]["claim_boundary"],
        },
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "receipt": str(args.receipt),
        "status": receipt["status"],
        "lanes": {
            threads: {
                "ratio_of_medians": lane["summary"]["ratio_of_medians"],
                "paired_median": lane["summary"]["paired_speedup"]["median"],
                "passed": lane["passed"],
            }
            for threads, lane in lanes.items()
        },
    }, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
