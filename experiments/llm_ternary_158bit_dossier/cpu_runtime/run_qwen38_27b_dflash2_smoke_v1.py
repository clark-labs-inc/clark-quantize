#!/usr/bin/env python3
"""Restartable exact-greedy speculative CPU feasibility smoke for Qwen3.8-27B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request


PROMPTS = (
    ("code", "Implement binary search in Python. Return only code."),
    ("mathematical_reasoning", "Find all integer solutions to x squared minus y squared equals 45. Explain briefly."),
    ("technical_prose", "Explain in four sentences why CPU language-model decode is memory-bandwidth bound."),
)

EXPECTED_TARGET_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
EXPECTED_DFLASH2_SHA256 = "18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594"
EXPECTED_DSPARK_SHA256 = "b007a76b2ce57c1a3ecf36046543aebf60a763861624647d653f16d336c781d2"

DRAFT_SPECS = {
    "dflash2": {
        "sha256": EXPECTED_DFLASH2_SHA256,
        "bytes": 1_143_006_752,
        "spec_type": "draft-dflash",
    },
    "dspark": {
        "sha256": EXPECTED_DSPARK_SHA256,
        "bytes": 1_455_376_576,
        "spec_type": "draft-dspark",
    },
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during load with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server was not ready within {timeout:.1f}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def process_memory(pid: int) -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "vm_peak_kib": None,
        "vm_hwm_kib": None,
        "vm_rss_kib": None,
        "vm_swap_kib": None,
    }
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return values
    keys = {
        "VmPeak": "vm_peak_kib",
        "VmHWM": "vm_hwm_kib",
        "VmRSS": "vm_rss_kib",
        "VmSwap": "vm_swap_kib",
    }
    for line in status.read_text().splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in keys:
            values[keys[name]] = int(raw.strip().split()[0])
    return values


def system_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            result[f"{name.lower()}_kib"] = int(raw.strip().split()[0])
    return result


def server_command(args: argparse.Namespace, label: str, port: int) -> list[str]:
    command = [
        "taskset",
        "-c",
        args.cpu_list,
        str(args.server),
        "-m",
        str(args.target),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(args.context),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads),
        "-ngl",
        "0",
        "-np",
        "1",
        "--fit",
        "on",
        "--no-webui",
        "-lv",
        str(args.server_verbosity),
    ]
    if label == args.draft_kind:
        command.extend(
            [
                "-md",
                str(args.draft),
                "--spec-type",
                DRAFT_SPECS[args.draft_kind]["spec_type"],
                "--spec-draft-n-max",
                str(args.draft_n_max),
                "-td",
                str(args.threads),
                "-tbd",
                str(args.threads),
                "-ngld",
                "0",
            ]
        )
        if args.draft_p_min > 0.0:
            command.extend(["--spec-draft-p-min", str(args.draft_p_min)])
    elif label != "target":
        raise ValueError(f"unsupported configuration: {label}")
    if args.disable_repack:
        command.append("--no-repack")
    if label == args.draft_kind and args.enable_draft_repack:
        command.append("--spec-draft-repack")
    return command


def response_path(output_dir: Path, label: str, prompt_index: int) -> Path:
    return output_dir / label / f"prompt-{prompt_index:02d}.json"


def completed_response(path: Path, label: str, prompt: str, n_predict: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("terminal") is True
        and payload.get("configuration") == label
        and payload.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest()
        and payload.get("n_predict") == n_predict
        and len(payload.get("response", {}).get("tokens", [])) == n_predict
    )


def run_configuration(args: argparse.Namespace, label: str, config_index: int) -> None:
    expected = [
        response_path(args.output_dir, label, index) for index in range(len(PROMPTS))
    ]
    memory_path = args.output_dir / label / "memory.json"
    if memory_path.exists() and all(
        completed_response(path, label, prompt, args.n_predict)
        for path, (_, prompt) in zip(expected, PROMPTS)
    ):
        print(json.dumps({"event": "configuration_resumed", "configuration": label}), flush=True)
        return

    port = args.port_base + config_index
    command = server_command(args, label, port)
    log_path = args.output_dir / label / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        samples: list[dict[str, Any]] = []
        try:
            wait_ready(process, port, args.load_timeout)
            samples.append({"stage": "ready", "process": process_memory(process.pid), "system": system_memory()})
            for prompt_index, (prompt_class, prompt) in enumerate(PROMPTS):
                path = response_path(args.output_dir, label, prompt_index)
                if completed_response(path, label, prompt, args.n_predict):
                    continue
                request_payload = {
                    "prompt": prompt,
                    "n_predict": args.n_predict,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 42,
                    "cache_prompt": False,
                    "stream": False,
                    "return_tokens": True,
                    "ignore_eos": True,
                }
                started = time.monotonic()
                response = post_json(
                    f"http://127.0.0.1:{port}/completion",
                    request_payload,
                    timeout=args.request_timeout,
                )
                wall_seconds = time.monotonic() - started
                tokens = response.get("tokens", [])
                if len(tokens) != args.n_predict:
                    raise RuntimeError(f"{label} prompt {prompt_index} returned {len(tokens)} tokens")
                receipt = {
                    "schema": "qwen38_27b_speculative_smoke_prompt_receipt_v1",
                    "terminal": True,
                    "configuration": label,
                    "prompt_index": prompt_index,
                    "prompt_class": prompt_class,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "n_predict": args.n_predict,
                    "client_wall_seconds": wall_seconds,
                    "response": response,
                }
                atomic_json(path, receipt)
                sample = {
                    "stage": f"prompt_{prompt_index:02d}",
                    "process": process_memory(process.pid),
                    "system": system_memory(),
                }
                samples.append(sample)
                print(
                    json.dumps(
                        {
                            "event": "prompt_terminal",
                            "configuration": label,
                            "prompt_index": prompt_index,
                            "accepted_tokens_per_second": response["timings"]["predicted_per_second"],
                            "draft_n": response["timings"].get("draft_n", 0),
                            "draft_n_accepted": response["timings"].get("draft_n_accepted", 0),
                            "vm_hwm_kib": sample["process"]["vm_hwm_kib"],
                        }
                    ),
                    flush=True,
                )
            atomic_json(
                memory_path,
                {
                    "schema": "qwen38_27b_speculative_smoke_memory_v1",
                    "terminal": True,
                    "configuration": label,
                    "samples": samples,
                },
            )
        finally:
            stop_server(process)


def load_receipts(output_dir: Path, label: str) -> list[dict[str, Any]]:
    return [
        json.loads(response_path(output_dir, label, index).read_text())
        for index in range(len(PROMPTS))
    ]


def weighted_tps(receipts: list[dict[str, Any]]) -> float:
    timings = [receipt["response"]["timings"] for receipt in receipts]
    accepted_decode_tokens = sum(int(timing["predicted_n"]) - 1 for timing in timings)
    predicted_seconds = sum(float(timing["predicted_ms"]) for timing in timings) / 1000.0
    return accepted_decode_tokens / predicted_seconds


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    target = load_receipts(args.output_dir, "target")
    candidate = load_receipts(args.output_dir, args.draft_kind)
    parity = [
        candidate_row["response"]["tokens"] == control["response"]["tokens"]
        for control, candidate_row in zip(target, candidate)
    ]
    target_tps = weighted_tps(target)
    candidate_tps = weighted_tps(candidate)
    timings = [receipt["response"]["timings"] for receipt in candidate]
    accepted_decode_tokens = sum(int(timing["predicted_n"]) - 1 for timing in timings)
    draft_n = sum(int(timing.get("draft_n", 0)) for timing in timings)
    draft_n_accepted = sum(int(timing.get("draft_n_accepted", 0)) for timing in timings)
    target_calls = accepted_decode_tokens - draft_n_accepted
    memory_payload = json.loads((args.output_dir / args.draft_kind / "memory.json").read_text())
    vm_hwm_values = [
        sample["process"]["vm_hwm_kib"]
        for sample in memory_payload["samples"]
        if sample["process"]["vm_hwm_kib"] is not None
    ]
    vm_swap_values = [
        sample["process"]["vm_swap_kib"]
        for sample in memory_payload["samples"]
        if sample["process"]["vm_swap_kib"] is not None
    ]
    max_vm_hwm_kib = max(vm_hwm_values)
    max_vm_swap_kib = max(vm_swap_values)
    checks = {
        "three_prompts_x_64_tokens": len(candidate) == 3 and all(len(row["response"]["tokens"]) == args.n_predict for row in candidate),
        "exact_greedy_token_parity_3_of_3": all(parity),
        "no_server_swap": max_vm_swap_kib == 0,
        "candidate_vm_hwm_at_most_28_5_gib": max_vm_hwm_kib <= int(28.5 * 1024 * 1024),
    }
    return {
        "schema": "qwen38_27b_speculative_amd_cpu_feasibility_smoke_summary_v1",
        "terminal": True,
        "passed": all(checks.values()),
        "classification": "gate_0_and_gate_1_feasibility_smoke_only",
        "checks": checks,
        "model": "Qwen/Qwen3.8-27B",
        "artifacts": {
            "target": {"path": str(args.target), "bytes": args.target.stat().st_size, "expected_sha256": args.target_sha256},
            "draft": {
                "kind": args.draft_kind,
                "path": str(args.draft),
                "bytes": args.draft.stat().st_size,
                "sha256": sha256_file(args.draft),
            },
        },
        "runtime": {
            "path": str(args.server),
            "sha256": sha256_file(args.server),
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "context": args.context,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "target_repack": not args.disable_repack,
            "draft_repack": args.enable_draft_repack or not args.disable_repack,
            "draft_n_max": args.draft_n_max,
            "draft_p_min": args.draft_p_min,
        },
        "generation": {"prompts": len(PROMPTS), "tokens_per_prompt": args.n_predict, "temperature": 0.0, "top_k": 1, "seed": 42},
        "parity": {"per_prompt": parity, "exact": all(parity)},
        "throughput_diagnostic": {
            "target_weighted_tps": target_tps,
            "candidate_kind": args.draft_kind,
            "candidate_weighted_tps": candidate_tps,
            "candidate_over_target": candidate_tps / target_tps,
            "candidate_median_prompt_tps": statistics.median(float(timing["predicted_per_second"]) for timing in timings),
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
            "draft_acceptance_fraction": draft_n_accepted / draft_n if draft_n else None,
            "estimated_target_verification_calls": target_calls,
            "mean_committed_tokens_per_target_call": accepted_decode_tokens / target_calls if target_calls else None,
        },
        "memory": {"maximum_vm_hwm_kib": max_vm_hwm_kib, "maximum_vm_swap_kib": max_vm_swap_kib},
        "claim_boundary": "Infrastructure and deterministic greedy-parity smoke only. Throughput is diagnostic until the frozen paired rotation is complete.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--draft", "--dflash2-draft", dest="draft", type=Path, required=True)
    parser.add_argument("--draft-kind", choices=sorted(DRAFT_SPECS), default="dflash2")
    parser.add_argument("--draft-n-max", type=int, default=7)
    parser.add_argument("--draft-p-min", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-list", default="0,2,4,6")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ubatch", type=int, default=64)
    parser.add_argument("--port-base", type=int, default=18920)
    parser.add_argument("--server-verbosity", type=int, default=3)
    parser.add_argument("--disable-repack", action="store_true")
    parser.add_argument("--enable-draft-repack", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.server, args.target, args.draft):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.target.stat().st_size != 18973870432:
        raise ValueError("unexpected target size")
    draft_spec = DRAFT_SPECS[args.draft_kind]
    if args.draft.stat().st_size != draft_spec["bytes"]:
        raise ValueError(f"unexpected {args.draft_kind} draft size")
    if args.target_sha256 != EXPECTED_TARGET_SHA256:
        raise ValueError("unexpected frozen target SHA-256 argument")
    if sha256_file(args.draft) != draft_spec["sha256"]:
        raise ValueError(f"{args.draft_kind} draft SHA-256 mismatch")
    if args.n_predict != 64:
        raise ValueError("the frozen smoke requires exactly 64 generated tokens")
    if not 1 <= args.draft_n_max <= 7:
        raise ValueError("the frozen Qwen3.8 draft models support widths 1 through 7")
    if not 0.0 <= args.draft_p_min <= 1.0:
        raise ValueError("--draft-p-min must be in [0, 1]")
    if args.enable_draft_repack and not args.disable_repack:
        raise ValueError("--enable-draft-repack is only meaningful with --disable-repack")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(("target", args.draft_kind)):
        run_configuration(args, label, index)
    summary = summarize(args)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"event": "terminal", "passed": summary["passed"], "throughput": summary["throughput_diagnostic"]}, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
