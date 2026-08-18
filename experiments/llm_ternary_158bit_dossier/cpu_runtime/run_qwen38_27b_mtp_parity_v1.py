#!/usr/bin/env python3
"""Restartable exact-greedy Qwen3.8-27B MTP CPU preflight."""

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
    ("general_instruction", "List five practical ways to make a scientific experiment reproducible."),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def response_path(output_dir: Path, label: str, prompt_index: int) -> Path:
    return output_dir / label / f"prompt-{prompt_index:02d}.json"


def completed_response(
    path: Path,
    *,
    label: str,
    prompt: str,
    n_predict: int,
) -> bool:
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


def server_command(args: argparse.Namespace, width: int, port: int) -> list[str]:
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
    ]
    if width:
        command.extend(
            [
                "-md",
                str(args.draft),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(width),
                "--spec-draft-p-min",
                str(args.draft_p_min),
                "-td",
                str(args.threads),
                "-tbd",
                str(args.threads),
                "-ngld",
                "0",
            ]
        )
    return command


def run_configuration(args: argparse.Namespace, width: int, config_index: int) -> None:
    label = "target" if width == 0 else f"mtp{width}"
    expected = [
        response_path(args.output_dir, label, index)
        for index in range(len(PROMPTS))
    ]
    if all(
        completed_response(path, label=label, prompt=prompt, n_predict=args.n_predict)
        for path, (_, prompt) in zip(expected, PROMPTS)
    ):
        print(json.dumps({"event": "configuration_resumed", "configuration": label}), flush=True)
        return

    port = args.port_base + config_index
    command = server_command(args, width, port)
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
        try:
            wait_ready(process, port, args.load_timeout)
            for prompt_index, (prompt_class, prompt) in enumerate(PROMPTS):
                path = response_path(args.output_dir, label, prompt_index)
                if completed_response(
                    path,
                    label=label,
                    prompt=prompt,
                    n_predict=args.n_predict,
                ):
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
                    raise RuntimeError(
                        f"{label} prompt {prompt_index} returned {len(tokens)} tokens"
                    )
                atomic_json(
                    path,
                    {
                        "schema": "qwen38_27b_mtp_prompt_receipt_v1",
                        "terminal": True,
                        "configuration": label,
                        "draft_width": width,
                        "prompt_index": prompt_index,
                        "prompt_class": prompt_class,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "n_predict": args.n_predict,
                        "client_wall_seconds": wall_seconds,
                        "response": response,
                    },
                )
                print(
                    json.dumps(
                        {
                            "event": "prompt_terminal",
                            "configuration": label,
                            "prompt_index": prompt_index,
                            "accepted_tokens_per_second": response["timings"][
                                "predicted_per_second"
                            ],
                            "draft_n": response["timings"].get("draft_n", 0),
                            "draft_n_accepted": response["timings"].get(
                                "draft_n_accepted", 0
                            ),
                        }
                    ),
                    flush=True,
                )
        finally:
            stop_server(process)


def load_prompt_receipts(output_dir: Path, label: str) -> list[dict[str, Any]]:
    return [
        json.loads(response_path(output_dir, label, index).read_text())
        for index in range(len(PROMPTS))
    ]


def summarize(args: argparse.Namespace, widths: list[int]) -> dict[str, Any]:
    target_receipts = load_prompt_receipts(args.output_dir, "target")
    target_tokens = [receipt["response"]["tokens"] for receipt in target_receipts]
    configurations: list[dict[str, Any]] = []
    all_parity = True
    target_tps = 0.0
    for width in widths:
        label = "target" if width == 0 else f"mtp{width}"
        receipts = load_prompt_receipts(args.output_dir, label)
        timings = [receipt["response"]["timings"] for receipt in receipts]
        output_tokens = sum(int(timing["predicted_n"]) for timing in timings)
        # llama-server produces the first output token during prompt eval. Its
        # reported decode timing covers the remaining predicted_n - 1 tokens.
        # Keep that TTFT token out of the decode numerator so the weighted value
        # matches the runtime's per-prompt predicted_per_second convention.
        accepted_decode_tokens = sum(
            int(timing["predicted_n"]) - 1 for timing in timings
        )
        predicted_seconds = sum(float(timing["predicted_ms"]) for timing in timings) / 1000.0
        weighted_tps = accepted_decode_tokens / predicted_seconds
        parity = [
            receipt["response"]["tokens"] == expected
            for receipt, expected in zip(receipts, target_tokens)
        ]
        config_parity = all(parity)
        if width:
            all_parity = all_parity and config_parity
        else:
            target_tps = weighted_tps
        draft_n = sum(int(timing.get("draft_n", 0)) for timing in timings)
        draft_accepted = sum(
            int(timing.get("draft_n_accepted", 0)) for timing in timings
        )
        # Every target verification commits one target-sampled token in addition
        # to any accepted draft prefix. This identity makes target-call
        # amortization observable without relying on a private runtime counter.
        target_calls = accepted_decode_tokens - draft_accepted
        target_bytes_per_accepted_token = (
            args.target.stat().st_size * target_calls / accepted_decode_tokens
        )
        draft_bytes_per_accepted_token = (
            args.draft.stat().st_size * draft_n / accepted_decode_tokens
            if width
            else 0.0
        )
        streamed_weight_bytes_per_accepted_token = (
            target_bytes_per_accepted_token + draft_bytes_per_accepted_token
        )
        configurations.append(
            {
                "configuration": label,
                "draft_width": width,
                "prompt_token_parity": parity,
                "exact_token_parity": config_parity,
                "output_tokens_including_prompt_eval_token": output_tokens,
                "accepted_decode_tokens": accepted_decode_tokens,
                "accepted_tokens_per_second_weighted": weighted_tps,
                "accepted_tokens_per_second_median_prompt": statistics.median(
                    float(timing["predicted_per_second"]) for timing in timings
                ),
                "draft_n": draft_n,
                "draft_n_accepted": draft_accepted,
                "draft_acceptance_fraction": (
                    draft_accepted / draft_n if draft_n else None
                ),
                "estimated_target_verification_calls": target_calls,
                "mean_committed_tokens_per_target_call": (
                    accepted_decode_tokens / target_calls
                ),
                "target_weight_bytes_per_accepted_token_lower_bound": (
                    target_bytes_per_accepted_token
                ),
                "draft_weight_bytes_per_accepted_token_lower_bound": (
                    draft_bytes_per_accepted_token
                ),
                "combined_weight_bytes_per_accepted_token_lower_bound": (
                    streamed_weight_bytes_per_accepted_token
                ),
                "minimum_weight_bandwidth_bytes_per_second_at_100_tps": (
                    streamed_weight_bytes_per_accepted_token * 100.0
                ),
                "server_log_sha256": sha256_file(
                    args.output_dir / label / "server.log"
                ),
            }
        )
    for config in configurations:
        config["accepted_decode_speedup_vs_target"] = (
            config["accepted_tokens_per_second_weighted"] / target_tps
        )

    return {
        "schema": "qwen38_27b_mtp_parity_summary_v1",
        "terminal": True,
        "passed": all_parity,
        "classification": "development preflight only",
        "model": "Qwen/Qwen3.8-27B",
        "target": {
            "path": str(args.target),
            "bytes": args.target.stat().st_size,
            "sha256": sha256_file(args.target),
        },
        "draft": {
            "path": str(args.draft),
            "bytes": args.draft.stat().st_size,
            "sha256": sha256_file(args.draft),
        },
        "runtime": {
            "path": str(args.server),
            "sha256": sha256_file(args.server),
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "context": args.context,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "draft_p_min": args.draft_p_min,
        },
        "generation": {
            "prompt_count": len(PROMPTS),
            "tokens_per_prompt": args.n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 42,
            "ignore_eos": True,
        },
        "configurations": configurations,
        "claim_boundary": "Four-prompt exact-greedy development preflight only; no stochastic, capability, final-node, 100 accepted tok/s, or breakthrough claim.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--widths", default="0,1,2,3")
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-list", default="0,2,4,6")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ubatch", type=int, default=64)
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--draft-p-min", type=float, default=0.0)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    widths = [int(part) for part in args.widths.split(",")]
    if not widths or widths[0] != 0 or any(width < 0 for width in widths):
        raise ValueError("widths must start with target width 0 and be non-negative")
    if not 0.0 <= args.draft_p_min <= 1.0:
        raise ValueError("draft-p-min must be between 0 and 1")
    for path in (args.server, args.target, args.draft):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config_index, width in enumerate(widths):
        run_configuration(args, width, config_index)
    summary = summarize(args, widths)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"event": "terminal", "passed": summary["passed"]}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
