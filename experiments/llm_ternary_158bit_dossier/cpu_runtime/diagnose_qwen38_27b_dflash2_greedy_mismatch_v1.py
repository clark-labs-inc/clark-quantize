#!/usr/bin/env python3
"""Capture target and DFlash2 top logits at the frozen greedy mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from run_qwen38_27b_dflash2_smoke_v1 import atomic_json, post_json, server_command, stop_server, wait_ready


PROMPT = "Explain in four sentences why CPU language-model decode is memory-bandwidth bound."


def run_one(args: argparse.Namespace, label: str, index: int) -> dict[str, Any]:
    port = args.port_base + index
    command = server_command(args, label, port)
    log_path = args.output_dir / label / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_ready(process, port, args.load_timeout)
            started = time.monotonic()
            response = post_json(
                f"http://127.0.0.1:{port}/completion",
                {
                    "prompt": PROMPT,
                    "n_predict": 36,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 42,
                    "cache_prompt": False,
                    "stream": False,
                    "return_tokens": True,
                    "ignore_eos": True,
                    "n_probs": 5,
                },
                timeout=args.request_timeout,
            )
            payload = {
                "schema": "qwen38_27b_dflash2_greedy_mismatch_response_v1",
                "terminal": True,
                "configuration": label,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "client_wall_seconds": time.monotonic() - started,
                "response": response,
            }
            atomic_json(args.output_dir / label / "response.json", payload)
            return payload
        finally:
            stop_server(process)


def probability_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("completion_probabilities", "probs"):
        rows = response.get(key)
        if isinstance(rows, list):
            return rows
    raise KeyError("response has no completion probability rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dflash2-draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-list", default="0,2,4,6")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ubatch", type=int, default=64)
    parser.add_argument("--port-base", type=int, default=18940)
    parser.add_argument("--server-verbosity", type=int, default=4)
    parser.add_argument("--load-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = run_one(args, "target", 0)
    dflash2 = run_one(args, "dflash2", 1)
    target_tokens = target["response"]["tokens"]
    dflash2_tokens = dflash2["response"]["tokens"]
    first = next((i for i, pair in enumerate(zip(target_tokens, dflash2_tokens)) if pair[0] != pair[1]), None)
    target_probs = probability_rows(target["response"])
    dflash2_probs = probability_rows(dflash2["response"])
    summary = {
        "schema": "qwen38_27b_dflash2_greedy_mismatch_diagnostic_summary_v1",
        "terminal": True,
        "first_mismatch_index": first,
        "target_token": target_tokens[first] if first is not None else None,
        "dflash2_token": dflash2_tokens[first] if first is not None else None,
        "target_probability_row": target_probs[first] if first is not None else None,
        "dflash2_probability_row": dflash2_probs[first] if first is not None else None,
        "dflash2_selected_token_is_its_target_top_candidate": (
            first is not None
            and dflash2_probs[first].get("id") == dflash2_tokens[first]
        ),
        "runtime_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
        "claim_boundary": "One-prompt mismatch diagnosis only.",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
