#!/usr/bin/env python3
"""Run a frozen math/code forgetting canary against llama.cpp's chat API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def extract_last_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    open_brace = text.find("{", starts[-1])
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("$").strip()
    for source, target in (
        (r"\left", ""),
        (r"\right", ""),
        (r"\!", ""),
        (r"\,", ""),
        (r"\;", ""),
        (" ", ""),
        ("−", "-"),
        (r"\dfrac", r"\frac"),
        (r"\tfrac", r"\frac"),
    ):
        value = value.replace(source, target)
    if re.fullmatch(r"-?[0-9,]+(?:\.[0-9]+)?", value):
        value = value.replace(",", "")
    if value.endswith(".0") and re.fullmatch(r"-?\d+\.0", value):
        value = value[:-2]
    return value


def normalize_code_output(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:text|python)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value).strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    value = lines[0] if lines else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def request_chat(url: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict[str, Any], float]:
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260817,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"chat API returned HTTP {error.code}: {detail}") from error
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8091")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    claimed = manifest["content_sha256"]
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_digest(body) != claimed:
        raise RuntimeError("manifest content hash mismatch")

    started = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    max_tokens = int(manifest["prompt_contract"]["max_tokens"])

    for item in manifest["math"]["items"]:
        prompt = item["problem"] + "\n" + manifest["prompt_contract"]["math"]
        response, latency_ms = request_chat(args.server, prompt, max_tokens, args.timeout)
        choice = response["choices"][0]
        output = choice["message"].get("content") or ""
        prediction = normalize_math(extract_last_boxed(output))
        gold = normalize_math(item["answer"])
        records.append(
            {
                "task": "math500",
                "id": item["id"],
                "correct": prediction == gold,
                "prediction": prediction,
                "gold": gold,
                "output": output,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "latency_ms": latency_ms,
            }
        )
        print(json.dumps({key: records[-1][key] for key in ("task", "id", "correct", "prediction", "gold")}), flush=True)

    for item in manifest["code"]["items"]:
        response, latency_ms = request_chat(args.server, item["prompt"], max_tokens, args.timeout)
        choice = response["choices"][0]
        output = choice["message"].get("content") or ""
        prediction = normalize_code_output(output)
        gold = item["answer"].strip()
        records.append(
            {
                "task": "code_trace",
                "id": item["id"],
                "correct": prediction == gold,
                "prediction": prediction,
                "gold": gold,
                "output": output,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "latency_ms": latency_ms,
            }
        )
        print(json.dumps({key: records[-1][key] for key in ("task", "id", "correct", "prediction", "gold")}), flush=True)

    summaries = []
    for task in ("math500", "code_trace"):
        selected = [record for record in records if record["task"] == task]
        correct = sum(bool(record["correct"]) for record in selected)
        summaries.append(
            {"task": task, "correct": correct, "n": len(selected), "accuracy": correct / len(selected)}
        )

    finished = datetime.now(timezone.utc)
    report = {
        "schema": "cpu_math_code_canary_result_v1",
        "model_label": args.model_label,
        "model_sha256": args.model_sha256,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest": str(args.manifest),
        "manifest_content_sha256": claimed,
        "server": args.server,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "summaries": summaries,
        "records": records,
    }
    report["report_content_sha256"] = canonical_digest(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"out": str(args.out), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
