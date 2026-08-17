#!/usr/bin/env python3
"""Run the frozen CUTE-Quick-20 character-level forgetting tripwire.

The task prompts below are from Leukas/CUTE at the revision pinned in the
manifest and are used under that repository's MIT license. This evaluator is a
small development slice, not a replacement for or score on the full benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPTS = {
    "contains_char": """Answer whether the specified letter is in the given word, based on the following examples:

1. Is there a "a" in "alphabet"? Answer: "Yes"
2. Is there a "z" in "alphabet"? Answer: "No"
3. Is there a "u" in "hello"? Answer: "No"
4. Is there a "o" in "hello"? Answer: "Yes"

Question: Is there a "{}" in "{}"?""",
    "ins_char": """Add the specified letter after every instance of the second specified letter in a given word, based on the following examples:

1. Add an "e" after every "a" in "alphabet". Answer: "aelphaebet"
2. Add an "l" after every "l" in "hello". Answer: "hellllo"
3. Add an "t" after every "z" in "zebra". Answer: "ztebra"
4. Add an "f" after every "u" in "tongue". Answer: "tongufe"

Question: Add an "{}" after every "{}" in "{}".""",
    "del_char": """Delete every instance of a specified letter in a given word, based on the following examples:

1. Delete every instance of "a" in "alphabet". Answer: "lphbet"
2. Delete every instance of "l" in "hello". Answer: "heo"
3. Delete every instance of "z" in "zebra". Answer: "ebra"
4. Delete every instance of "u" in "tongue". Answer: "tonge"

Question: Delete every instance of "{}" in "{}".""",
    "swap_char": """Swap the positions of two specified letters in a given word, based on the following examples:

1. Swap "l" and "b" in "alphabet". Answer: "abphalet"
2. Swap "h" and "e" in "hello". Answer: "ehllo"
3. Swap "z" and "a" in "zebra". Answer: "aebrz"
4. Swap "u" and "e" in "tongue". Answer: "tongeu"

Question: Swap "{}" and "{}" in "{}".""",
    "sub_char": """Substitute the first specified letter with the second specified letter in a given word, based on the following examples:

1. Substitute "a" with "b" in "alphabet". Answer: "blphbbet"
2. Substitute "h" with "e" in "hello". Answer: "eello"
3. Substitute "z" with "a" in "zebra". Answer: "aebra"
4. Substitute "u" with "e" in "tongue". Answer: "tongee"

Question: Substitute "{}" with "{}" in "{}".""",
}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_answer(text: str) -> str:
    """Remove only common response wrappers; keep answer semantics strict."""
    value = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    value = re.sub(r"^```(?:text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value).strip()
    if value.lower().startswith("answer:"):
        value = value.split(":", 1)[1].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    value = lines[0] if lines else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def request_chat(
    url: str, prompt: str, max_tokens: int, timeout: float, seed: int
) -> tuple[dict[str, Any], float]:
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
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
    if manifest["selection"]["leaderboard_comparable"]:
        raise RuntimeError("quick slice must not claim leaderboard comparability")

    contract = manifest["prompt_contract"]
    started = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    task_counts = Counter(item["task"] for item in manifest["items"])
    if task_counts != Counter({task: 4 for task in PROMPTS}):
        raise RuntimeError(f"unexpected task balance: {task_counts}")

    for item in manifest["items"]:
        task = item["task"]
        prompt = PROMPTS[task].format(*item["inputs"])
        prompt += "\n" + contract["suffix"]
        response, latency_ms = request_chat(
            args.server,
            prompt,
            int(contract["max_tokens"]),
            args.timeout,
            int(contract["seed"]),
        )
        choice = response["choices"][0]
        output = choice["message"].get("content") or ""
        prediction = normalize_answer(output)
        record = {
            "task": task,
            "id": item["id"],
            "source_line": item["source_line"],
            "correct": prediction == item["label"],
            "prediction": prediction,
            "gold": item["label"],
            "output": output,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "latency_ms": latency_ms,
        }
        records.append(record)
        print(
            json.dumps(
                {key: record[key] for key in ("task", "id", "correct", "prediction", "gold")}
            ),
            flush=True,
        )

    summaries = []
    for task in PROMPTS:
        selected = [record for record in records if record["task"] == task]
        correct = sum(bool(record["correct"]) for record in selected)
        summaries.append(
            {"task": task, "correct": correct, "n": len(selected), "accuracy": correct / len(selected)}
        )
    correct = sum(bool(record["correct"]) for record in records)
    summaries.append(
        {"task": "cute_quick_20", "correct": correct, "n": len(records), "accuracy": correct / len(records)}
    )

    finished = datetime.now(timezone.utc)
    report = {
        "schema": "cpu_cute_quick_result_v1",
        "model_label": args.model_label,
        "model_sha256": args.model_sha256,
        "manifest": str(args.manifest),
        "manifest_content_sha256": claimed,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "leaderboard_comparable": False,
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
