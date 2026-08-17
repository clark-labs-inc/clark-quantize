#!/usr/bin/env python3
"""Run the frozen four-chunk WikiText perplexity preflight with a JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--llama-perplexity",
        default=Path("/workspace/build/llama.cpp/bin/llama-perplexity"),
        type=Path,
    )
    parser.add_argument("--chunks", default=4, type=int)
    parser.add_argument("--context", default=512, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.chunks <= 0 or args.context <= 0 or args.threads <= 0:
        parser.error("chunks, context, and threads must be positive")
    for required in (args.model, args.corpus, args.llama_perplexity):
        if not required.is_file():
            raise FileNotFoundError(required)

    command = [
        str(args.llama_perplexity.resolve()),
        "-m",
        str(args.model.resolve()),
        "-f",
        str(args.corpus.resolve()),
        "-c",
        str(args.context),
        "-b",
        str(args.context),
        "-ub",
        str(args.context),
        "-t",
        str(args.threads),
        "--chunks",
        str(args.chunks),
        "--ppl-output-type",
        "1",
    ]
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started_clock
    finished = datetime.now(timezone.utc)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    estimates = re.findall(
        r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)",
        completed.stdout,
    )
    chunk_rows = [
        {
            "offset": int(match.group(1)),
            "running_ppl": float(match.group(2)),
            "running_log_ppl": float(match.group(3)),
            "running_uncertainty": float(match.group(4)),
        }
        for match in re.finditer(
            r"^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
            completed.stdout,
            flags=re.MULTILINE,
        )
    ]
    valid = completed.returncode == 0 and len(estimates) == 1 and len(chunk_rows) == args.chunks
    result: dict[str, Any] = {
        "schema": "cpu_wikitext_quick_result_v1",
        "status": "completed" if valid else "infrastructure_invalid",
        "valid": valid,
        "model_label": args.model_label,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": sha256_file(args.corpus),
        "llama_perplexity": str(args.llama_perplexity.resolve()),
        "llama_perplexity_sha256": sha256_file(args.llama_perplexity),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {
            "chunks": args.chunks,
            "context": args.context,
            "batch": args.context,
            "ubatch": args.context,
            "threads": args.threads,
            "ppl_output_type": 1,
        },
        "command": command,
        "returncode": completed.returncode,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "maximum_resident_kib": int(usage.ru_maxrss),
        "chunks": chunk_rows,
        "final_ppl": float(estimates[0][0]) if len(estimates) == 1 else None,
        "final_uncertainty": float(estimates[0][1]) if len(estimates) == 1 else None,
        "raw_output": completed.stdout,
    }
    result["report_content_sha256"] = canonical_digest(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "model_label",
                    "model_sha256",
                    "elapsed_seconds",
                    "maximum_resident_kib",
                    "final_ppl",
                    "final_uncertainty",
                    "report_content_sha256",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
