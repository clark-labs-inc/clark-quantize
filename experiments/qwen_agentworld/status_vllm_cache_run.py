"""Report progress for a vLLM teacher-cache build directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def pid_elapsed_seconds(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etimes="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[-1].strip())
    except ValueError:
        return None


def read_pid(run_dir: Path) -> int | None:
    path = run_dir / "build.pid"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def parse_log(log_path: Path) -> dict[str, Any]:
    rows = 0
    tokens = 0
    partials = 0
    skips = 0
    tasks: Counter[str] = Counter()
    recent: list[dict[str, Any]] = []
    nonfatal_warnings: list[str] = []
    errors: list[str] = []
    if not log_path.exists():
        return {
            "rows": rows,
            "tokens": tokens,
            "partials": partials,
            "skips": skips,
            "tasks": {},
            "recent": recent,
            "warnings": nonfatal_warnings,
            "errors": errors,
        }
    for raw in log_path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        is_warning = "WARNING" in line
        if is_warning:
            nonfatal_warnings.append(line)
            nonfatal_warnings = nonfatal_warnings[-5:]
        lowered = line.lower()
        if not is_warning and any(marker in lowered for marker in ("outofmemory", "traceback", "runtimeerror", "failed", "error")):
            errors.append(line)
            errors = errors[-10:]
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "partial_out" in event:
            partials += 1
            continue
        if "skip" in event:
            skips += 1
            continue
        if "index" in event and "tokens" in event:
            rows += 1
            tokens += int(event.get("tokens") or 0)
            source = str(event.get("source") or "unknown")
            task = source.split(":", 1)[0]
            tasks[task] += 1
            recent.append(event)
            recent = recent[-5:]
    return {
        "rows": rows,
        "tokens": tokens,
        "partials": partials,
        "skips": skips,
        "tasks": dict(tasks),
        "recent": recent,
        "warnings": nonfatal_warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    pid = read_pid(args.run_dir)
    elapsed = pid_elapsed_seconds(pid) if pid is not None else None
    log = parse_log(args.run_dir / "build.log")
    rows = int(log["rows"])
    tokens = int(log["tokens"])
    rows_per_sec = rows / elapsed if elapsed and elapsed > 0 else None
    tokens_per_sec = tokens / elapsed if elapsed and elapsed > 0 else None
    remaining_rows = max(args.max_rows - rows, 0)
    eta_seconds = remaining_rows / rows_per_sec if rows_per_sec and rows_per_sec > 0 else None
    final_files = sorted(str(path) for path in args.run_dir.glob("*.pt") if not str(path).endswith(".partial"))
    partial_files = sorted(str(path) for path in args.run_dir.glob("*.pt.partial"))

    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "pid": pid,
                "elapsed_seconds": elapsed,
                "rows": rows,
                "max_rows": args.max_rows,
                "progress": rows / args.max_rows if args.max_rows > 0 else None,
                "tokens": tokens,
                "rows_per_sec": rows_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "eta_seconds": eta_seconds,
                "partials": log["partials"],
                "skips": log["skips"],
                "tasks": log["tasks"],
                "recent": log["recent"],
                "final_files": final_files,
                "partial_files": partial_files,
                "warnings": log["warnings"],
                "errors": log["errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
