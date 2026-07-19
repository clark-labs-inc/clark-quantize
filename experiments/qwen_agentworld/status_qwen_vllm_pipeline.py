"""Summarize the Qwen vLLM cache -> smoke -> LR sweep pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def pid_status(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"pid": None, "alive": False, "elapsed_seconds": None, "cmd": None}
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etimes=,cmd="],
        check=False,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if not text:
        return {"pid": pid, "alive": False, "elapsed_seconds": None, "cmd": None}
    first, _, cmd = text.partition(" ")
    try:
        elapsed = int(first.strip())
    except ValueError:
        elapsed = None
    return {"pid": pid, "alive": True, "elapsed_seconds": elapsed, "cmd": cmd.strip()}


def parse_build_log(log_path: Path) -> dict[str, Any]:
    rows = 0
    tokens = 0
    partials = 0
    skips = 0
    tasks: Counter[str] = Counter()
    recent: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    if not log_path.exists():
        return {
            "rows": rows,
            "tokens": tokens,
            "partials": partials,
            "skips": skips,
            "tasks": {},
            "recent": [],
            "warnings": [],
            "errors": [],
        }
    for raw in log_path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        is_warning = "WARNING" in line
        if is_warning:
            warnings.append(line)
            warnings = warnings[-5:]
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
            tasks[source.split(":", 1)[0]] += 1
            recent.append(event)
            recent = recent[-5:]
    return {
        "rows": rows,
        "tokens": tokens,
        "partials": partials,
        "skips": skips,
        "tasks": dict(tasks),
        "recent": recent,
        "warnings": warnings,
        "errors": errors,
    }


def tail_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-limit:]


def latest_smoke_dir(run_dir: Path) -> str | None:
    logs = [run_dir / "handoff_latest.log"]
    logs.extend(sorted(run_dir.glob("handoff_*.log"), key=lambda path: path.stat().st_mtime, reverse=True))
    for log in logs:
        if not log.exists():
            continue
        for line in reversed(log.read_text(errors="replace").splitlines()):
            marker = "train_smoke_done out_dir="
            if marker in line:
                return line.split(marker, 1)[1].strip()
    return None


def latest_log_tail(pattern: str, *, limit: int = 8) -> list[str]:
    logs = sorted(Path("/").glob(pattern.lstrip("/")), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        return []
    return tail_lines(logs[0], limit=limit)


def report_summary(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return {"path": str(report_path), "error": "invalid_json"}
    history = report.get("history") or []
    last = history[-1] if history else None
    return {
        "path": str(report_path),
        "history_steps": len(history),
        "last": last,
        "early_stop": report.get("early_stop"),
        "skip_final_save": report.get("skip_final_save"),
    }


def latest_dirs(patterns: list[str], limit: int = 5) -> list[str]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path("/").glob(pattern.lstrip("/")))
    paths = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in paths[:limit]]


def gpu_memory() -> list[int] | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError:
            pass
    return values


def pgrep_lines(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [
        line
        for line in result.stdout.splitlines()
        if line.strip() and "status_qwen_vllm_pipeline.py" not in line
    ]


def lr_sweep_report_summaries(sweep_dirs: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        root = Path(sweep_dir)
        arms = sorted(path for path in root.glob("lr_*") if path.is_dir())
        if not arms:
            summaries.append({"sweep_dir": sweep_dir, "arms": []})
            continue
        summaries.append(
            {
                "sweep_dir": sweep_dir,
                "arms": [
                    {
                        "arm": arm.name,
                        "report": report_summary(arm / "report.json"),
                        "train_log_tail": tail_lines(arm / "train.log", limit=3),
                    }
                    for arm in arms
                ],
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019",
        type=Path,
    )
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    build_pid = read_pid(args.run_dir / "build.pid")
    handoff_pid = read_pid(args.run_dir / "handoff.pid")
    sweep_wait_pid = read_pid(args.run_dir / "post_smoke_lr_sweep.pid")
    corrected_sweep_wait_pid = read_pid(Path("/workspace/agentworld_runs/corrected_lr_sweep_waiter.pid"))
    build = parse_build_log(args.run_dir / "build.log")
    build_status = pid_status(build_pid)
    rows = int(build["rows"])
    elapsed = build_status.get("elapsed_seconds")
    rows_per_sec = rows / elapsed if elapsed and elapsed > 0 else None
    remaining_rows = max(args.max_rows - rows, 0)
    eta_seconds = remaining_rows / rows_per_sec if rows_per_sec and rows_per_sec > 0 else None
    final_files = sorted(str(path) for path in args.run_dir.glob("*.pt") if not str(path).endswith(".partial"))
    partial_files = sorted(str(path) for path in args.run_dir.glob("*.pt.partial"))
    smoke_dir = latest_smoke_dir(args.run_dir)
    smoke_report = report_summary(Path(smoke_dir) / "report.json") if smoke_dir else None
    sweep_dirs = latest_dirs(
        [
            "/workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_corrected_shuffle_*",
            "/workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_after_smoke_*",
        ],
        limit=5,
    )
    sweep_reports = lr_sweep_report_summaries(sweep_dirs)

    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "cache": {
                    "pid": build_status,
                    "rows": rows,
                    "max_rows": args.max_rows,
                    "progress": rows / args.max_rows if args.max_rows > 0 else None,
                    "tokens": build["tokens"],
                    "rows_per_sec": rows_per_sec,
                    "eta_seconds": eta_seconds,
                    "partials": build["partials"],
                    "skips": build["skips"],
                    "tasks": build["tasks"],
                    "recent": build["recent"],
                    "final_files": final_files,
                    "partial_files": partial_files,
                    "warnings": build["warnings"],
                    "errors": build["errors"],
                },
                "handoff": {
                    "pid": pid_status(handoff_pid),
                    "tail": tail_lines(args.run_dir / "handoff_latest.log", limit=8),
                },
                "post_smoke_lr_sweep": {
                    "pid": pid_status(sweep_wait_pid),
                    "tail": tail_lines(args.run_dir / "post_smoke_lr_sweep_latest.log", limit=8),
                },
                "corrected_lr_sweep_waiter": {
                    "pid": pid_status(corrected_sweep_wait_pid),
                    "tail": latest_log_tail("/workspace/agentworld_runs/corrected_lr_sweep_waiter_*.log", limit=8),
                },
                "smoke": {
                    "dir": smoke_dir,
                    "report": smoke_report,
                },
                "lr_sweep": {
                    "dirs": sweep_dirs,
                    "reports": sweep_reports,
                },
                "active_training_processes": pgrep_lines("train_all2d_ternary_qat.py"),
                "gpu_memory_used_mib": gpu_memory(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
