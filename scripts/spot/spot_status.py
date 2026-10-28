#!/usr/bin/env python3
"""Write a small, atomic health receipt for an interruptible GPU node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def optional_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/status.json"))
    parser.add_argument("--event", default="heartbeat")
    parser.add_argument("--job")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()

    workspace = args.workspace if args.workspace.exists() else Path.cwd()
    disk = shutil.disk_usage(workspace)
    gpu_line = command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    process_alive = None
    if args.pid is not None:
        try:
            os.kill(args.pid, 0)
            process_alive = True
        except OSError:
            process_alive = False

    repo = Path(os.environ.get("QUANT_REPO", "/workspace/clark-quantize"))
    receipt = {
        "schema": "clark_quant_spot_status_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": args.event,
        "hostname": command("hostname"),
        "boot_id": optional_text(Path("/proc/sys/kernel/random/boot_id")),
        "job": args.job,
        "pid": args.pid,
        "process_alive": process_alive,
        "exit_code": args.exit_code,
        "git_head": command("git", "-C", str(repo), "rev-parse", "HEAD"),
        "gpu_csv": gpu_line,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.out.parent, delete=False) as handle:
        json.dump(receipt, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, args.out)

    history = args.out.with_name("events.jsonl")
    if args.event != "heartbeat":
        with history.open("a") as handle:
            json.dump(receipt, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    directory = os.open(args.out.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


if __name__ == "__main__":
    main()
