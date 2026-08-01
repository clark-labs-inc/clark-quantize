#!/usr/bin/env python3
"""Export a trained realtime student into the exact fused WGSL payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from realtime_student import (
    RealtimeLumaStudent,
    RealtimeStudentConfig,
    export_wgsl_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "backdropkit-realtime-student-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    student_config = payload["manifest"]["student"]
    model = RealtimeLumaStudent(RealtimeStudentConfig(
        channels=int(student_config["channels"]),
        blocks=int(student_config["blocks"]),
        color_mode=student_config.get("color_mode", "luma"),
    ))
    model.load_state_dict(payload["model"], strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.out / "model.bin"
    manifest = export_wgsl_weights(model, binary)
    manifest["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest["source_checkpoint"] = str(args.checkpoint.resolve())
    (args.out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
