#!/usr/bin/env python3
"""Run the sealed MPS capacity search using one process per batch size."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


FORMAT = "backdropkit-swin2sr-batch-capacity-decision-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted(receipt: dict | None) -> bool:
    return bool(receipt and receipt.get("status") == "accepted")


def refinement_candidate(lower: int, upper: int, quantum: int) -> int | None:
    lower_units = lower // quantum
    upper_units = upper // quantum
    if upper_units - lower_units <= 1:
        return None
    return ((lower_units + upper_units) // 2) * quantum


def verify_protocol(protocol: dict, root: Path) -> None:
    if protocol.get("status") != "sealed_before_capacity_outputs":
        raise ValueError("batch-capacity protocol is not sealed")
    for relative, expected in protocol["source_sha256"].items():
        observed = file_sha256(root / relative)
        if observed != expected:
            raise ValueError(f"sealed source changed: {relative}")
    for path, expected in protocol["input_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"sealed input changed: {path}")


def command_for(protocol: dict, root: Path, batch_size: int, receipt: Path) -> list[str]:
    config = protocol["probe_config"]
    command = [
        protocol["python"],
        str(root / "probe_batch.py"),
        "--parent", protocol["parent_checkpoint"],
        "--out", str(receipt),
        "--island", config["island"],
        "--precision", config["precision"],
        "--batch-size", str(batch_size),
        "--seed", str(config["seed"]),
        "--warmup-steps", str(config["warmup_steps"]),
        "--measure-steps", str(config["measure_steps"]),
        "--learning-rate", str(config["learning_rate"]),
        "--memory-fraction", str(config["memory_fraction"]),
        "--headroom-fraction", str(config["headroom_fraction"]),
        "--device", "mps",
    ]
    for data_root in protocol["training_data"]:
        command.extend(("--data", data_root))
    return command


def run_probe(protocol: dict, root: Path, batch_size: int) -> dict | None:
    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = output_root / f"batch-{batch_size}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    command = command_for(protocol, root, batch_size, receipt)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = protocol["pythonpath"]
    completed = subprocess.run(
        command,
        cwd=protocol["repository_root"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_root / f"batch-{batch_size}.log").write_text(completed.stdout)
    if completed.returncode != 0 or not receipt.exists():
        failure = {
            "format": "backdropkit-swin2sr-batch-capacity-failure-v1",
            "status": "failed",
            "batch_size": batch_size,
            "returncode": completed.returncode,
            "log": str(output_root / f"batch-{batch_size}.log"),
            "command": command,
        }
        (output_root / f"batch-{batch_size}-failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        return None
    return json.loads(receipt.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    root = args.protocol.resolve().parent
    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol, root)
    search = protocol["search"]
    quantum = int(search["quantum"])
    lower = None
    upper = None
    tested = {}
    started = time.time()

    batch_size = int(search["start"])
    while batch_size <= int(search["maximum"]):
        receipt = run_probe(protocol, root, batch_size)
        tested[str(batch_size)] = receipt
        if accepted(receipt):
            lower = batch_size
            batch_size *= int(search["geometric_multiplier"])
        else:
            upper = batch_size
            break

    if lower is None:
        raise RuntimeError("even the sealed baseline batch was not accepted")
    if upper is not None:
        while True:
            candidate = refinement_candidate(lower, upper, quantum)
            if candidate is None:
                break
            receipt = run_probe(protocol, root, candidate)
            tested[str(candidate)] = receipt
            if accepted(receipt):
                lower = candidate
            else:
                upper = candidate

    accepted_receipts = {
        size: receipt for size, receipt in tested.items() if accepted(receipt)
    }
    decision = {
        "format": FORMAT,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": file_sha256(args.protocol),
        "status": "capacity_capped" if upper is None else "capacity_bounded",
        "largest_stable_physical_batch": lower,
        "first_rejected_or_failed_batch": upper,
        "search_quantum": quantum,
        "selection_rule": protocol["selection_rule"],
        "accepted_receipts": {
            size: {
                "sha256": file_sha256(root / "outputs" / f"batch-{size}.json"),
                "compute_images_per_second": receipt["throughput"][
                    "compute_images_per_second"
                ],
                "driver_fraction": receipt["acceptance"][
                    "observed_driver_fraction_of_recommended"
                ],
            }
            for size, receipt in sorted(
                accepted_receipts.items(), key=lambda item: int(item[0])
            )
        },
        "tested_batches": sorted(int(size) for size in tested),
        "elapsed_seconds": time.time() - started,
        "next_gate": (
            "Matched equal-images batch and learning-rate study; this capacity "
            "screen alone does not authorize changing the training default."
        ),
    }
    decision_path = root / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
