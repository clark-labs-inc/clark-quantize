#!/usr/bin/env python3
"""Run the frozen ggml compact-residual graph gate for Qwen3.8-27B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def checked_git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def verify_frozen_inputs(repo: Path, manifest: dict[str, Any]) -> None:
    failures: list[str] = []
    for name, descriptor in manifest["frozen_inputs"].items():
        if descriptor.get("external"):
            continue
        path = repo / descriptor["path"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        observed = sha256_path(path)
        if observed != descriptor["sha256"]:
            failures.append(
                f"{name}: sha256 {observed} != {descriptor['sha256']}"
            )
    if failures:
        raise RuntimeError("frozen input verification failed: " + "; ".join(failures))


def block_summary(blocks: list[list[float]]) -> dict[str, Any]:
    if not blocks or any(not block for block in blocks):
        raise RuntimeError("timing block is empty")
    medians = [statistics.median(block) for block in blocks]
    center = statistics.median(medians)
    if center <= 0:
        raise RuntimeError("timing median is nonpositive")
    return {
        "raw_ns_per_batch_blocks": blocks,
        "block_medians_ns_per_batch": medians,
        "median_of_block_medians_ns_per_batch": center,
        "block_median_range_fraction": (max(medians) - min(medians)) / center,
    }


def run_case(
    binary: Path,
    payload: Path,
    *,
    layer: int,
    tokens: int,
    threads: int,
    seed: int,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    command = [
        str(binary),
        "--payload",
        str(payload),
        "--tokens",
        str(tokens),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
        "--blocks",
        str(measurement["blocks_per_arm_per_case"]),
        "--warmups",
        str(measurement["warmups_per_arm_per_block"]),
        "--samples",
        str(measurement["samples_per_arm_per_block"]),
        "--inner",
        str(measurement["inner_batch_calls_per_sample"]),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("ggml harness did not emit one JSON object") from error
    if (
        raw.get("schema") != "qwen38_calibration_residual_ggml_case_v40"
        or raw.get("complete") is not True
        or raw.get("tokens") != tokens
        or raw.get("threads") != threads
        or raw.get("blocks") != measurement["blocks_per_arm_per_case"]
        or raw.get("samples_per_block")
        != measurement["samples_per_arm_per_block"]
        or raw.get("inner_calls_per_sample")
        != measurement["inner_batch_calls_per_sample"]
    ):
        raise RuntimeError("ggml harness case metadata changed")
    direct = block_summary(raw.pop("direct_ns_per_batch_blocks"))
    ggml = block_summary(raw.pop("ggml_ns_per_batch_blocks"))
    ggml_ns = ggml["median_of_block_medians_ns_per_batch"]
    direct_ns = direct["median_of_block_medians_ns_per_batch"]
    return {
        "layer": layer,
        "tokens": tokens,
        "threads": threads,
        "seed": seed,
        "payload_path": str(payload),
        "payload_sha256": sha256_path(payload),
        "payload_bytes": payload.stat().st_size,
        "maximum_absolute_error_vs_direct": raw[
            "maximum_absolute_error_vs_direct"
        ],
        "rank_scratch_bytes_per_task": raw["rank_scratch_bytes_per_task"],
        "direct": direct,
        "ggml": ggml,
        "ggml_overhead_ratio_vs_direct": ggml_ns / direct_ns,
        "ggml_ns_per_token": ggml_ns / tokens,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != (
        "qwen38_27b_calibration_residual_ggml_preregistration_v40"
    ):
        raise RuntimeError("unexpected v40 preregistration schema")
    verify_frozen_inputs(repo, manifest)

    llama_revision = checked_git(args.llama_source, "rev-parse", "HEAD")
    llama_status = checked_git(args.llama_source, "status", "--porcelain")
    if llama_revision != manifest["llama_cpp"]["revision"]:
        raise RuntimeError("llama.cpp revision changed")
    if llama_status:
        raise RuntimeError("llama.cpp source is dirty")

    external = manifest["frozen_external_binaries"]
    binaries = {
        "harness": args.binary,
        "ggml_cpu": args.llama_build / "bin/libggml-cpu.so.0.20.0",
        "ggml_base": args.llama_build / "bin/libggml-base.so.0.20.0",
    }
    for name, path in binaries.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen external binary: {path}")
        observed = sha256_path(path)
        if observed != external[name]["sha256"]:
            raise RuntimeError(
                f"{name} sha256 {observed} != {external[name]['sha256']}"
            )

    measurement = manifest["measurement"]
    cases: list[dict[str, Any]] = []
    for payload_index, descriptor in enumerate(manifest["payloads"]):
        payload = repo / descriptor["path"]
        if (
            sha256_path(payload) != descriptor["sha256"]
            or payload.stat().st_size != descriptor["bytes"]
        ):
            raise RuntimeError("trained payload identity changed")
        for token_index, tokens in enumerate(measurement["token_counts"]):
            seed = (
                int(measurement["seed"])
                + 1009 * payload_index
                + 9176 * token_index
            )
            cases.append(
                run_case(
                    args.binary,
                    payload,
                    layer=descriptor["layer"],
                    tokens=tokens,
                    threads=measurement["threads"],
                    seed=seed,
                    measurement=measurement,
                )
            )

    gate = manifest["gate"]
    decode_cases = [case for case in cases if case["tokens"] == 1]
    microbatch_cases = [case for case in cases if case["tokens"] == 8]
    microbatch_ratios = []
    for decode in decode_cases:
        microbatch = next(
            case for case in microbatch_cases if case["layer"] == decode["layer"]
        )
        microbatch_ratios.append(
            {
                "layer": decode["layer"],
                "eight_token_per_token_vs_decode_ratio": (
                    microbatch["ggml_ns_per_token"] / decode["ggml_ns_per_token"]
                ),
            }
        )

    checks = {
        "required_exact_payloads": len({case["layer"] for case in cases})
        == gate["required_exact_payloads"],
        "required_token_counts": sorted({case["tokens"] for case in cases})
        == gate["required_token_counts"],
        "maximum_absolute_error_vs_direct": max(
            case["maximum_absolute_error_vs_direct"] for case in cases
        )
        <= gate["maximum_absolute_error_vs_direct"],
        "required_payload_bytes": all(
            case["payload_bytes"] == gate["required_payload_bytes_each"]
            for case in cases
        ),
        "required_rank_scratch_bytes": all(
            case["rank_scratch_bytes_per_task"]
            == gate["required_rank_scratch_bytes_per_task"]
            for case in cases
        ),
        "maximum_decode_graph_overhead_ratio": max(
            case["ggml_overhead_ratio_vs_direct"] for case in decode_cases
        )
        <= gate["maximum_decode_graph_overhead_ratio_vs_direct"],
        "maximum_projected_all_64_layers_decode_ns": max(
            case["ggml_ns_per_token"] * 64 for case in decode_cases
        )
        <= gate["maximum_projected_all_64_layers_decode_ns"],
        "maximum_graph_block_median_range_fraction": max(
            case["ggml"]["block_median_range_fraction"] for case in cases
        )
        <= gate["maximum_graph_block_median_range_fraction"],
        "maximum_eight_token_per_token_vs_decode_ratio": max(
            row["eight_token_per_token_vs_decode_ratio"]
            for row in microbatch_ratios
        )
        <= gate["maximum_eight_token_per_token_vs_decode_ratio"],
    }
    passed = all(checks.values())
    receipt: dict[str, Any] = {
        "schema": "qwen38_27b_calibration_residual_ggml_receipt_v40",
        "experiment_id": manifest["experiment_id"],
        "complete": True,
        "terminal": True,
        "passed": passed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "repo_revision": checked_git(repo, "rev-parse", "HEAD"),
        "llama_cpp": {
            "revision": llama_revision,
            "source_clean": not bool(llama_status),
            "source_path": str(args.llama_source),
        },
        "external_binary_sha256": {
            name: sha256_path(path) for name, path in binaries.items()
        },
        "measurement": measurement,
        "cases": cases,
        "microbatch_ratios": microbatch_ratios,
        "gate": {"thresholds": gate, "checks": checks, "passed": passed},
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_json(args.receipt, receipt)
    decision = {
        "schema": "qwen38_27b_calibration_residual_ggml_terminal_decision_v40",
        "experiment_id": manifest["experiment_id"],
        "terminal": True,
        "passed": passed,
        "decision": (
            "admit_exact_compact_node_to_llama_adapter_loader_integration_only"
            if passed
            else "reject_v40_ggml_graph_integration"
        ),
        "receipt_path": str(args.receipt),
        "receipt_sha256": sha256_path(args.receipt),
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_json(args.decision, decision)
    atomic_text(
        args.completion_marker,
        json.dumps(
            {
                "terminal": True,
                "passed": passed,
                "receipt_sha256": sha256_path(args.receipt),
                "decision_sha256": sha256_path(args.decision),
            },
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"event": "terminal", "passed": passed}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
