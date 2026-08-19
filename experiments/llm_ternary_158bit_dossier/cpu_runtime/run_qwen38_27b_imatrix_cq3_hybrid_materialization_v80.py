#!/usr/bin/env python3
"""Materialize the frozen imatrix-Q4 plus trained-CQ3 Qwen3.8-27B target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


LAYERS = (
    1, 3, 4, 5, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 19,
    20, 21, 23, 24, 25, 27, 28, 29,
    36, 37, 39, 40, 41, 43, 44, 45,
)
EXPECTED_SOURCE_BYTES = 15_606_631_072
EXPECTED_CANDIDATE_BYTES = 13_066_455_712
EXPECTED_SOURCE_SHA256 = "6176472b7d039a6d911b34eea78dd4507256b4a22b3003d40b8587b711014f2a"
EXPECTED_TRAINING_SHA256 = "1f33a04cf6d5433724e5d37d12fa0bfa3edfd710ab365033a8db69b517f73c1f"
EXPECTED_CONVERTER_SHA256 = "30c8e7234e1e3486077f238901df5f83accd49248e928b80586f93e84bb87b7d"
EXPECTED_TEST_CQ3_SHA256 = "2c1dfa7f1e1fca79d7fef2ff16de8315349b16a7346865c39a57a70288e01834"
EXPECTED_SCOPE_VERIFIER_SHA256 = "3a2a864ad8d93b28154c04bddf2410a2f22c6887c373abea08c84c78cd526cce"
EXPECTED_MATERIALIZER_SHA256 = "2001728d0463f30a1ac060aa254c008442fdb4395a5ba0a7e6487574b2d9a85a"
EXPECTED_V67_SUMMARY_SHA256 = "0f895cea20c3ed5fa4e7be39dd890b92f45113198d73aed8ea34683fe64452cd"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {observed}"
            )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def target_regex(layers: tuple[int, ...] = LAYERS) -> str:
    joined = "|".join(str(layer) for layer in layers)
    return rf"^blk\.({joined})\.ffn_(down|gate|up)\.weight$"


def expected_tensors(layers: tuple[int, ...] = LAYERS) -> set[str]:
    return {
        f"blk.{layer}.ffn_{role}.weight"
        for layer in layers
        for role in ("down", "gate", "up")
    }


def run(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with status {completed.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/workspace/src/clark-quantize-qwen38-v68"))
    parser.add_argument("--source", type=Path, default=Path("/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-imatrix-Q4_K-v10.gguf"))
    parser.add_argument("--candidate", type=Path, default=Path("/workspace/models/candidates/Qwen3.8-27B-imatrix-Q4-trained-CQ3-sensitivity32-v80.gguf"))
    parser.add_argument("--state", type=Path, default=Path("/workspace/quant-state/qwen38-imatrix-cq3-hybrid-v80"))
    parser.add_argument("--run-dir", type=Path, default=Path("/workspace/quant-runs/qwen38-imatrix-cq3-hybrid-v80"))
    parser.add_argument("--converter", type=Path, default=Path("/workspace/build/llama.cpp-qwen38-selective-v1/bin/llama-quantize"))
    parser.add_argument("--test-cq3", type=Path, default=Path("/workspace/build/llama.cpp-qwen38-selective-v1/bin/test-cq3"))
    parser.add_argument("--gguf-py", type=Path, default=Path("/workspace/src/llama.cpp-qwen38-selective-v1/gguf-py"))
    args = parser.parse_args()

    runtime = args.repo / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    prereg = runtime / "qwen38_27b_imatrix_cq3_hybrid_materialization_preregistration_v80.json"
    scope_verifier = runtime / "verify_gguf_selective_conversion.py"
    materializer = runtime / "materialize_qwen38_27b_sensitivity_selected_cq3_v67.py"
    training_receipt = runtime / "receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
    v67_summary = runtime / "receipts/qwen38_27b_sensitivity_selected_cq3_half_v67/summary.json"
    training_root = Path("/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44")
    partial = args.candidate.with_suffix(args.candidate.suffix + ".partial")
    scope = args.state / "gguf-scope.json"
    materialization = args.state / "materialization.json"
    summary = args.state / "summary.json"
    marker = args.state / "EXPERIMENT_COMPLETE.json"

    args.state.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for path, digest in (
        (args.source, EXPECTED_SOURCE_SHA256),
        (args.converter, EXPECTED_CONVERTER_SHA256),
        (args.test_cq3, EXPECTED_TEST_CQ3_SHA256),
        (scope_verifier, EXPECTED_SCOPE_VERIFIER_SHA256),
        (materializer, EXPECTED_MATERIALIZER_SHA256),
        (training_receipt, EXPECTED_TRAINING_SHA256),
        (v67_summary, EXPECTED_V67_SUMMARY_SHA256),
    ):
        require_file(path, digest)
    require_file(prereg)
    if args.source.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise RuntimeError("imatrix parent byte count changed")
    if not args.gguf_py.is_dir() or not training_root.is_dir():
        raise FileNotFoundError("pinned GGUF Python package or v44 training root is absent")

    if args.candidate.exists() and not materialization.is_file():
        raise RuntimeError("candidate exists without its materialization receipt")
    if not args.candidate.exists():
        if partial.exists():
            partial.unlink()
        available = shutil.disk_usage(args.candidate.parent).free
        required = EXPECTED_CANDIDATE_BYTES + 2 * 1024**3
        if available < required:
            raise RuntimeError(f"need {required} free bytes, observed {available}")
        run([str(args.test_cq3)], args.run_dir / "test-cq3.log")
        expression = target_regex()
        run(
            [
                str(args.converter),
                "--allow-requantize",
                "--tensor-type",
                f"{expression}=CQ3_0",
                str(args.source),
                str(partial),
                "COPY",
                "4",
            ],
            args.run_dir / "conversion.log",
        )
        if partial.stat().st_size != EXPECTED_CANDIDATE_BYTES:
            raise RuntimeError(
                f"candidate template bytes changed: {partial.stat().st_size}"
            )
        conversion_text = (args.run_dir / "conversion.log").read_text()
        observed = set(
            re.findall(
                r"(blk\.[0-9]+\.ffn_(?:down|gate|up)\.weight)\s+- applying selective COPY override",
                conversion_text,
            )
        )
        if observed != expected_tensors() or conversion_text.count(
            "applying selective COPY override"
        ) != 96:
            raise RuntimeError("selective conversion tensor denominator changed")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.gguf_py)
        run(
            [
                str(Path(os.sys.executable)),
                str(scope_verifier),
                "--baseline",
                str(args.source),
                "--candidate",
                str(partial),
                "--target-regex",
                expression,
                "--expected-target-count",
                "96",
                "--baseline-target-type",
                "12",
                "--candidate-target-type",
                "43",
                "--out",
                str(scope),
            ],
            args.run_dir / "scope.log",
            env=env,
        )
        env["PYTHONPATH"] = f"{args.gguf_py}:{runtime}"
        run(
            [
                str(Path(os.sys.executable)),
                str(materializer),
                "--template-partial",
                str(partial),
                "--template-scope-receipt",
                str(scope),
                "--training-receipt",
                str(training_receipt),
                "--artifact-root",
                str(training_root),
                "--layers",
                ",".join(str(layer) for layer in LAYERS),
                "--output",
                str(args.candidate),
                "--out",
                str(materialization),
                "--receipt-schema",
                "qwen38_27b_imatrix_cq3_hybrid_materialization_receipt_v80",
            ],
            args.run_dir / "materialization.log",
            env=env,
        )

    receipt = json.loads(materialization.read_text())
    checks = receipt.get("checks", {})
    if not receipt.get("valid") or not all(checks.values()):
        raise RuntimeError("materialization receipt is not valid")
    if receipt.get("selected_layers") != list(LAYERS) or receipt.get("selected_tensor_count") != 96:
        raise RuntimeError("materialization selected denominator changed")
    if receipt["candidate"]["bytes"] != EXPECTED_CANDIDATE_BYTES:
        raise RuntimeError("materialized candidate bytes changed")
    if args.candidate.stat().st_size != EXPECTED_CANDIDATE_BYTES:
        raise RuntimeError("candidate file bytes changed after publication")

    source_commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "schema": "qwen38_27b_imatrix_cq3_hybrid_materialization_summary_v80",
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": source_commit,
        "preregistration_sha256": sha256_file(prereg),
        "source": {
            "path": str(args.source),
            "bytes": EXPECTED_SOURCE_BYTES,
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        "candidate": receipt["candidate"],
        "selected_layers": list(LAYERS),
        "selected_tensor_count": 96,
        "storage": {
            "bytes_saved_vs_imatrix_q4": EXPECTED_SOURCE_BYTES - EXPECTED_CANDIDATE_BYTES,
            "fraction_saved_vs_imatrix_q4": (EXPECTED_SOURCE_BYTES - EXPECTED_CANDIDATE_BYTES) / EXPECTED_SOURCE_BYTES,
            "honest_complete_file_bits_per_text_parameter": EXPECTED_CANDIDATE_BYTES * 8 / 26_895_998_464,
        },
        "checks": checks,
        "decision": "admit_to_whole_model_quality_gate",
        "quality_ran": False,
        "performance_ran": False,
        "claim_boundary": "Exact complete artifact materialization only; quality, throughput, energy, novelty, and breakthrough remain unmeasured.",
        "complete": True,
        "terminal": True,
    }
    atomic_json(summary, payload)
    marker_payload = {
        "schema": "qwen38_27b_imatrix_cq3_hybrid_materialization_marker_v80",
        "summary_sha256": sha256_file(summary),
        "candidate_sha256": receipt["candidate"]["sha256"],
        "complete": True,
        "terminal": True,
    }
    atomic_json(marker, marker_payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
