#!/usr/bin/env python3
"""Run the frozen complete-model capability gate for the v80 hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATE_SHA256 = "09880f4d1b740f29b5e0cf0daf70ea649b7f5d2b57fb27df064fc1d03c9fab7d"
EXPECTED_CANDIDATE_BYTES = 13_066_455_712
EXPECTED_V80_SUMMARY_SHA256 = "329ab257cd2f8c6d7d57a750df7f2783c97bb79d0a817069cc960f7dda954aa3"
EXPECTED_V80_MATERIALIZATION_SHA256 = "7f8074b3291145c3fe5917cd82a867ed1a37b3d2cf9fbebd429eaa0a227e1261"
EXPECTED_SERVER_SHA256 = "58d611c2a2bb38fdb8d32121d4b7a54180ad89233020037a11cdd67b6774177d"
EXPECTED_PERPLEXITY_SHA256 = "be9149ccdc77343087fff7de7d2370f3285cb0082ac23c20b458445053101c34"
EXPECTED_CUTE_EVALUATOR_SHA256 = "9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab"
EXPECTED_MATH_CODE_EVALUATOR_SHA256 = "fa3d0f870b3b0ed1d7d907cc8d146854b164e6d75cfa9d2bd20743979af74677"
EXPECTED_WIKITEXT_EVALUATOR_SHA256 = "54458dc377a1d07aaf26df90823f65f973b625f55a6ac73895d0c6ec9a666b07"
EXPECTED_MMLU_EVALUATOR_SHA256 = "a604ca9a528231fa17de78ab2a7ffed7c1b9a0b9a9ca74459c2d894f1f3612d3"
EXPECTED_CUTE_MANIFEST_SHA256 = "19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16"
EXPECTED_MATH_CODE_MANIFEST_SHA256 = "1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0"
EXPECTED_MMLU_MANIFEST_SHA256 = "e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba"
EXPECTED_WIKI_CORPUS_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
Q4_WIKITEXT_PPL = 5.7847


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


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


def wait_for_health(port: int, process: subprocess.Popen[Any], timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited before health with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def start_server(
    server: Path,
    candidate: Path,
    build: Path,
    log: Path,
    port: int,
) -> tuple[subprocess.Popen[Any], Any]:
    stream = log.open("a")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(build / "bin")
    command = [
        "/usr/bin/taskset", "-c", "0,2,4,6", str(server),
        "--model", str(candidate), "--threads", "4", "--threads-batch", "4",
        "--ctx-size", "8192", "--parallel", "1", "--batch-size", "512",
        "--ubatch-size", "64", "--host", "127.0.0.1", "--port", str(port),
        "--no-warmup", "--log-disable",
    ]
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    try:
        wait_for_health(port, process)
    except BaseException:
        stop_server(process, stream)
        raise
    return process, stream


def stop_server(process: subprocess.Popen[Any], stream: Any) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    stream.close()


def summary_correct(payload: dict[str, Any], task: str) -> int:
    for row in payload["summaries"]:
        if row["task"] == task:
            return int(row["correct"])
    raise ValueError(f"missing task summary: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/workspace/src/clark-quantize-qwen38-v68"))
    parser.add_argument("--candidate", type=Path, default=Path("/workspace/models/candidates/Qwen3.8-27B-imatrix-Q4-trained-CQ3-sensitivity32-v80.gguf"))
    parser.add_argument("--build", type=Path, default=Path("/workspace/build/llama.cpp-qwen38-cq3-aff3-v63"))
    parser.add_argument("--wiki-corpus", type=Path, default=Path("/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw"))
    parser.add_argument("--state", type=Path, default=Path("/workspace/quant-state/qwen38-imatrix-cq3-hybrid-quality-v81"))
    parser.add_argument("--run-dir", type=Path, default=Path("/workspace/quant-runs/qwen38-imatrix-cq3-hybrid-quality-v81"))
    parser.add_argument("--port", type=int, default=8111)
    args = parser.parse_args()

    runtime = args.repo / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    receipts = runtime / "receipts/qwen38_27b_imatrix_cq3_hybrid_materialization_v80"
    paths = {
        "v80_summary": receipts / "summary.json",
        "v80_materialization": receipts / "materialization.json",
        "server": args.build / "bin/llama-server",
        "perplexity": args.build / "bin/llama-perplexity",
        "cute_eval": runtime / "eval_llama_cpp_cute_quick.py",
        "math_eval": runtime / "eval_llama_cpp_math_code_canary.py",
        "wiki_eval": runtime / "eval_llama_cpp_wikitext_quick.py",
        "mmlu_eval": runtime / "eval_llama_cpp_mmlu.py",
        "cute_manifest": runtime / "cute_quick_v1.json",
        "math_manifest": runtime / "math_code_canary_v1.json",
        "mmlu_manifest": runtime / "qwen35_4b_cpu_layer30_mmlu_v1.json",
        "prereg": runtime / "qwen38_27b_imatrix_cq3_hybrid_quality_preregistration_v81.json",
    }
    expected = {
        "v80_summary": EXPECTED_V80_SUMMARY_SHA256,
        "v80_materialization": EXPECTED_V80_MATERIALIZATION_SHA256,
        "server": EXPECTED_SERVER_SHA256,
        "perplexity": EXPECTED_PERPLEXITY_SHA256,
        "cute_eval": EXPECTED_CUTE_EVALUATOR_SHA256,
        "math_eval": EXPECTED_MATH_CODE_EVALUATOR_SHA256,
        "wiki_eval": EXPECTED_WIKITEXT_EVALUATOR_SHA256,
        "mmlu_eval": EXPECTED_MMLU_EVALUATOR_SHA256,
        "cute_manifest": EXPECTED_CUTE_MANIFEST_SHA256,
        "math_manifest": EXPECTED_MATH_CODE_MANIFEST_SHA256,
        "mmlu_manifest": EXPECTED_MMLU_MANIFEST_SHA256,
    }
    args.state.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for key, digest in expected.items():
        require_sha256(paths[key], digest)
    require_sha256(args.wiki_corpus, EXPECTED_WIKI_CORPUS_SHA256)
    require_sha256(args.candidate, EXPECTED_CANDIDATE_SHA256)
    if args.candidate.stat().st_size != EXPECTED_CANDIDATE_BYTES:
        raise RuntimeError("candidate byte count changed")

    candidate_sha = EXPECTED_CANDIDATE_SHA256
    cute_path = args.state / "cute.json"
    math_path = args.state / "math-code.json"
    wiki_path = args.state / "wikitext.json"
    quick_path = args.state / "quick-quality-summary.json"
    mmlu_path = args.state / "mmlu.json"
    final_path = args.state / "summary.json"
    marker_path = args.state / "EXPERIMENT_COMPLETE.json"
    if marker_path.is_file() and final_path.is_file():
        print(final_path.read_text(), end="")
        return

    process: subprocess.Popen[Any] | None = None
    stream: Any = None
    try:
        if not cute_path.is_file() or not math_path.is_file():
            process, stream = start_server(
                paths["server"], args.candidate, args.build, args.run_dir / "server-quick.log", args.port
            )
            if not cute_path.is_file():
                run(
                    [os.sys.executable, str(paths["cute_eval"]), "--manifest", str(paths["cute_manifest"]),
                     "--server", f"http://127.0.0.1:{args.port}", "--model-label", "qwen38-v80-imatrix-cq3",
                     "--model-sha256", candidate_sha, "--out", str(cute_path)],
                    args.run_dir / "cute.log",
                )
            if not math_path.is_file():
                run(
                    [os.sys.executable, str(paths["math_eval"]), "--manifest", str(paths["math_manifest"]),
                     "--server", f"http://127.0.0.1:{args.port}", "--model-label", "qwen38-v80-imatrix-cq3",
                     "--model-sha256", candidate_sha, "--out", str(math_path), "--timeout", "300"],
                    args.run_dir / "math-code.log",
                )
    finally:
        if process is not None and stream is not None:
            stop_server(process, stream)

    if not wiki_path.is_file():
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(args.build / "bin")
        run(
            [os.sys.executable, str(paths["wiki_eval"]), "--model", str(args.candidate),
             "--model-label", "qwen38-v80-imatrix-cq3", "--corpus", str(args.wiki_corpus),
             "--llama-perplexity", str(paths["perplexity"]), "--chunks", "4", "--context", "512",
             "--threads", "4", "--out", str(wiki_path)],
            args.run_dir / "wikitext.log",
            env=env,
        )

    cute = json.loads(cute_path.read_text())
    math_code = json.loads(math_path.read_text())
    wiki = json.loads(wiki_path.read_text())
    cute_correct = summary_correct(cute, "cute_quick_20")
    math_correct = summary_correct(math_code, "math500")
    code_correct = summary_correct(math_code, "code_trace")
    if len(cute.get("records", [])) != 20 or len(math_code.get("records", [])) != 14:
        raise RuntimeError("free-generation denominator changed")
    if not wiki.get("valid") or len(wiki.get("chunks", [])) != 4:
        raise RuntimeError("WikiText receipt is not valid")
    wiki_ratio = float(wiki["final_ppl"]) / Q4_WIKITEXT_PPL
    quick_checks = {
        "cute_at_least_8_of_20": cute_correct >= 8,
        "math500_at_least_4_of_8": math_correct >= 4,
        "code_at_least_4_of_6": code_correct >= 4,
        "wikitext_ratio_at_most_1_05": wiki_ratio <= 1.05,
    }
    quick = {
        "schema": "qwen38_27b_imatrix_cq3_hybrid_quick_quality_v81",
        "candidate_sha256": candidate_sha,
        "observed": {
            "cute_correct_of_20": cute_correct,
            "math500_correct_of_8": math_correct,
            "code_correct_of_6": code_correct,
            "wikitext_ppl": wiki["final_ppl"],
            "wikitext_ppl_ratio_vs_q4": wiki_ratio,
        },
        "checks": quick_checks,
        "passed": all(quick_checks.values()),
        "complete": True,
        "terminal": True,
    }
    atomic_json(quick_path, quick)

    mmlu_correct: int | None = None
    if quick["passed"]:
        process = None
        stream = None
        try:
            if not mmlu_path.is_file():
                process, stream = start_server(
                    paths["server"], args.candidate, args.build, args.run_dir / "server-mmlu.log", args.port
                )
                command = [
                    os.sys.executable, str(paths["mmlu_eval"]), "--manifest", str(paths["mmlu_manifest"]),
                    "--model", str(args.candidate), "--model-label", "qwen38-v80-imatrix-cq3",
                    "--model-sha256", candidate_sha, "--server", f"http://127.0.0.1:{args.port}",
                    "--workers", "1", "--timeout", "900", "--checkpoint-every", "4", "--out", str(mmlu_path),
                ]
                if mmlu_path.with_suffix(mmlu_path.suffix + ".partial").is_file():
                    command.append("--resume")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(args.repo)
                run(command, args.run_dir / "mmlu.log", env=env)
        finally:
            if process is not None and stream is not None:
                stop_server(process, stream)
        mmlu = json.loads(mmlu_path.read_text())
        mmlu_correct = int(mmlu["summary"]["correct"])

    mmlu_passed = mmlu_correct is not None and mmlu_correct >= 50
    passed = bool(quick["passed"] and mmlu_passed)
    result = {
        "schema": "qwen38_27b_imatrix_cq3_hybrid_quality_result_v81",
        "model": "Qwen/Qwen3.8-27B",
        "candidate": {"path": str(args.candidate), "bytes": EXPECTED_CANDIDATE_BYTES, "sha256": candidate_sha},
        "preregistration_sha256": sha256_file(paths["prereg"]),
        "quick_quality": quick,
        "mmlu": {"ran": mmlu_correct is not None, "correct_of_64": mmlu_correct, "passed": mmlu_passed},
        "passed": passed,
        "decision": "admit_to_combined_imatrix_cq3_clang_mtp3_runtime" if passed else "reject_composition_before_kernel_integration",
        "performance_ran": False,
        "cute_is_adaptive_lineage_evidence": True,
        "claim_boundary": "Complete-model quality result only; no throughput, energy, novelty, or scientific-breakthrough claim.",
        "complete": True,
        "terminal": True,
    }
    atomic_json(final_path, result)
    atomic_json(
        marker_path,
        {
            "schema": "qwen38_27b_imatrix_cq3_hybrid_quality_marker_v81",
            "summary_sha256": sha256_file(final_path),
            "candidate_sha256": candidate_sha,
            "complete": True,
            "terminal": True,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
