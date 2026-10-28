#!/usr/bin/env python3
"""Restartable MATH-500 quick-8 evaluation for the Qwen3.8-27B CQ3 runtime."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from transformers import __version__ as transformers_version

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_aligned2_cache import Qwen38Aligned2RuntimeImage
from clark_quantize.ternary_cpu import cpu_features, native_available
from eval_llama_cpp_cute_quick import canonical_digest
from eval_llama_cpp_math_code_canary import extract_last_boxed, normalize_math
from eval_qwen38_27b_aligned2_cute_v1 import (
    generate_one,
    peak_rss_bytes,
    physical_cpu_ids,
    sha256_file,
    verified_tokenizer,
)


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, record in preregistration["frozen_inputs"].items():
        path = REPO / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen Qwen3.8 MATH input changed: {label}")


def load_progress(
    path: Path,
    *,
    model_sha256: str,
    manifest_sha256: str,
    expected_ids: list[str],
) -> list[dict]:
    if not path.is_file():
        return []
    progress = json.loads(path.read_text())
    records = progress.get("records")
    if (
        progress.get("schema") != "qwen38_27b_aligned2_math500_progress_v1"
        or progress.get("model_sha256") != model_sha256
        or progress.get("manifest_content_sha256") != manifest_sha256
        or not isinstance(records, list)
        or [record.get("id") for record in records]
        != expected_ids[: len(records)]
    ):
        raise RuntimeError("Qwen3.8 MATH resume checkpoint identity changed")
    return records


def save_progress(
    path: Path,
    *,
    model_sha256: str,
    manifest_sha256: str,
    records: list[dict],
) -> None:
    progress = {
        "schema": "qwen38_27b_aligned2_math500_progress_v1",
        "updated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "model_sha256": model_sha256,
        "manifest_content_sha256": manifest_sha256,
        "records": records,
    }
    atomic_bytes(path, canonical_json(progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-cq3-row-normalized-container-v1/"
            "Qwen3.8-27B-row-normalized.cq3"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/workspace/quant-runs/qwen38-aligned2-cache-v1/"
            "Qwen3.8-27B-row-normalized.a2.manifest.json"
        ),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-cute-v1.json"
        ),
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-math500-v1.progress.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-aligned2-math500-v1.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_aligned2_math500_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    native_library = os.environ.get("CLARK_TERNARY_CPU_LIB")
    if not native_available() or not native_library or not Path(native_library).is_file():
        raise SystemExit("infrastructure-invalid: pinned native library is absent")
    features = cpu_features()
    if not all(features[name] for name in ("avx2", "f16c", "fma")):
        raise SystemExit("infrastructure-invalid: AVX2/F16C/FMA are required")
    threads = int(prereg["measurement"]["physical_threads"])
    cpu_ids = physical_cpu_ids()
    if len(cpu_ids) < threads:
        raise SystemExit(
            f"infrastructure-invalid: only {len(cpu_ids)} physical CPUs are available"
        )
    cpu_ids = cpu_ids[:threads]

    parent_content = args.parent_receipt.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema") != "qwen38_27b_aligned2_cute_receipt_v1"
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("CUTE parent gate did not pass")
    source = parent["artifacts"]["source_container"]
    cache = parent["artifacts"]["cache"]
    if (
        args.container.resolve() != Path(source["path"]).resolve()
        or args.container.stat().st_size != int(source["bytes"])
        or args.cache.resolve() != Path(cache["path"]).resolve()
        or args.cache.stat().st_size != int(cache["bytes"])
        or args.manifest.resolve() != Path(cache["manifest_path"]).resolve()
        or sha256_file(args.manifest) != cache["manifest_sha256"]
    ):
        raise RuntimeError("MATH artifacts differ from the CUTE parent")

    manifest_path = REPO / prereg["evaluation"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest["content_sha256"]
    manifest_body = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if canonical_digest(manifest_body) != claimed:
        raise RuntimeError("MATH manifest content hash changed")
    items = manifest["math"]["items"]
    if len(items) != 8:
        raise RuntimeError("MATH quick denominator changed")

    q4_report_path = REPO / prereg["control"]["q4_report_path"]
    q4_report = json.loads(q4_report_path.read_text())
    q4_summary = next(
        (
            value
            for value in q4_report.get("summaries", [])
            if value.get("task") == "math500"
        ),
        None,
    )
    if (
        q4_report.get("schema") != "cpu_math_code_canary_result_v1"
        or q4_report.get("manifest_content_sha256") != claimed
        or q4_report.get("model_sha256")
        != prereg["control"]["q4_model_sha256"]
        or not isinstance(q4_summary, dict)
        or q4_summary.get("correct") != prereg["control"]["q4_correct"]
        or q4_summary.get("n") != prereg["control"]["q4_denominator"]
    ):
        raise RuntimeError("frozen Q4 MATH control changed")

    expected_ids = [item["id"] for item in items]
    records = load_progress(
        args.progress,
        model_sha256=source["sha256"],
        manifest_sha256=claimed,
        expected_ids=expected_ids,
    )
    preexisting_records = len(records)
    max_tokens = int(manifest["prompt_contract"]["max_tokens"])
    instruction = manifest["prompt_contract"]["math"]
    started = datetime.now(timezone.utc)

    with Qwen38Aligned2RuntimeImage(
        args.container,
        args.cache,
        args.manifest,
        verify_source_container=False,
        verify_full_cache=False,
    ) as image:
        model = image.text_model()
        tokenizer, tokenizer_files = verified_tokenizer(image, args.model_dir)
        generation_config = json.loads(
            image.auxiliary_file("generation_config.json").tobytes()
        )
        raw_eos = generation_config.get("eos_token_id", tokenizer.eos_token_id)
        eos_token_ids = {
            int(value) for value in (raw_eos if isinstance(raw_eos, list) else [raw_eos])
        }
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for item in items[len(records) :]:
                generated = generate_one(
                    model,
                    tokenizer,
                    item["problem"] + "\n" + instruction,
                    max_tokens=max_tokens,
                    threads=threads,
                    executor=executor,
                    cpu_ids=cpu_ids,
                    row_chunk=int(prereg["candidate"]["streaming_row_chunk"]),
                    eos_token_ids=eos_token_ids,
                )
                prediction = normalize_math(
                    extract_last_boxed(generated["output"])
                )
                gold = normalize_math(item["answer"])
                record = {
                    "task": "math500",
                    "id": item["id"],
                    "correct": prediction == gold,
                    "prediction": prediction,
                    "gold": gold,
                    **generated,
                }
                records.append(record)
                save_progress(
                    args.progress,
                    model_sha256=source["sha256"],
                    manifest_sha256=claimed,
                    records=records,
                )
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "task",
                                "id",
                                "correct",
                                "prediction",
                                "gold",
                                "observed_tokens_per_second",
                            )
                        }
                    ),
                    flush=True,
                )
        verified_tensors = len(image._verified_tensors)

    if len(records) != 8:
        raise RuntimeError("MATH evaluation did not reach its frozen denominator")
    total_correct = sum(bool(record["correct"]) for record in records)
    total_forward_tokens = sum(int(record["forward_tokens"]) for record in records)
    total_model_seconds = sum(float(record["elapsed_seconds"]) for record in records)
    finite = bool(
        total_forward_tokens > 0
        and math.isfinite(total_model_seconds)
        and total_model_seconds > 0.0
        and all(
            math.isfinite(float(record["observed_tokens_per_second"]))
            and float(record["observed_tokens_per_second"]) > 0.0
            for record in records
        )
    )
    peak_rss = peak_rss_bytes()
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_correct": total_correct,
        "observed_denominator": len(records),
        "observed_q4_correct": int(prereg["control"]["q4_correct"]),
        "observed_correct_loss_vs_q4": (
            int(prereg["control"]["q4_correct"]) - total_correct
        ),
        "observed_authenticated_aligned_tensor_count": verified_tensors,
        "observed_total_forward_tokens": total_forward_tokens,
        "observed_total_model_seconds": total_model_seconds,
        "observed_aggregate_tokens_per_second": (
            total_forward_tokens / total_model_seconds
        ),
        "observed_process_peak_rss_bytes": peak_rss,
        "observed_all_measurements_finite": finite,
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_denominator"])
        and total_correct >= int(gate_spec["minimum_correct"])
        and int(prereg["control"]["q4_correct"]) - total_correct
        <= int(gate_spec["maximum_correct_loss_vs_q4"])
        and verified_tensors
        == int(gate_spec["required_authenticated_aligned_tensor_count"])
        and peak_rss <= int(gate_spec["maximum_process_peak_rss_bytes"])
        and finite
    )
    finished = datetime.now(timezone.utc)
    receipt = {
        "schema": "qwen38_27b_aligned2_math500_receipt_v1",
        "created_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent": {
            "path": str(args.parent_receipt),
            "sha256": hashlib.sha256(parent_content).hexdigest(),
        },
        "artifacts": {
            "source_container": source,
            "cache": cache,
            "all_666_aligned_tensor_sha256_values_reverified": True,
        },
        "native_library": {
            "path": native_library,
            "sha256": sha256_file(Path(native_library)),
        },
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "physical_cpu_ids": cpu_ids,
        "cpu_features": features,
        "transformers_version": transformers_version,
        "tokenizer_files": tokenizer_files,
        "eos_token_ids": sorted(eos_token_ids),
        "manifest": {
            "path": str(manifest_path),
            "content_sha256": claimed,
            "file_sha256": sha256_file(manifest_path),
        },
        "control": {
            "path": str(q4_report_path),
            "sha256": sha256_file(q4_report_path),
            "model_sha256": q4_report["model_sha256"],
            "correct": q4_summary["correct"],
            "denominator": q4_summary["n"],
        },
        "restartability": {
            "progress_path": str(args.progress),
            "progress_sha256": sha256_file(args.progress),
            "preexisting_completed_records": preexisting_records,
            "atomic_checkpoint_after_each_item": True,
        },
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summaries": [
            {
                "task": "math500",
                "correct": total_correct,
                "n": len(records),
                "accuracy": total_correct / len(records),
            }
        ],
        "records": records,
        "measurement": {
            "total_forward_tokens": total_forward_tokens,
            "total_model_seconds": total_model_seconds,
            "aggregate_tokens_per_second": total_forward_tokens
            / total_model_seconds,
            "process_peak_rss_bytes": peak_rss,
        },
        "gate": gate,
        "claim_boundary": {
            "frozen_math500_quick_8_complete": True,
            "leaderboard_comparable": False,
            "math_forgetting_tripwire_passed": gate["passed"],
            "subject_balanced_mmlu_proven": False,
            "full_benchmark_capability_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    receipt["report_content_sha256"] = canonical_digest(receipt)
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
