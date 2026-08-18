#!/usr/bin/env python3
"""Reconcile the complete HF checkpoint with the deployed llama.cpp text graph."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_row_scale_storage import row_normalized_scale_ledger
from clark_quantize.qwen38_storage_budget import (
    StorageStrategy,
    canonical_inventory_digest,
    inventory_from_safetensors,
    strategy_ledger,
)
from clark_quantize.qwen38_text_scope import (
    qwen38_checkpoint_scope_partition,
    tensor_scope_summary,
)


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
FULL_INVENTORY_SHA256 = "8533a0edfba093faf2a865391405408461613b92ef83e8d511436f2ed96bc14d"
TEXT_INVENTORY_SHA256 = "23a5c5137b7893b6d551fcc0085c4a994bf7ae158938d57974d32c7ed856361f"
STORAGE_RECEIPT_SHA256 = "1e37e00ff6736e6614d77f7dd1a1f69aecae96503e1659b0d027df9a37caec92"
Q4_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B")
    )
    parser.add_argument(
        "--q4-benchmark",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-layer0-mlp-cq3-paired-throughput-v1/"
            "baseline-prefill-t1.json"
        ),
    )
    parser.add_argument(
        "--q4-inputs",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-layer0-mlp-cq3-paired-throughput-v1/"
            "inputs.sha256"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-text-graph-scope-v1.json"),
    )
    args = parser.parse_args()

    runtime = Path(__file__).resolve().parent
    amendment_path = runtime / "qwen38_27b_text_graph_scope_amendment_v1.json"
    storage_path = runtime / "receipts/qwen38-27b-storage-budget-v1.json"
    if sha256_file(storage_path) != STORAGE_RECEIPT_SHA256:
        raise RuntimeError("frozen full-checkpoint storage receipt changed")
    storage_receipt = json.loads(storage_path.read_text())
    amendment = json.loads(amendment_path.read_text())

    download_path = args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    download = json.loads(download_path.read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": INDEX_SHA256,
        "model_id": MODEL,
        "revision": REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8 checkpoint identity changed")

    tensors, source = inventory_from_safetensors(args.model_dir)
    if (
        source["index_sha256"] != INDEX_SHA256
        or source["inventory_sha256"] != FULL_INVENTORY_SHA256
        or source["auxiliary_files"] != storage_receipt["source"]["auxiliary_files"]
    ):
        raise RuntimeError("Qwen3.8 checkpoint inventory changed")
    partition = qwen38_checkpoint_scope_partition(tensors)
    observed = {
        label: tensor_scope_summary(values) for label, values in partition.items()
    }
    observed["checkpoint"] = tensor_scope_summary(tensors)
    text_inventory_sha256 = canonical_inventory_digest(partition["text"])
    if text_inventory_sha256 != TEXT_INVENTORY_SHA256:
        raise RuntimeError("Qwen3.8 text inventory digest changed")

    frozen = amendment["selected_text_graph_scope"]
    expected_text = {
        "tensors": frozen["tensors"],
        "parameters": frozen["parameters"],
        "rank_ge_2_tensors": frozen["rank_ge_2_tensors"],
        "rank_ge_2_parameters": frozen["rank_ge_2_parameters"],
        "retained_exact_tensors": frozen["retained_exact_tensors"],
        "retained_exact_parameters": frozen["retained_exact_parameters"],
        "output_rows": frozen["output_rows"],
        "group_128_scales": frozen["group_128_scales"],
    }
    if observed["text"] != expected_text:
        raise RuntimeError("observed text scope differs from the frozen amendment")
    for label, amendment_label in (("vision", "vision"), ("mtp_nextn", "mtp_nextn")):
        expected = amendment["excluded_non_text_scopes"][amendment_label]
        if (
            observed[label]["tensors"] != expected["tensors"]
            or observed[label]["parameters"] != expected["parameters"]
        ):
            raise RuntimeError(f"observed {label} scope differs from the amendment")

    base_storage = strategy_ledger(
        partition["text"],
        StorageStrategy("base3_group128_codebook4", "base3", 128, 4, True),
        source["auxiliary_bytes"],
    )
    storage = row_normalized_scale_ledger(partition["text"], base_storage)
    if (
        storage["bytes"]["complete_bundle"]
        != frozen["projected_complete_bundle_bytes"]
        or not math.isclose(
            storage["effective_bits_per_total_parameter"],
            frozen["projected_complete_bundle_bpw"],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("text-graph storage projection changed")

    q4_rows = json.loads(args.q4_benchmark.read_text())
    q4_input_lines = [line.split(maxsplit=1) for line in args.q4_inputs.read_text().splitlines()]
    q4_input_bound = any(
        len(parts) == 2
        and parts[0] == Q4_SHA256
        and Path(parts[1].strip()).name == "Qwen3.8-27B-Q4_K_M.gguf"
        for parts in q4_input_lines
    )
    if (
        not isinstance(q4_rows, list)
        or not q4_rows
        or {int(row["model_n_params"]) for row in q4_rows} != {frozen["parameters"]}
        or {Path(row["model_filename"]).name for row in q4_rows}
        != {"Qwen3.8-27B-Q4_K_M.gguf"}
        or not q4_input_bound
    ):
        raise RuntimeError("llama.cpp Q4 evidence has another logical graph")

    arithmetic = {
        "text_plus_vision_plus_mtp_equals_checkpoint": (
            observed["text"]["parameters"]
            + observed["vision"]["parameters"]
            + observed["mtp_nextn"]["parameters"]
            == observed["checkpoint"]["parameters"]
        ),
        "text_matches_llama_cpp_model_n_params": all(
            int(row["model_n_params"]) == observed["text"]["parameters"]
            for row in q4_rows
        ),
        "text_storage_bpw_at_most_1_64": (
            storage["effective_bits_per_total_parameter"] <= 1.64
        ),
    }
    receipt = {
        "schema": "qwen38_27b_text_graph_scope_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
        "revision": REVISION,
        "checkpoint": {
            "download_receipt": str(download_path),
            "download_receipt_sha256": sha256_file(download_path),
            "index_sha256": source["index_sha256"],
            "inventory_sha256": source["inventory_sha256"],
        },
        "text_inventory_sha256": text_inventory_sha256,
        "scopes": observed,
        "llama_cpp_q4_evidence": {
            "path": str(args.q4_benchmark),
            "sha256": sha256_file(args.q4_benchmark),
            "inputs_path": str(args.q4_inputs),
            "inputs_sha256": sha256_file(args.q4_inputs),
            "model_sha256_from_inputs_contract": Q4_SHA256,
            "model_n_params": frozen["parameters"],
        },
        "projected_text_storage": storage,
        "amendment": {
            "path": str(amendment_path),
            "sha256": sha256_file(amendment_path),
        },
        "checks": arithmetic,
        "valid": all(arithmetic.values()),
        "complete": True,
        "terminal": True,
        "claim_boundary": (
            "exact graph-scope and projected byte reconciliation only; no scale "
            "reconstruction, artifact, capability, throughput advantage, or breakthrough"
        ),
    }
    if not receipt["valid"]:
        raise RuntimeError("Qwen3.8 text-scope reconciliation checks failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
