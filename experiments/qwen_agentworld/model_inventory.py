"""Inspect Qwen AgentWorld safetensors index and build a compression ledger."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def classify_tensor(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if ".norm" in name or "layernorm" in name.lower() or "rms_norm" in name:
        return "norm"
    if ".experts." in name:
        return "moe_expert"
    if "shared_expert" in name:
        return "shared_expert"
    if "router" in name or name.endswith(".gate.weight") or ".mlp.gate." in name:
        return "router"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "full_attention"
    if ".linear_attn." in name:
        if any(k in name for k in ("in_proj", "out_proj")):
            return "linear_attention_projection"
        return "linear_attention_state"
    if any(k in name for k in ("up_proj", "down_proj", "gate_proj")):
        return "mlp"
    return "other"


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    return json.loads(index_path.read_text())


def product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def tensor_metadata(model_dir: Path, weight_map: dict[str, str]) -> dict[str, dict]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    meta = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for name in names:
                tensor_slice = f.get_slice(name)
                shape = [int(dim) for dim in tensor_slice.get_shape()]
                dtype = tensor_slice.get_dtype()
                numel = product(shape)
                bytes_per_param = DTYPE_BYTES.get(dtype)
                meta[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                    "bytes": numel * bytes_per_param if bytes_per_param else None,
                }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--out", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    args = parser.parse_args()

    index = load_index(args.model_dir)
    metadata = index.get("metadata", {})
    weight_map = index.get("weight_map", {})
    per_tensor = tensor_metadata(args.model_dir, weight_map)
    buckets: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "params": 0, "bytes": 0, "dtype_counts": defaultdict(int), "names": []}
    )
    tensor_summary = {}
    for name, shard in weight_map.items():
        bucket = classify_tensor(name)
        tensor_summary[name] = {
            **per_tensor[name],
            "bucket": bucket,
            "shard": shard,
        }
        buckets[bucket]["tensors"] += 1
        buckets[bucket]["params"] += per_tensor[name]["numel"]
        if per_tensor[name]["bytes"] is not None:
            buckets[bucket]["bytes"] += per_tensor[name]["bytes"]
        buckets[bucket]["dtype_counts"][per_tensor[name]["dtype"]] += 1
        if len(buckets[bucket]["names"]) < 20:
            buckets[bucket]["names"].append(name)

    total_size = int(metadata.get("total_size", 0))
    bucket_summary = {}
    for bucket, info in sorted(buckets.items()):
        bucket_summary[bucket] = {
            "tensors": info["tensors"],
            "params": info["params"],
            "bytes": info["bytes"],
            "gib": info["bytes"] / 1024**3,
            "dtype_counts": dict(sorted(info["dtype_counts"].items())),
            "names": info["names"],
        }
    result = {
        "model_dir": str(args.model_dir),
        "metadata": metadata,
        "total_size_bytes": total_size,
        "total_size_gib": total_size / 1024**3 if total_size else None,
        "tensor_count": len(weight_map),
        "bucket_summary": bucket_summary,
        "tensor_summary": tensor_summary,
        "target_8x_bytes": total_size / 8 if total_size else None,
        "target_8x_gib": total_size / 8 / 1024**3 if total_size else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
