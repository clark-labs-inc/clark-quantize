from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


GGUF_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    42: "Q2_0",
}

GGUF_META_TYPES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        if len(out) != n:
            raise EOFError("unexpected EOF")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack_from("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.read(4))[0]

    def f32(self) -> float:
        return struct.unpack_from("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack_from("<d", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")


def read_meta_value(r: Reader, typ: int) -> Any:
    if typ == 0:
        return r.read(1)[0]
    if typ == 1:
        return struct.unpack_from("<b", r.read(1))[0]
    if typ == 2:
        return struct.unpack_from("<H", r.read(2))[0]
    if typ == 3:
        return struct.unpack_from("<h", r.read(2))[0]
    if typ == 4:
        return r.u32()
    if typ == 5:
        return r.i32()
    if typ == 6:
        return r.f32()
    if typ == 7:
        return bool(r.read(1)[0])
    if typ == 8:
        return r.string()
    if typ == 9:
        elem_type = r.u32()
        n = r.u64()
        return [read_meta_value(r, elem_type) for _ in range(n)]
    if typ == 10:
        return r.u64()
    if typ == 11:
        return struct.unpack_from("<q", r.read(8))[0]
    if typ == 12:
        return r.f64()
    raise ValueError(f"unsupported metadata type {typ}")


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_nbytes(tensor_type: int, n_elements: int) -> int:
    if tensor_type == 0:
        return n_elements * 4
    if tensor_type == 1:
        return n_elements * 2
    if tensor_type == 42:
        if n_elements % 128:
            raise ValueError(f"Q2_0 tensor has non-multiple-of-128 elements: {n_elements}")
        return n_elements // 128 * 34
    raise ValueError(f"unknown tensor type {tensor_type}")


def parse_gguf(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    data = path.read_bytes()
    r = Reader(data)
    magic = r.read(4)
    if magic != b"GGUF":
        raise ValueError(f"bad magic: {magic!r}")
    version = r.u32()
    tensor_count = r.u64()
    metadata_count = r.u64()
    metadata: dict[str, Any] = {"gguf.version": version}
    for _ in range(metadata_count):
        key = r.string()
        typ = r.u32()
        metadata[key] = read_meta_value(r, typ)

    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        tensor_type = r.u32()
        offset = r.u64()
        n_elements = math.prod(dims)
        tensors.append(
            {
                "name": name,
                "shape": dims,
                "tensor_type": tensor_type,
                "type_name": GGUF_TYPE_NAMES.get(tensor_type, str(tensor_type)),
                "offset": offset,
                "n_elements": n_elements,
                "n_bytes": tensor_nbytes(tensor_type, n_elements),
            }
        )
    alignment = int(metadata.get("general.alignment", 32))
    data_start = align(r.pos, alignment)
    return metadata, tensors, data_start


def classify_tensor(name: str, shape: list[int]) -> str:
    if name == "token_embd.weight":
        return "embed"
    if name == "output.weight":
        return "lm_head"
    if "attn_q.weight" in name:
        return "q_proj"
    if "attn_k.weight" in name:
        return "k_proj"
    if "attn_v.weight" in name:
        return "v_proj"
    if "attn_output.weight" in name:
        return "o_proj"
    if "ffn_gate.weight" in name:
        return "gate_proj"
    if "ffn_up.weight" in name:
        return "up_proj"
    if "ffn_down.weight" in name:
        return "down_proj"
    if len(shape) == 1:
        return "norm_1d"
    return "other"


def q2_counts_and_scales(path: Path, tensors: list[dict[str, Any]], data_start: int) -> dict[str, Any]:
    q_lut = np.zeros((256, 4), dtype=np.int64)
    for byte in range(256):
        for slot in range(4):
            q_lut[byte, (byte >> (2 * slot)) & 3] += 1

    global_counts = np.zeros(4, dtype=np.int64)
    by_kind: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    by_layer: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    zero_hist = np.zeros(129, dtype=np.int64)
    plus_hist = np.zeros(129, dtype=np.int64)
    scale_rows = []
    tensor_rows = []
    with path.open("rb") as f:
        for t in tensors:
            if t["tensor_type"] != 42:
                continue
            kind = classify_tensor(t["name"], t["shape"])
            layer = "non_layer"
            parts = t["name"].split(".")
            if len(parts) > 2 and parts[0] == "blk":
                layer = parts[1]
            f.seek(data_start + int(t["offset"]))
            raw = np.frombuffer(f.read(t["n_bytes"]), dtype=np.uint8)
            blocks = raw.reshape(-1, 34)
            scales = blocks[:, :2].reshape(-1).view(np.float16).astype(np.float32)
            packed = blocks[:, 2:].reshape(-1)
            counts = q_lut[packed].sum(axis=0)
            block_counts = q_lut[blocks[:, 2:]].sum(axis=1)
            zero_hist += np.bincount(block_counts[:, 1], minlength=129)
            plus_hist += np.bincount(block_counts[:, 2], minlength=129)
            global_counts += counts
            by_kind[kind] += counts
            by_layer[layer] += counts
            scale_rows.append(
                {
                    "name": t["name"],
                    "kind": kind,
                    "layer": layer,
                    "blocks": int(blocks.shape[0]),
                    "scale_mean": float(np.mean(scales)),
                    "scale_abs_mean": float(np.mean(np.abs(scales))),
                    "scale_p01": float(np.quantile(scales, 0.01)),
                    "scale_p50": float(np.quantile(scales, 0.50)),
                    "scale_p99": float(np.quantile(scales, 0.99)),
                    "zero_scale_blocks": int(np.sum(scales == 0)),
                    "negative_scale_blocks": int(np.sum(scales < 0)),
                }
            )
            tensor_rows.append(
                {
                    "name": t["name"],
                    "kind": kind,
                    "shape": t["shape"],
                    "n_elements": t["n_elements"],
                    "counts": [int(x) for x in counts],
                    "zero_fraction": float(counts[1] / counts.sum()),
                    "plus_fraction": float(counts[2] / counts.sum()),
                    "minus_fraction": float(counts[0] / counts.sum()),
                    "unused_q3_fraction": float(counts[3] / counts.sum()),
                    "scale_abs_mean": float(np.mean(np.abs(scales))),
                    "scale_p50": float(np.quantile(np.abs(scales), 0.50)),
                    "scale_p99": float(np.quantile(np.abs(scales), 0.99)),
                }
            )
    return {
        "global_counts": [int(x) for x in global_counts],
        "by_kind": {k: [int(x) for x in v] for k, v in sorted(by_kind.items())},
        "by_layer": {k: [int(x) for x in v] for k, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)},
        "zero_hist": [int(x) for x in zero_hist],
        "plus_hist": [int(x) for x in plus_hist],
        "scale_rows": scale_rows,
        "tensor_rows": tensor_rows,
    }


def summarize(metadata: dict[str, Any], tensors: list[dict[str, Any]], q2: dict[str, Any]) -> dict[str, Any]:
    type_counts = Counter(t["type_name"] for t in tensors)
    type_params = Counter()
    type_bytes = Counter()
    kind_params = Counter()
    for t in tensors:
        type_params[t["type_name"]] += int(t["n_elements"])
        type_bytes[t["type_name"]] += int(t["n_bytes"])
        kind_params[classify_tensor(t["name"], t["shape"])] += int(t["n_elements"])
    total_params = sum(t["n_elements"] for t in tensors)
    q = np.array(q2["global_counts"], dtype=np.float64)
    q_total = q.sum()
    kind_summary = {}
    for kind, counts in q2["by_kind"].items():
        arr = np.array(counts, dtype=np.float64)
        denom = arr.sum()
        kind_summary[kind] = {
            "params": int(denom),
            "minus": float(arr[0] / denom),
            "zero": float(arr[1] / denom),
            "plus": float(arr[2] / denom),
            "q3": float(arr[3] / denom),
        }
    layer_summary = {}
    for layer, counts in q2["by_layer"].items():
        arr = np.array(counts, dtype=np.float64)
        denom = arr.sum()
        layer_summary[layer] = {
            "params": int(denom),
            "minus": float(arr[0] / denom),
            "zero": float(arr[1] / denom),
            "plus": float(arr[2] / denom),
            "q3": float(arr[3] / denom),
        }
    zero_hist = np.array(q2["zero_hist"], dtype=np.float64)
    zero_total = zero_hist.sum()
    zero_cdf = np.cumsum(zero_hist) / zero_total
    zero_quantiles = {
        f"p{int(q * 100):02d}": int(np.searchsorted(zero_cdf, q))
        for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    }
    return {
        "metadata_selected": {
            key: metadata.get(key)
            for key in [
                "general.name",
                "general.architecture",
                "general.file_type",
                "qwen3.context_length",
                "qwen3.block_count",
                "qwen3.embedding_length",
                "qwen3.feed_forward_length",
                "qwen3.attention.head_count",
                "qwen3.attention.head_count_kv",
                "tokenizer.ggml.tokens",
            ]
            if key in metadata
        },
        "tensor_count": len(tensors),
        "total_params": int(total_params),
        "type_counts": dict(type_counts),
        "type_params": dict(type_params),
        "type_bytes": dict(type_bytes),
        "kind_params": dict(kind_params),
        "q2_global": {
            "counts": [int(x) for x in q],
            "minus": float(q[0] / q_total),
            "zero": float(q[1] / q_total),
            "plus": float(q[2] / q_total),
            "q3": float(q[3] / q_total),
        },
        "q2_by_kind": kind_summary,
        "q2_by_layer": layer_summary,
        "block_zero_hist": q2["zero_hist"],
        "block_plus_hist": q2["plus_hist"],
        "block_zero_count_quantiles": zero_quantiles,
        "top_zero_tensors": sorted(q2["tensor_rows"], key=lambda r: r["zero_fraction"], reverse=True)[:20],
        "lowest_zero_tensors": sorted(q2["tensor_rows"], key=lambda r: r["zero_fraction"])[:20],
        "top_scale_tensors": sorted(q2["tensor_rows"], key=lambda r: r["scale_abs_mean"], reverse=True)[:20],
        "q3_tensors": [r for r in q2["tensor_rows"] if r["unused_q3_fraction"] > 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata, tensors, data_start = parse_gguf(args.gguf)
    q2 = q2_counts_and_scales(args.gguf, tensors, data_start)
    report = summarize(metadata, tensors, q2)
    report["data_start"] = data_start
    report["tensors"] = tensors
    report["scale_rows"] = q2["scale_rows"]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["tensor_count", "total_params", "type_counts", "type_params", "type_bytes", "q2_global"]}, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
