"""Honest storage accounting for a deployable Qwen3.8-27B ternary image."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


LOG2_3 = math.log2(3.0)
LAYOUT_ALIGNMENT = 64
LAYOUT_HEADER_BYTES = 128
LAYOUT_FOOTER_BYTES = 32
TENSOR_RECORD_FIXED_BYTES = 64

DTYPE_BITS = {
    "BF16": 16,
    "F16": 16,
    "F32": 32,
    "F64": 64,
    "I8": 8,
    "U8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
}


def align_up(value: int, alignment: int = LAYOUT_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("value must be non-negative and alignment positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name or not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor name and positive dimensions are required")
        if self.dtype not in DTYPE_BITS:
            raise ValueError(f"unsupported tensor dtype: {self.dtype}")

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)

    @property
    def quantized(self) -> bool:
        return len(self.shape) >= 2

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def columns(self) -> int:
        return math.prod(self.shape[1:])

    @property
    def retained_bytes(self) -> int:
        return math.ceil(self.parameters * DTYPE_BITS[self.dtype] / 8)


@dataclass(frozen=True)
class StorageStrategy:
    name: str
    codec: str
    group_size: int | None
    scale_bits: int
    scale_codebook: bool = False

    def __post_init__(self) -> None:
        if self.codec not in {"base3", "radix5", "aligned2"}:
            raise ValueError(f"unsupported codec: {self.codec}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group size must be positive")
        if self.scale_bits <= 0 or self.scale_bits > 16:
            raise ValueError("scale bits must be in [1, 16]")
        if self.scale_codebook != (self.scale_bits < 16):
            raise ValueError("sub-16-bit scales require a per-tensor FP16 codebook")


def canonical_inventory_digest(tensors: Sequence[TensorSpec]) -> str:
    payload = json.dumps(
        [
            {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
            for tensor in sorted(tensors, key=lambda item: item.name)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _codeword_bits(elements: int, codec: str) -> int:
    if elements <= 0:
        return 0
    if codec == "base3":
        return math.ceil(elements * LOG2_3)
    if codec == "radix5":
        return math.ceil(elements / 5) * 8
    if codec == "aligned2":
        return math.ceil(elements / 4) * 8
    raise ValueError(codec)


def tensor_code_bits(tensor: TensorSpec, strategy: StorageStrategy) -> int:
    if not tensor.quantized:
        return 0
    columns = tensor.columns
    group = strategy.group_size or columns
    full_groups, remainder = divmod(columns, group)
    bits_per_row = full_groups * _codeword_bits(group, strategy.codec)
    bits_per_row += _codeword_bits(remainder, strategy.codec)
    return tensor.rows * bits_per_row


def tensor_scale_count(tensor: TensorSpec, strategy: StorageStrategy) -> int:
    if not tensor.quantized:
        return 0
    group = strategy.group_size or tensor.columns
    return tensor.rows * math.ceil(tensor.columns / group)


def tensor_record_bytes(tensor: TensorSpec) -> int:
    variable = len(tensor.name.encode("utf-8")) + 8 * len(tensor.shape)
    return align_up(TENSOR_RECORD_FIXED_BYTES + variable, 8)


def layout_metadata_bytes(tensors: Sequence[TensorSpec]) -> int:
    return align_up(
        LAYOUT_HEADER_BYTES
        + LAYOUT_FOOTER_BYTES
        + sum(tensor_record_bytes(tensor) for tensor in tensors)
    )


def strategy_ledger(
    tensors: Sequence[TensorSpec],
    strategy: StorageStrategy,
    auxiliary_bytes: int = 0,
) -> dict:
    if auxiliary_bytes < 0:
        raise ValueError("auxiliary bytes must be non-negative")
    quantized = [tensor for tensor in tensors if tensor.quantized]
    retained = [tensor for tensor in tensors if not tensor.quantized]
    total_parameters = sum(tensor.parameters for tensor in tensors)
    if total_parameters <= 0:
        raise ValueError("empty tensor inventory")

    code_payload_bytes = 0
    code_section_bytes = 0
    scale_index_bytes = 0
    scale_codebook_bytes = 0
    scale_section_bytes = 0
    scale_count = 0
    for tensor in quantized:
        code_bytes = math.ceil(tensor_code_bits(tensor, strategy) / 8)
        code_payload_bytes += code_bytes
        code_section_bytes += align_up(code_bytes)

        count = tensor_scale_count(tensor, strategy)
        indices = math.ceil(count * strategy.scale_bits / 8)
        codebook = (1 << strategy.scale_bits) * 2 if strategy.scale_codebook else 0
        scale_count += count
        scale_index_bytes += indices
        scale_codebook_bytes += codebook
        scale_section_bytes += align_up(indices + codebook)

    retained_payload_bytes = sum(tensor.retained_bytes for tensor in retained)
    retained_section_bytes = sum(align_up(tensor.retained_bytes) for tensor in retained)
    metadata_bytes = layout_metadata_bytes(tensors)
    payload_bytes = code_payload_bytes + scale_index_bytes + scale_codebook_bytes + retained_payload_bytes
    aligned_weight_bytes = code_section_bytes + scale_section_bytes + retained_section_bytes
    alignment_padding_bytes = aligned_weight_bytes - payload_bytes
    complete_bundle_bytes = aligned_weight_bytes + metadata_bytes + auxiliary_bytes
    return {
        "strategy": {
            "name": strategy.name,
            "codec": strategy.codec,
            "group_size": strategy.group_size,
            "group_semantics": "one flattened row" if strategy.group_size is None else "fixed",
            "scale_bits": strategy.scale_bits,
            "scale_codebook": strategy.scale_codebook,
        },
        "parameters": {
            "total": total_parameters,
            "ternary": sum(tensor.parameters for tensor in quantized),
            "retained_exact": sum(tensor.parameters for tensor in retained),
        },
        "tensors": {
            "total": len(tensors),
            "ternary": len(quantized),
            "retained_exact": len(retained),
        },
        "scales": {
            "count": scale_count,
            "index_bytes": scale_index_bytes,
            "fp16_codebook_bytes": scale_codebook_bytes,
        },
        "bytes": {
            "ternary_code_payload": code_payload_bytes,
            "ternary_code_sections_aligned": code_section_bytes,
            "scale_payload_and_codebooks": scale_index_bytes + scale_codebook_bytes,
            "scale_sections_aligned": scale_section_bytes,
            "retained_exact_payload": retained_payload_bytes,
            "retained_exact_sections_aligned": retained_section_bytes,
            "alignment_padding": alignment_padding_bytes,
            "tensor_table_and_layout_metadata": metadata_bytes,
            "auxiliary_model_files": auxiliary_bytes,
            "complete_bundle": complete_bundle_bytes,
        },
        "effective_bits_per_total_parameter": complete_bundle_bytes * 8 / total_parameters,
        "ternary_code_payload_bits_per_ternary_parameter": code_payload_bytes * 8
        / sum(tensor.parameters for tensor in quantized),
    }

def entropy_lower_bound(
    tensors: Sequence[TensorSpec], auxiliary_bytes: int = 0
) -> dict:
    quantized = [tensor for tensor in tensors if tensor.quantized]
    retained = [tensor for tensor in tensors if not tensor.quantized]
    total = sum(tensor.parameters for tensor in tensors)
    code_bits = sum(tensor.parameters for tensor in quantized) * LOG2_3
    retained_bits = sum(tensor.retained_bytes for tensor in retained) * 8
    fixed_bits = (layout_metadata_bytes(tensors) + auxiliary_bytes) * 8
    total_bits = code_bits + retained_bits + fixed_bits
    return {
        "description": "unattainable lower bound with no scale representation or section alignment",
        "ternary_entropy_bits": code_bits,
        "retained_exact_bits": retained_bits,
        "metadata_and_auxiliary_bits": fixed_bits,
        "effective_bits_per_total_parameter": total_bits / total,
    }


def default_strategies() -> tuple[StorageStrategy, ...]:
    return (
        StorageStrategy("base3_group128_fp16", "base3", 128, 16),
        StorageStrategy("base3_group128_codebook6", "base3", 128, 6, True),
        StorageStrategy("base3_group128_codebook4", "base3", 128, 4, True),
        StorageStrategy("base3_group512_fp16", "base3", 512, 16),
        StorageStrategy("base3_row_fp16", "base3", None, 16),
        StorageStrategy("radix5_group128_fp16", "radix5", 128, 16),
        StorageStrategy("aligned2_group128_fp16", "aligned2", 128, 16),
    )


def inventory_from_safetensors(model_dir: Path) -> tuple[list[TensorSpec], dict]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to inventory the checkpoint") from error

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    index_content = index_path.read_bytes()
    index = json.loads(index_content)
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)

    tensors: list[TensorSpec] = []
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                view = handle.get_slice(name)
                tensors.append(TensorSpec(name, tuple(view.get_shape()), str(view.get_dtype())))

    auxiliary = []
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        auxiliary.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return tensors, {
        "index_path": str(index_path),
        "index_sha256": hashlib.sha256(index_content).hexdigest(),
        "inventory_sha256": canonical_inventory_digest(tensors),
        "auxiliary_files": auxiliary,
        "auxiliary_bytes": sum(item["bytes"] for item in auxiliary),
    }


def build_receipt(model_dir: Path) -> dict:
    tensors, source = inventory_from_safetensors(model_dir)
    auxiliary_bytes = source["auxiliary_bytes"]
    ledgers = [strategy_ledger(tensors, strategy, auxiliary_bytes) for strategy in default_strategies()]
    return {
        "schema": "qwen38_27b_storage_budget_v1",
        "model_dir": str(Path(model_dir).resolve()),
        "source": source,
        "inventory": {
            "tensor_count": len(tensors),
            "total_parameters": sum(tensor.parameters for tensor in tensors),
            "ternary_rank_ge_2_tensors": sum(tensor.quantized for tensor in tensors),
            "ternary_rank_ge_2_parameters": sum(
                tensor.parameters for tensor in tensors if tensor.quantized
            ),
            "retained_rank_1_tensors": sum(not tensor.quantized for tensor in tensors),
            "retained_rank_1_parameters": sum(
                tensor.parameters for tensor in tensors if not tensor.quantized
            ),
        },
        "theoretical_lower_bound": entropy_lower_bound(tensors, auxiliary_bytes),
        "strategies": {ledger["strategy"]["name"]: ledger for ledger in ledgers},
        "candidate": {
            "name": "base3_group128_codebook4",
            "reason": "retains group-128 ternary granularity while reducing scale storage; scale-codebook quality and direct-decode speed remain unproven",
        },
        "claim_boundary": {
            "projected_layout_only": True,
            "artifact_exported": False,
            "model_quality_proven": False,
            "direct_kernel_speed_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
