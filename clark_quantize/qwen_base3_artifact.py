"""Read and reconstruct tensors from a Qwen full-base3 research artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .base3 import frame_metadata, unpack_base3


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def decode_artifact_record(root: str | Path, group_size: int, record: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """Decode and authenticate one record without loading the full manifest.

    This top-level function is intentionally process-pool safe.  Large-model
    consumers can parallelize the arithmetic decoder while keeping model
    mutation ordered and confined to their parent process.
    """
    root = Path(root)
    index = int(record["index"])
    descriptor = record["descriptor"]
    frame = (root / "codes.cq3d" / "chunks" / f"{index:08d}.cq3a").read_bytes()
    if _sha256(frame) != record["frame_sha256"]:
        raise ValueError(f"chunk {index} frame SHA-256 mismatch")
    if frame_metadata(frame) != record["frame_metadata"]:
        raise ValueError(f"chunk {index} frame metadata mismatch")
    codes = unpack_base3(frame)
    if _sha256(codes.tobytes()) != record["raw_sha256"]:
        raise ValueError(f"chunk {index} raw SHA-256 mismatch")
    rows = descriptor["row_stop"] - descriptor["row_start"]
    columns = descriptor["tensor_shape"][1]
    if codes.size != rows * columns:
        raise ValueError(f"chunk {index} code shape mismatch")
    codes = codes.reshape(rows, columns)

    scale_content = (root / "scales" / descriptor["scale_file"]).read_bytes()
    if len(scale_content) != descriptor["scale_bytes"]:
        raise ValueError(f"chunk {index} scale length mismatch")
    if _sha256(scale_content) != descriptor["scale_sha256"]:
        raise ValueError(f"chunk {index} scale SHA-256 mismatch")
    scales = np.frombuffer(scale_content, dtype="<f2").reshape(descriptor["scale_shape"]).copy()
    expected_groups = (columns + group_size - 1) // group_size
    if scales.shape != (rows, expected_groups):
        raise ValueError(f"chunk {index} scale shape mismatch")
    return index, codes, scales


class QwenBase3ArtifactReader:
    """Integrity-checking, bounded-memory reader for ``full_base3_pack.py``.

    ``tensor_batches`` is the production-safe interface. ``tensor`` is a
    convenience for tests or callers that intentionally want a whole matrix.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.codes_root = self.root / "codes.cq3d"
        self.manifest_path = self.codes_root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != "clark_chunked_base3_v1" or not self.manifest.get("complete"):
            raise ValueError("Qwen base-3 artifact manifest is missing or incomplete")
        identity = self.manifest.get("identity", {})
        if identity.get("schema") != "qwen38_full_base3_identity_v1":
            raise ValueError("Qwen base-3 artifact identity mismatch")
        self.group_size = int(identity["quantizer"]["group_size"])
        self.sources = {source["name"]: source for source in identity["mapped_sources"]}
        self.records_by_tensor: dict[str, list[dict]] = {name: [] for name in self.sources}
        for expected_index, record in enumerate(self.manifest["chunks"]):
            if record.get("index") != expected_index:
                raise ValueError("Qwen base-3 artifact records are not contiguous")
            tensor = record["descriptor"]["tensor"]
            if tensor not in self.records_by_tensor:
                raise ValueError(f"artifact record references unknown tensor {tensor}")
            self.records_by_tensor[tensor].append(record)
        for name, records in self.records_by_tensor.items():
            expected_row = 0
            for record in records:
                descriptor = record["descriptor"]
                if descriptor["row_start"] != expected_row:
                    raise ValueError(f"tensor {name} row batches are not contiguous")
                expected_row = descriptor["row_stop"]
            if expected_row != self.sources[name]["shape"][0]:
                raise ValueError(f"tensor {name} row coverage is incomplete")

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self.sources)

    def raw_batch(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return verified int8 codes and FP16 scales for one manifest record."""
        _, codes, scales = decode_artifact_record(self.root, self.group_size, record)
        return codes, scales

    def tensor_batches(self, name: str, dtype: torch.dtype = torch.float16) -> Iterator[torch.Tensor]:
        """Yield integrity-checked dequantized row batches for ``name``."""
        if name not in self.records_by_tensor:
            raise KeyError(name)
        for record in self.records_by_tensor[name]:
            codes, scales = self.raw_batch(record)
            code_tensor = torch.from_numpy(codes)
            scale_tensor = torch.from_numpy(scales)
            columns = code_tensor.shape[1]
            padding = (-columns) % self.group_size
            if padding:
                code_tensor = torch.nn.functional.pad(code_tensor, (0, padding))
            reconstructed = (
                code_tensor.reshape(code_tensor.shape[0], -1, self.group_size).to(dtype)
                * scale_tensor.to(dtype).unsqueeze(-1)
            ).reshape(code_tensor.shape[0], -1)[:, :columns]
            yield reconstructed.contiguous()

    def tensor(self, name: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return torch.cat(list(self.tensor_batches(name, dtype=dtype)), dim=0)

    def validate(self) -> dict[str, int]:
        """Decode every batch and return exact tensor/trit/scale counts."""
        trits = scales = batches = 0
        for name in self.tensor_names:
            for record in self.records_by_tensor[name]:
                codes, block_scales = self.raw_batch(record)
                trits += int(codes.size)
                scales += int(block_scales.size)
                batches += 1
        return {"tensors": len(self.tensor_names), "batches": batches, "trits": trits, "scales": scales}
