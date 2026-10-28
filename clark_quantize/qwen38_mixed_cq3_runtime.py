"""Stored CQ3 helpers for the Qwen3.8-27B mixed-mode pilot."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as functional

from .qwen38_layer0_qat import (
    expected_cq3_payload_bytes,
    pack_cq3_rows,
    unpack_cq3_rows,
)
from .ternary import ternary_codes_scales


def tensor_sha256(tensor: torch.Tensor) -> str:
    content = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_step_zero_cq3_tensor(
    source: torch.Tensor,
    destination: Path,
    *,
    row_shard: int = 128,
) -> dict[str, object]:
    """Write the optimal-MSE group-128 CQ3 payload without a dense copy."""
    if (
        source.ndim != 2
        or source.shape[1] % 256
        or row_shard <= 0
        or row_shard % 8
    ):
        raise ValueError("invalid CQ3 source matrix or row shard")
    rows, columns = source.shape
    expected = expected_cq3_payload_bytes(rows, columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for start in range(0, rows, row_shard):
                stop = min(start + row_shard, rows)
                codes, scales = ternary_codes_scales(
                    source[start:stop].float(), group=128, method="optimal"
                )
                payload = pack_cq3_rows(codes, scales)
                checked_codes, checked_scales = unpack_cq3_rows(
                    payload, rows=stop - start, columns=columns
                )
                if not torch.equal(codes, checked_codes) or not torch.equal(
                    scales, checked_scales
                ):
                    raise AssertionError("step-zero CQ3 payload changed on roundtrip")
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != expected:
            raise AssertionError("step-zero CQ3 payload byte count changed")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "shape": [rows, columns],
        "row_shard": row_shard,
        "group_size": 128,
        "bytes": expected,
        "sha256": file_sha256(destination),
        "complete": True,
        "terminal": True,
    }


class StreamedCq3Linear(torch.nn.Module):
    """Execute a stored CQ3 payload without retaining a dense weight matrix."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        *,
        row_shard: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if (
            codes.ndim != 2
            or codes.dtype != torch.int8
            or scales.shape != (codes.shape[0], codes.shape[1] // 128)
            or row_shard <= 0
        ):
            raise ValueError("invalid stored CQ3 linear payload")
        self.register_buffer("codes", codes.contiguous(), persistent=False)
        self.register_buffer("scales", scales.contiguous(), persistent=False)
        self.row_shard = row_shard
        self.output_dtype = output_dtype
        self.in_features = codes.shape[1]
        self.out_features = codes.shape[0]
        self.bias = None

    @classmethod
    def from_payload(
        cls,
        path: Path,
        *,
        rows: int,
        columns: int,
        row_shard: int = 128,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "StreamedCq3Linear":
        if path.stat().st_size != expected_cq3_payload_bytes(rows, columns):
            raise ValueError("stored CQ3 payload byte count changed")
        codes, scales = unpack_cq3_rows(
            path.read_bytes(), rows=rows, columns=columns
        )
        if not bool(torch.isfinite(scales.float()).all()):
            raise ValueError("stored CQ3 payload contains non-finite scales")
        return cls(
            codes,
            scales,
            row_shard=row_shard,
            output_dtype=output_dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError("stored CQ3 linear input dimension changed")
        flat = inputs.reshape(-1, self.in_features).float()
        chunks = []
        for start in range(0, self.out_features, self.row_shard):
            stop = min(start + self.row_shard, self.out_features)
            weight = (
                self.codes[start:stop]
                .reshape(stop - start, -1, 128)
                .float()
                * self.scales[start:stop].float().unsqueeze(-1)
            ).reshape(stop - start, self.in_features)
            chunks.append(functional.linear(flat, weight))
        output = torch.cat(chunks, dim=-1)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("stored CQ3 linear produced non-finite output")
        return output.reshape(*inputs.shape[:-1], self.out_features).to(
            self.output_dtype
        )
