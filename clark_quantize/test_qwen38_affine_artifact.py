from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from clark_quantize.qwen38_affine_artifact import (
    load_affine_tensor,
    verify_descriptor,
    write_affine_tensor,
)
from clark_quantize.qwen38_affine_butterfly import AffineTernary
from clark_quantize.qwen38_affine_codebook import compress_affine_parameters


def test_packed_affine_tensor_roundtrip_authenticates_actual_storage() -> None:
    torch.manual_seed(93)
    payload = AffineTernary(
        codes=torch.randint(-1, 2, (5, 259), dtype=torch.int8),
        mu=torch.randn(5, 3) * 0.2,
        alpha=torch.rand(5, 3) + 0.1,
        group_size=128,
    )
    compressed = compress_affine_parameters(payload, chunk_rows=2)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        descriptor = write_affine_tensor(root, "projection.weight", compressed)
        verify_descriptor(root, descriptor)
        rebuilt = load_affine_tensor(root, descriptor)
        torch.testing.assert_close(
            rebuilt.reconstruct(), compressed.payload.reconstruct(), rtol=0, atol=0
        )
        assert descriptor["storage_bytes"] == (
            descriptor["codes"]["bytes"] + compressed.parameter_storage_bytes
        )


def test_packed_affine_tensor_rejects_tampering() -> None:
    payload = AffineTernary(
        codes=torch.zeros((2, 128), dtype=torch.int8),
        mu=torch.zeros((2, 1)),
        alpha=torch.ones((2, 1)),
        group_size=128,
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        descriptor = write_affine_tensor(
            root, "projection.weight", compress_affine_parameters(payload)
        )
        path = root / descriptor["codes"]["path"]
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
        try:
            load_affine_tensor(root, descriptor)
        except ValueError as error:
            assert "authentication" in str(error)
        else:
            raise AssertionError("tampered affine codes were accepted")
