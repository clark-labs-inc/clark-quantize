from __future__ import annotations

from pathlib import Path

import torch

from clark_quantize.qwen38_layer0_qat import (
    expected_cq3_payload_bytes,
    unpack_cq3_rows,
)
from clark_quantize.qwen38_mixed_cq3_runtime import (
    StreamedCq3Linear,
    write_step_zero_cq3_tensor,
)


def test_step_zero_writer_is_exact_and_storage_bounded(tmp_path: Path) -> None:
    torch.manual_seed(19)
    source = torch.randn(16, 256, dtype=torch.bfloat16) * 0.02
    output = tmp_path / "weight.cq3"
    descriptor = write_step_zero_cq3_tensor(source, output, row_shard=8)
    assert descriptor["bytes"] == expected_cq3_payload_bytes(16, 256)
    assert output.stat().st_size == descriptor["bytes"]
    assert len(descriptor["sha256"]) == 64
    codes, scales = unpack_cq3_rows(output.read_bytes(), rows=16, columns=256)
    assert codes.dtype == torch.int8
    assert bool(torch.isfinite(scales.float()).all())
    assert descriptor["complete"] is True
    assert descriptor["terminal"] is True


def test_streamed_cq3_linear_matches_the_stored_algebra(tmp_path: Path) -> None:
    torch.manual_seed(23)
    source = torch.randn(16, 256, dtype=torch.bfloat16) * 0.02
    payload = tmp_path / "weight.cq3"
    write_step_zero_cq3_tensor(source, payload, row_shard=8)
    codes, scales = unpack_cq3_rows(payload.read_bytes(), rows=16, columns=256)
    module = StreamedCq3Linear.from_payload(
        payload,
        rows=16,
        columns=256,
        row_shard=8,
        output_dtype=torch.float32,
    )
    inputs = torch.randn(2, 3, 256)
    observed = module(inputs)
    weight = (
        codes.reshape(16, 2, 128).float() * scales.float().unsqueeze(-1)
    ).reshape(16, 256)
    expected = torch.nn.functional.linear(inputs.float(), weight)
    assert torch.equal(observed, expected)
    assert "weight" not in dict(module.named_buffers())
    assert module.in_features == 256
    assert module.out_features == 16


def test_streamed_cq3_rejects_dimension_and_byte_mismatches(tmp_path: Path) -> None:
    source = torch.randn(8, 256, dtype=torch.bfloat16)
    payload = tmp_path / "weight.cq3"
    write_step_zero_cq3_tensor(source, payload, row_shard=8)
    module = StreamedCq3Linear.from_payload(payload, rows=8, columns=256)
    try:
        module(torch.randn(4, 128))
    except ValueError as error:
        assert "input dimension" in str(error)
    else:
        raise AssertionError("stored CQ3 accepted the wrong input dimension")
    broken = tmp_path / "broken.cq3"
    broken.write_bytes(payload.read_bytes()[:-1])
    try:
        StreamedCq3Linear.from_payload(broken, rows=8, columns=256)
    except ValueError as error:
        assert "byte count" in str(error)
    else:
        raise AssertionError("stored CQ3 accepted a truncated payload")
