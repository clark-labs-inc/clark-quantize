from collections import OrderedDict

import torch

from analyze_checkpoint import inspect


def test_inventory_detects_symmetric_ternary_pack(tmp_path):
    prefix = "transformer_blocks.0.attn.to_q"
    state = OrderedDict(
        {
            f"{prefix}.W_q": torch.tensor([[0b100100]] * 32, dtype=torch.uint8),
            f"{prefix}.scales": torch.tensor([[0.5]], dtype=torch.float32),
            f"{prefix}.zeros": torch.tensor([[-0.5]], dtype=torch.float32),
            f"{prefix}.metadata": torch.tensor(
                [0, 2, 128, 3, 4, 1, 1, 0, 0, 0, 4, 1], dtype=torch.int32
            ),
            f"{prefix}.orig_shape": torch.tensor([1, 128], dtype=torch.int32),
            "norm_out.linear.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        }
    )
    checkpoint = tmp_path / "state.pt"
    torch.save(state, checkpoint)

    report = inspect(checkpoint, expected_sha256=None)

    assert report["quantized_linears"] == 1
    assert report["quantized_logical_parameters"] == 128
    assert report["packed_codes"]["histogram"] == [64, 32, 32, 0]
    assert report["packed_codes"]["unused_code_3"] is True
    assert report["scale_and_zero_storage"]["zeros_are_negative_scales"] is True
    assert report["scale_and_zero_storage"]["scales_are_bf16_exact"] is True
    assert report["retained_high_precision"]["norm_out"]["bytes"] == 8
