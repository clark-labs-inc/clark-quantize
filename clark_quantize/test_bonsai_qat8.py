from __future__ import annotations

import unittest

import torch

from clark_quantize.bonsai_qat8 import (
    BonsaiQAT8,
    _dequantize_blockwise_int8,
    _quantize_blockwise_int8,
)
from clark_quantize.qwen_qat import state_tree_equal


class BonsaiQAT8Tests(unittest.TestCase):
    def test_blockwise_int8_moment_roundtrip_is_bounded(self) -> None:
        source = torch.linspace(-2.0, 3.0, 37).reshape(1, 37)
        quantized, scales, shape = _quantize_blockwise_int8(source, block_size=16)
        restored = _dequantize_blockwise_int8(
            quantized,
            scales,
            shape,
            block_size=16,
            device="cpu",
        )

        self.assertEqual(restored.shape, source.shape)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertLessEqual(
            float((restored - source).abs().max()),
            float(scales.max()) / 127.0 + 1e-6,
        )

    def test_matrix_path_builds_low_state_and_stays_finite(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(4, 6))
        optimizer = BonsaiQAT8(
            [{"params": [parameter], "role": "matrix_qat8", "lr": 1e-2}]
        )
        parameter.grad = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24

        optimizer.step()

        state = optimizer.state[parameter]
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertEqual(state["m_q"].dtype, torch.int8)
        self.assertEqual(tuple(state["v_row"].shape), (4,))
        self.assertEqual(tuple(state["v_col"].shape), (6,))
        self.assertEqual(state["step"], 1)
        self.assertEqual(optimizer.global_step, 1)

    def test_state_dict_preserves_global_schedule_step(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(3, 5))
        optimizer = BonsaiQAT8(
            [{"params": [parameter], "role": "matrix_qat8", "lr": 1e-2}]
        )
        for _ in range(3):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()

        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = BonsaiQAT8(
            [{"params": [resumed_parameter], "role": "matrix_qat8", "lr": 1e-2}]
        )
        resumed.load_state_dict(optimizer.state_dict())

        self.assertEqual(resumed.global_step, 3)
        self.assertEqual(resumed.state[resumed_parameter]["m_q"].dtype, torch.int8)
        self.assertTrue(state_tree_equal(resumed.state_dict(), optimizer.state_dict()))
        resumed_parameter.grad = torch.ones_like(resumed_parameter)
        resumed.step()
        self.assertEqual(resumed.global_step, 4)

    def test_legacy_state_infers_global_step(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = BonsaiQAT8(
            [{"params": [parameter], "role": "matrix_qat8", "lr": 1e-2}]
        )
        for _ in range(2):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        legacy = optimizer.state_dict()
        legacy.pop("bonsai_global_step")

        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = BonsaiQAT8(
            [{"params": [resumed_parameter], "role": "matrix_qat8", "lr": 1e-2}]
        )
        resumed.load_state_dict(legacy)

        self.assertEqual(resumed.global_step, 2)


if __name__ == "__main__":
    unittest.main()
