from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantizer_controls import ParetoSeq, ProgressiveLowBit, quantize_ternary_absmean


class QuantizerControlTests(unittest.TestCase):
    def test_absmean_uses_three_codes_and_group_scales(self) -> None:
        weight = torch.tensor([[0.1, -0.8, 2.0, -3.0]])
        quantized = quantize_ternary_absmean(weight, group_size=2)

        self.assertEqual(quantized.codes.tolist(), [0, -1, 1, -1])
        torch.testing.assert_close(quantized.scales, torch.tensor([0.45, 2.5]))

    def test_paretoq_seq_has_exact_ternary_grid(self) -> None:
        weight = torch.tensor([[-2.0, -0.3, 0.0, 0.3, 2.0]])
        alpha = torch.tensor([[1.5]], requires_grad=True)
        output = ParetoSeq.apply(weight, alpha)

        torch.testing.assert_close(
            output,
            torch.tensor([[-1.0, 0.0, 0.0, 0.0, 1.0]]),
        )
        output.sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_progressive_seq_trains_scale_at_hard_endpoint(self) -> None:
        weight = torch.tensor([[-1.0, -0.2, 0.4, 1.0]], requires_grad=True)
        quantizer = ProgressiveLowBit(
            "ternary",
            "paretoq_seq",
            4,
            0.0,
            1.0,
            weight,
        )
        quantizer.set_progress(1.0)
        quantizer(weight).square().sum().backward()

        self.assertIsNotNone(quantizer.alpha.grad)
        self.assertGreater(float(quantizer.alpha.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
