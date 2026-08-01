import unittest

import torch

from clark_air.quant.optimal_ternary import quantize_ternary_mse
from residual_ternary import residual_ternary_decomposition


class ResidualTernaryTest(unittest.TestCase):
    def test_one_term_matches_the_pinned_quantizer(self):
        weight = torch.tensor([
            [-1.2, -0.4, -0.1, 0.2, 0.8, 1.4],
            [0.7, -0.9, 0.0, 0.3, -0.2, 1.1],
        ])
        branches, approximation = residual_ternary_decomposition(weight, 1, 3)
        expected = quantize_ternary_mse(weight, group_size=3).dequantize()

        self.assertEqual(len(branches), 1)
        torch.testing.assert_close(approximation, expected)

    def test_each_residual_term_cannot_increase_squared_error(self):
        generator = torch.Generator().manual_seed(20260729)
        weight = torch.randn(12, 20, generator=generator)
        previous_error = float("inf")
        for terms in range(1, 5):
            _branches, approximation = residual_ternary_decomposition(
                weight,
                terms,
                group_size=20,
            )
            error = (weight - approximation).square().sum().item()
            self.assertLessEqual(error, previous_error + 1e-7)
            previous_error = error

    def test_rejects_invalid_term_count(self):
        with self.assertRaisesRegex(ValueError, "terms must be positive"):
            residual_ternary_decomposition(torch.ones(2, 2), 0, 2)


if __name__ == "__main__":
    unittest.main()
