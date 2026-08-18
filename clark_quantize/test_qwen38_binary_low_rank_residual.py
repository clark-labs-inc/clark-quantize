from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_binary_low_rank_residual import (
    BinaryRightFactor,
    binary_low_rank_outputs,
    binary_right_values,
    expand_binary_low_rank,
    fit_binary_low_rank_residual,
    quantize_binary_rows,
    shared_binary_low_rank_storage_projection,
)


class Qwen38BinaryLowRankResidualTests(unittest.TestCase):
    def test_exported_factor_recovers_structured_residual(self) -> None:
        generator = torch.Generator().manual_seed(38631)
        rows, columns, rank = 48, 128, 4
        signs = torch.randint(
            0, 2, (rank, columns), generator=generator, dtype=torch.int8
        ) * 2 - 1
        right = BinaryRightFactor(
            signs=signs,
            scales=torch.tensor([0.02, 0.03, 0.04, 0.05], dtype=torch.float16),
        )
        left_codes = torch.randint(
            -7, 8, (rows, rank), generator=generator, dtype=torch.int8
        )
        left_scales = torch.linspace(0.01, 0.03, rows).to(torch.float16)
        source = (
            left_codes.float() * left_scales.float()[:, None]
        ) @ binary_right_values(right)
        fit = fit_binary_low_rank_residual(
            source, rank=rank, iterations=6, pca_iterations=3, seed=38632
        )
        reconstructed = expand_binary_low_rank(fit.left_coefficients, fit.right)
        reduction = 1.0 - float((source - reconstructed).square().sum()) / float(
            source.square().sum()
        )
        self.assertGreater(reduction, 0.65)
        self.assertGreaterEqual(fit.best_iteration, 1)
        self.assertLessEqual(
            min(fit.objective_history), fit.initial_squared_error
        )

    def test_implicit_output_matches_expanded_matrix(self) -> None:
        generator = torch.Generator().manual_seed(38633)
        inputs = torch.randn(7, 64, generator=generator)
        continuous_right = torch.randn(5, 64, generator=generator)
        right = quantize_binary_rows(continuous_right)
        left = torch.randn(11, 5, generator=generator)
        implicit = binary_low_rank_outputs(inputs, left, right)
        expanded = inputs @ expand_binary_low_rank(left, right).T
        torch.testing.assert_close(implicit, expanded, rtol=2e-5, atol=2e-5)

    def test_complete_gate_up_storage_remains_below_158_bits(self) -> None:
        ledger = shared_binary_low_rank_storage_projection(
            [(17408, 5120), (17408, 5120)],
            rank=40,
            group=128,
            nonzero_per_group=64,
        )
        self.assertEqual(ledger["group_width_bits"], 189)
        self.assertEqual(ledger["shared_right"]["binary_sign_bytes"], 25_600)
        self.assertLess(ledger["bits_per_weight"], 1.58)
        self.assertAlmostEqual(
            ledger["bits_per_weight"], 1.5464764763327206
        )

    def test_invalid_factor_is_rejected(self) -> None:
        bad = BinaryRightFactor(
            signs=torch.tensor([[0, 1]], dtype=torch.int8),
            scales=torch.ones(1, dtype=torch.float16),
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            binary_right_values(bad)


if __name__ == "__main__":
    unittest.main()
