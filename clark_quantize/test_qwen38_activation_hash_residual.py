from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_activation_hash_residual import (
    activation_weighted_hash_coefficients,
    output_error_stats,
    relative_error_reduction,
    train_int4_activation_hash_coefficients,
)
from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    least_squares_hash_coefficients,
    sketch_rows,
    sparse_jl_projection,
)


class Qwen38ActivationHashResidualTests(unittest.TestCase):
    def test_activation_weighted_solve_improves_anisotropic_output_error(self) -> None:
        generator = torch.Generator().manual_seed(3801)
        residual = torch.randn(6, 64, generator=generator)
        inputs = torch.randn(256, 64, generator=generator)
        inputs[:, :8] *= 12.0
        projection = sparse_jl_projection(64, 16, hashes_per_input=4, seed=27)
        data_free = least_squares_hash_coefficients(residual, projection)
        fitted = activation_weighted_hash_coefficients(
            residual,
            inputs,
            projection,
            reference_coefficients=data_free,
            relative_ridge=1e-5,
        )
        baseline = output_error_stats(residual, data_free, inputs, projection)
        candidate = output_error_stats(residual, fitted, inputs, projection)
        self.assertGreater(relative_error_reduction(candidate, baseline), 0.01)

    def test_underdetermined_solve_is_finite_and_reference_bounded(self) -> None:
        generator = torch.Generator().manual_seed(3802)
        residual = torch.randn(4, 96, generator=generator)
        inputs = torch.randn(12, 96, generator=generator)
        projection = sparse_jl_projection(96, 32, hashes_per_input=2, seed=28)
        reference = least_squares_hash_coefficients(residual, projection)
        fitted = activation_weighted_hash_coefficients(
            residual,
            inputs,
            projection,
            reference_coefficients=reference,
        )
        self.assertEqual(fitted.shape, (4, 32))
        self.assertTrue(bool(torch.isfinite(fitted).all()))

    def test_short_int4_training_keeps_canonical_best_export(self) -> None:
        generator = torch.Generator().manual_seed(3803)
        residual = torch.randn(5, 64, generator=generator)
        inputs = torch.randn(96, 64, generator=generator)
        projection = sparse_jl_projection(64, 8, hashes_per_input=2, seed=29)
        reference = least_squares_hash_coefficients(residual, projection)
        fitted = activation_weighted_hash_coefficients(
            residual,
            inputs,
            projection,
            reference_coefficients=reference,
        )
        trained = train_int4_activation_hash_coefficients(
            residual,
            inputs,
            projection,
            fitted,
            reference_coefficients=reference,
            steps=4,
        )
        self.assertEqual(trained.codes.dtype, torch.int8)
        self.assertEqual(trained.scales.dtype, torch.float16)
        self.assertTrue(bool(((trained.codes >= -7) & (trained.codes <= 7)).all()))
        torch.testing.assert_close(
            trained.coefficients,
            trained.codes.float() * trained.scales.float()[:, None],
            rtol=0,
            atol=0,
        )
        self.assertEqual(len(trained.calibration_history), 4)
        self.assertGreaterEqual(trained.best_step, 0)

    def test_runtime_algebra_matches_expanded_weight(self) -> None:
        generator = torch.Generator().manual_seed(3804)
        projection = sparse_jl_projection(64, 8, hashes_per_input=2, seed=30)
        coefficients = torch.randn(5, 8, generator=generator)
        inputs = torch.randn(9, 64, generator=generator)
        implicit = sketch_rows(inputs, projection) @ coefficients.T
        expanded = inputs @ expand_sketch_rows(coefficients, projection).T
        torch.testing.assert_close(implicit, expanded, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
