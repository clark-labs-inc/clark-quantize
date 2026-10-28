from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_activation_gain_hash_residual import (
    activation_weighted_gain_hash_coefficients,
    output_error_stats,
    relative_error_reduction,
    train_int4_activation_gain_hash_coefficients,
)
from clark_quantize.qwen38_gain_hash_residual import (
    expand_gain_hash,
    gain_hash_sketch_rows,
    learn_gain_hash_routing,
)


class Qwen38ActivationGainHashResidualTests(unittest.TestCase):
    @staticmethod
    def problem(seed: int = 3811):
        generator = torch.Generator().manual_seed(seed)
        residual = torch.randn(8, 64, generator=generator)
        projection = learn_gain_hash_routing(
            residual, sketch_dim=8, gain_bits=3, iterations=3
        ).projection
        return generator, residual, projection

    def test_activation_weighted_solve_improves_anisotropic_outputs(self) -> None:
        generator, residual, projection = self.problem()
        inputs = torch.randn(256, 64, generator=generator)
        inputs[:, :8] *= 10.0
        data_free = torch.linalg.lstsq(
            gain_hash_sketch_rows(inputs, projection), inputs @ residual.T
        ).solution.T.contiguous()
        perturbed = data_free + torch.randn(
            data_free.shape, generator=generator
        ) * 0.05
        fitted = activation_weighted_gain_hash_coefficients(
            residual,
            inputs,
            projection,
            reference_coefficients=perturbed,
            relative_ridge=1e-5,
        )
        baseline = output_error_stats(residual, perturbed, inputs, projection)
        candidate = output_error_stats(residual, fitted, inputs, projection)
        self.assertGreater(relative_error_reduction(candidate, baseline), 0.01)

    def test_underdetermined_solve_is_finite(self) -> None:
        generator, residual, projection = self.problem(3812)
        inputs = torch.randn(4, 64, generator=generator)
        reference = torch.randn(8, 8, generator=generator)
        fitted = activation_weighted_gain_hash_coefficients(
            residual,
            inputs,
            projection,
            reference_coefficients=reference,
        )
        self.assertEqual(fitted.shape, reference.shape)
        self.assertTrue(bool(torch.isfinite(fitted).all()))

    def test_short_hard_int4_training_keeps_canonical_best_export(self) -> None:
        generator, residual, projection = self.problem(3813)
        inputs = torch.randn(96, 64, generator=generator)
        initial = activation_weighted_gain_hash_coefficients(
            residual, inputs, projection
        )
        trained = train_int4_activation_gain_hash_coefficients(
            residual,
            inputs,
            projection,
            initial,
            reference_coefficients=initial,
            steps=4,
        )
        self.assertEqual(trained.codes.dtype, torch.int8)
        self.assertEqual(trained.scales.dtype, torch.float16)
        self.assertEqual(len(trained.calibration_history), 4)
        torch.testing.assert_close(
            trained.coefficients,
            trained.codes.float() * trained.scales.float()[:, None],
            rtol=0,
            atol=0,
        )

    def test_implicit_runtime_matches_expanded_weight(self) -> None:
        generator, _, projection = self.problem(3814)
        coefficients = torch.randn(7, 8, generator=generator)
        inputs = torch.randn(11, 64, generator=generator)
        implicit = gain_hash_sketch_rows(inputs, projection) @ coefficients.T
        explicit = inputs @ expand_gain_hash(coefficients, projection).T
        torch.testing.assert_close(implicit, explicit, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
