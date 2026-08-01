import math
import unittest

import torch

from qat_objective import pixel_loss, stabilized_log_mse


class StabilizedLogMseTests(unittest.TestCase):
    def test_matches_rmse_scale_at_the_frozen_floor(self):
        floor = 4e-4
        expected = torch.zeros(2, 1, 2, 2)
        actual = torch.full_like(expected, math.sqrt(floor))

        loss = stabilized_log_mse(actual, expected, floor)

        self.assertAlmostEqual(loss.item(), math.sqrt(floor), delta=1e-7)

    def test_reciprocal_error_scales_receive_balanced_gradients(self):
        floor = 4e-4
        expected = torch.zeros(2, 1, 2, 2)
        actual = torch.stack([
            torch.full((1, 2, 2), math.sqrt(floor / 4)),
            torch.full((1, 2, 2), math.sqrt(floor * 4)),
        ]).requires_grad_(True)

        stabilized_log_mse(actual, expected, floor).backward()
        gradient_norms = actual.grad.flatten(1).norm(dim=1)

        torch.testing.assert_close(gradient_norms[0], gradient_norms[1])

    def test_zero_error_is_finite_with_zero_gradient(self):
        actual = torch.zeros(1, 1, 2, 2, requires_grad=True)
        expected = torch.zeros_like(actual)

        loss = stabilized_log_mse(actual, expected, 1e-4)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(actual.grad, torch.zeros_like(actual))

    def test_pixel_loss_rejects_unknown_objective(self):
        values = torch.zeros(1, 1, 1, 1)

        with self.assertRaisesRegex(ValueError, "unknown pixel objective"):
            pixel_loss(
                values,
                values,
                objective="unknown",
                mse_floor=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
