import unittest

import torch

from affine_ternary import fit_affine_ternary_codes, quantize_affine_ternary_mse
from clark_air.quant.optimal_ternary import quantize_ternary_mse


class AffineTernaryTests(unittest.TestCase):
    def test_fit_never_worse_than_symmetric_initialization(self):
        generator = torch.Generator().manual_seed(20260731)
        weight = torch.randn(7, 13, generator=generator) * 0.7 + 0.35
        symmetric = quantize_ternary_mse(weight, group_size=16).dequantize()
        affine = quantize_affine_ternary_mse(weight, group_size=16).dequantize()
        symmetric_error = (weight - symmetric).square().sum()
        affine_error = (weight - affine).square().sum()
        self.assertLessEqual(affine_error, symmetric_error + 1e-6)

    def test_fit_recovers_biased_three_level_group(self):
        weight = torch.tensor([9.0, 10.0, 11.0, 9.0, 10.0, 11.0])
        affine = quantize_affine_ternary_mse(weight, group_size=6)
        torch.testing.assert_close(affine.dequantize(), weight)
        torch.testing.assert_close(affine.shifts, torch.tensor([10.0]))
        torch.testing.assert_close(affine.scales, torch.tensor([1.0]))

    def test_fixed_codes_preserve_map_and_reduce_symmetric_error(self):
        weight = torch.tensor([9.0, 10.0, 11.0, 9.0, 10.0, 11.0])
        symmetric = quantize_ternary_mse(weight, group_size=6)
        affine = fit_affine_ternary_codes(
            weight,
            symmetric.codes,
            symmetric.group_size,
            symmetric.pad,
        )
        torch.testing.assert_close(affine.codes, symmetric.codes)
        self.assertLessEqual(
            (weight - affine.dequantize()).square().sum(),
            (weight - symmetric.dequantize()).square().sum() + 1e-6,
        )

    def test_fit_preserves_shape_with_padding(self):
        weight = torch.tensor([[0.1, -0.2, 0.4], [0.8, -1.2, 0.3]])
        affine = quantize_affine_ternary_mse(weight, group_size=4)
        self.assertEqual(affine.pad, 2)
        self.assertEqual(affine.dequantize().shape, weight.shape)
        self.assertEqual(affine.codes.numel(), 8)

    def test_fit_rejects_invalid_controls(self):
        weight = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "group_size"):
            quantize_affine_ternary_mse(weight, group_size=0)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            quantize_affine_ternary_mse(weight, max_iterations=0)
        with self.assertRaisesRegex(ValueError, "code count"):
            fit_affine_ternary_codes(weight, torch.zeros(3), 4, 0)
