from __future__ import annotations

import unittest

import torch

from clark_quantize.ternary import ternarize, ternary_codes_scales


class TernaryExportTests(unittest.TestCase):
    def test_export_codes_reconstruct_existing_quantizer(self) -> None:
        generator = torch.Generator().manual_seed(20260815)
        weight = torch.randn((9, 259), generator=generator)
        for method in ("optimal", "absmean"):
            dequantized, _ = ternarize(weight, group=128, method=method)
            codes, scales = ternary_codes_scales(weight, group=128, method=method)
            padded = torch.nn.functional.pad(codes, (0, 384 - codes.shape[1]))
            reconstructed = (
                padded.reshape(9, 3, 128).float() * scales.float().unsqueeze(-1)
            ).reshape(9, -1)[:, :259]
            torch.testing.assert_close(reconstructed, dequantized, rtol=5e-4, atol=5e-4)
            self.assertEqual(codes.dtype, torch.int8)
            self.assertEqual(scales.dtype, torch.float16)

    def test_export_rejects_non_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            ternary_codes_scales(torch.zeros(8))


if __name__ == "__main__":
    unittest.main()
