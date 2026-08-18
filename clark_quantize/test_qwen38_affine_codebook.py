from __future__ import annotations

import unittest

import numpy as np
import torch

from clark_quantize.qwen38_affine_butterfly import AffineTernary
from clark_quantize.qwen38_affine_codebook import (
    compress_affine_parameters,
    fit_signed_row_normalized_codebook,
    reconstruct_signed_row_normalized,
)
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles


class Qwen38AffineCodebookTests(unittest.TestCase):
    def test_signed_codebook_is_deterministic_and_preserves_exact_zero(self) -> None:
        values = np.asarray(
            [
                [-0.12, -0.03, 0.0, 0.08, 0.2],
                [-2.0, -0.8, 0.0, 0.7, 1.9],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        first = fit_signed_row_normalized_codebook(values, chunk_rows=1)
        second = fit_signed_row_normalized_codebook(values, chunk_rows=2)
        np.testing.assert_array_equal(first.anchors, second.anchors)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.reconstructed[values == 0], 0)
        np.testing.assert_array_equal(
            first.reconstructed,
            reconstruct_signed_row_normalized(
                first.anchors, first.values, first.indices
            ),
        )
        packed = pack_nibbles(first.indices)
        restored = unpack_nibbles(packed, first.indices.size).reshape(first.indices.shape)
        np.testing.assert_array_equal(restored, first.indices)

    def test_affine_compression_changes_both_parameters_to_frozen_decodes(self) -> None:
        payload = AffineTernary(
            codes=torch.tensor(
                [[-1, 0, 1, 1, 0, -1], [1, 1, 0, -1, -1, 0]], dtype=torch.int8
            ),
            mu=torch.tensor([[-0.5, 0.2], [0.0, 0.8]]),
            alpha=torch.tensor([[0.25, 0.125], [1.0, 0.5]]),
            group_size=3,
        )
        result = compress_affine_parameters(payload, chunk_rows=1)
        self.assertEqual(result.alpha.indices.shape, (2, 2))
        self.assertEqual(result.mu.indices.shape, (2, 2))
        self.assertEqual(result.parameter_storage_bytes, 2 * (4 + 32 + 2))
        np.testing.assert_array_equal(
            result.payload.alpha.numpy(), result.alpha.reconstructed
        )
        np.testing.assert_array_equal(result.payload.mu.numpy(), result.mu.reconstructed)
        self.assertTrue(torch.isfinite(result.payload.reconstruct()).all())

    def test_invalid_signed_payloads_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            fit_signed_row_normalized_codebook(np.asarray([1.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "invalid"):
            reconstruct_signed_row_normalized(
                np.ones(2, dtype=np.float16),
                np.zeros(16, dtype=np.float16),
                np.zeros((3, 1), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
