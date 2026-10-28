from __future__ import annotations

import unittest

import numpy as np

from clark_quantize.qwen38_full_scale_codebook import (
    fit_scale_codebook_chunked,
    scale_error_metrics,
)
from clark_quantize.qwen38_row_scale_codebook import (
    fit_row_normalized_log_codebook,
    reconstruct_row_normalized_scales,
)
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles


class Qwen38RowScaleCodebookTests(unittest.TestCase):
    def test_row_normalization_handles_cross_row_dynamic_range(self) -> None:
        rng = np.random.default_rng(20260817)
        anchors = np.geomspace(2**-10, 2**5, 96, dtype=np.float32)
        ratios = rng.lognormal(mean=0.0, sigma=0.16, size=(96, 40)).astype(np.float32)
        scales = anchors[:, None] * ratios
        global_fit = fit_scale_codebook_chunked(scales, entries=16, chunk_size=257)
        row_fit = fit_row_normalized_log_codebook(scales, entries=16, chunk_rows=7)
        global_error = scale_error_metrics(scales, global_fit)["normalized_rmse"]
        row_error = scale_error_metrics(scales, row_fit)["normalized_rmse"]
        self.assertLess(row_error, 0.02)
        self.assertLess(row_error, global_error / 5)
        self.assertEqual(row_fit.storage_bytes, scales.size // 2 + 96 * 2 + 16 * 2)

    def test_fit_is_deterministic_and_indices_roundtrip(self) -> None:
        rng = np.random.default_rng(3)
        scales = rng.lognormal(-2.0, 0.3, size=(17, 13)).astype(np.float16)
        first = fit_row_normalized_log_codebook(scales, chunk_rows=3)
        second = fit_row_normalized_log_codebook(scales, chunk_rows=3)
        np.testing.assert_array_equal(first.anchors, second.anchors)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.reconstructed, second.reconstructed)
        packed = pack_nibbles(first.indices)
        restored = unpack_nibbles(packed, first.indices.size).reshape(first.indices.shape)
        np.testing.assert_array_equal(restored, first.indices)
        np.testing.assert_array_equal(
            first.reconstructed,
            reconstruct_row_normalized_scales(
                first.anchors, first.values, restored
            ),
        )

    def test_zero_rows_and_zero_entries_remain_exact(self) -> None:
        scales = np.asarray(
            [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.2, 0.15]], dtype=np.float32
        )
        fitted = fit_row_normalized_log_codebook(scales, entries=4, chunk_rows=1)
        self.assertEqual(float(fitted.anchors[0]), 0.0)
        np.testing.assert_array_equal(fitted.indices[0], 0)
        np.testing.assert_array_equal(fitted.reconstructed[scales == 0], 0.0)

    def test_all_zero_tensor_is_a_valid_exact_payload(self) -> None:
        scales = np.zeros((3, 5), dtype=np.float32)
        fitted = fit_row_normalized_log_codebook(scales, entries=4, chunk_rows=2)
        np.testing.assert_array_equal(fitted.anchors, 0)
        np.testing.assert_array_equal(fitted.indices, 0)
        np.testing.assert_array_equal(fitted.reconstructed, 0)
        np.testing.assert_array_equal(fitted.values[1:], 1)

    def test_invalid_inputs_fail_closed(self) -> None:
        for invalid in (
            np.asarray([], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([[1.0, -1.0]], dtype=np.float32),
            np.asarray([[np.nan]], dtype=np.float32),
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                fit_row_normalized_log_codebook(invalid)
        with self.assertRaisesRegex(ValueError, "invalid"):
            reconstruct_row_normalized_scales(
                np.ones(2, dtype=np.float16),
                np.ones(2, dtype=np.float16),
                np.zeros((3, 1), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
