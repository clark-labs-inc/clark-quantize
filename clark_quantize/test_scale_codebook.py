from __future__ import annotations

import unittest

import numpy as np

from clark_quantize.scale_codebook import (
    fit_scale_codebook,
    pack_nibbles,
    unpack_nibbles,
)


class ScaleCodebookTests(unittest.TestCase):
    def test_at_most_fifteen_positive_values_are_exact_after_fp16_freeze(self) -> None:
        source = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0], dtype=np.float16)
        result = fit_scale_codebook(source, entries=16)
        np.testing.assert_array_equal(result.reconstructed, source.astype(np.float32))
        self.assertEqual(result.weighted_mse, 0.0)
        self.assertEqual(result.maximum_absolute_error, 0.0)

    def test_weighted_fit_is_deterministic_and_finite(self) -> None:
        source = np.linspace(0.001, 2.0, 1000, dtype=np.float32) ** 2
        weights = np.arange(1, 1001, dtype=np.float64)
        first = fit_scale_codebook(source, weights, entries=16)
        second = fit_scale_codebook(source.copy(), weights.copy(), entries=16)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.indices, second.indices)
        self.assertTrue(np.isfinite(first.weighted_mse))
        self.assertGreater(first.maximum_absolute_error, 0.0)
        self.assertLessEqual(int(first.indices.max()), 15)

    def test_nibble_round_trip_and_padding_validation(self) -> None:
        indices = np.asarray([0, 15, 2, 7, 9], dtype=np.uint8)
        payload = pack_nibbles(indices)
        self.assertEqual(len(payload), 3)
        np.testing.assert_array_equal(unpack_nibbles(payload, len(indices)), indices)
        damaged = bytearray(payload)
        damaged[-1] |= 0xF0
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_nibbles(bytes(damaged), len(indices))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_scale_codebook(np.asarray([], dtype=np.float32))
        with self.assertRaises(ValueError):
            fit_scale_codebook(np.asarray([np.nan], dtype=np.float32))
        with self.assertRaises(ValueError):
            fit_scale_codebook(np.asarray([1.0]), np.asarray([0.0]))
        with self.assertRaises(ValueError):
            pack_nibbles(np.asarray([16], dtype=np.uint8))
        with self.assertRaises(ValueError):
            unpack_nibbles(b"", 1)


if __name__ == "__main__":
    unittest.main()
