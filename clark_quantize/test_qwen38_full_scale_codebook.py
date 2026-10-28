from __future__ import annotations

import unittest

import numpy as np

from clark_quantize.qwen38_full_scale_codebook import (
    aggregate_scale_records,
    fit_scale_codebook_chunked,
    scale_error_metrics,
)
from clark_quantize.scale_codebook import fit_scale_codebook


class Qwen38FullScaleCodebookTests(unittest.TestCase):
    def test_chunked_fit_matches_reference_after_fp16_freeze(self) -> None:
        rng = np.random.default_rng(20260817)
        scales = np.concatenate(
            [np.zeros(13, dtype=np.float32), rng.lognormal(-2.0, 0.7, 1003)]
        ).astype(np.float32)
        reference = fit_scale_codebook(scales, entries=16, iterations=50)
        chunked = fit_scale_codebook_chunked(
            scales, entries=16, iterations=50, chunk_size=37
        )
        np.testing.assert_array_equal(chunked.values, reference.values)
        np.testing.assert_array_equal(chunked.indices, reference.indices)
        np.testing.assert_array_equal(chunked.reconstructed, reference.reconstructed)

    def test_metrics_and_aggregate_preserve_exact_denominators(self) -> None:
        first = np.asarray([0.0, 0.1, 0.2, 0.4], dtype=np.float32)
        second = np.asarray([0.05, 0.07, 0.13], dtype=np.float32)
        records = []
        for name, scales in (("a.weight", first), ("b.weight", second)):
            fitted = fit_scale_codebook_chunked(scales, entries=4, chunk_size=2)
            records.append(
                {
                    "tensor": name,
                    "scale_count": scales.size,
                    "metrics": scale_error_metrics(scales, fitted),
                }
            )
        aggregate = aggregate_scale_records(records)
        self.assertEqual(aggregate.tensors, 2)
        self.assertEqual(aggregate.scales, 7)
        expected = sum(record["metrics"]["sum_error_squared"] for record in records)
        self.assertEqual(aggregate.error_squared, expected)
        self.assertGreaterEqual(aggregate.global_nrmse, 0.0)

    def test_invalid_or_duplicate_records_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_scale_records([])
        record = {
            "tensor": "duplicate",
            "scale_count": 1,
            "metrics": {
                "sum_source_squared": 1.0,
                "sum_error_squared": 0.0,
                "normalized_rmse": 0.0,
                "maximum_absolute_error": 0.0,
                "p99_relative_error_nonzero_scales": 0.0,
            },
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_scale_records([record, dict(record)])


if __name__ == "__main__":
    unittest.main()
