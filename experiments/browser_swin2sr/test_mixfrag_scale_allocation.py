import unittest

from mixfrag_scale_allocation import (
    PrecisionChoice,
    aligned_scale_storage_bytes,
    allocate_precision,
)


class MixFragScaleAllocationTests(unittest.TestCase):
    def test_exact_scale_storage(self):
        self.assertEqual(aligned_scale_storage_bytes(10), 20)
        self.assertEqual(aligned_scale_storage_bytes(3), 8)
        self.assertEqual(aligned_scale_storage_bytes(10, bit_width=1), 8)
        self.assertEqual(
            aligned_scale_storage_bytes(10, bit_width=2, center_count=3),
            12,
        )

    def test_knapsack_selects_global_optimum(self):
        choices = {
            "a": [
                PrecisionChoice("q1", 2, 5.0),
                PrecisionChoice("q4", 5, 0.0),
            ],
            "b": [
                PrecisionChoice("q1", 2, 4.0),
                PrecisionChoice("q4", 4, 1.0),
            ],
        }
        allocation = allocate_precision(choices, 7)
        self.assertEqual(allocation.storage_bytes, 7)
        self.assertEqual(allocation.fragility, 4.0)
        self.assertEqual(allocation.choices["a"].label, "q4")
        self.assertEqual(allocation.choices["b"].label, "q1")

    def test_impossible_budget_is_rejected(self):
        choices = {"a": [PrecisionChoice("q1", 2, 1.0)]}
        with self.assertRaises(ValueError):
            allocate_precision(choices, 1)


if __name__ == "__main__":
    unittest.main()
