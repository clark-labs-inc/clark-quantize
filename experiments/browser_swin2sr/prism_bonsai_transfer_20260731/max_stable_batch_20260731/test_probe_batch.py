import unittest
from types import SimpleNamespace

from probe_batch import percentile, update_peak, validate_args


class BatchProbeTests(unittest.TestCase):
    def test_peak_tracks_process_allocations(self):
        peak = {}
        update_peak(peak, {
            "current_allocated_bytes": 10,
            "driver_allocated_bytes": 20,
            "recommended_max_bytes": 100,
        })
        update_peak(peak, {
            "current_allocated_bytes": 30,
            "driver_allocated_bytes": 15,
            "recommended_max_bytes": 100,
        })
        self.assertEqual(peak["current_allocated_bytes"], 30)
        self.assertEqual(peak["driver_allocated_bytes"], 20)

    def test_percentile_is_deterministic(self):
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.50), 3.0)

    def test_invalid_resource_contract_is_rejected(self):
        args = SimpleNamespace(
            batch_size=0,
            warmup_steps=3,
            measure_steps=8,
            learning_rate=4e-6,
            memory_fraction=0.80,
            headroom_fraction=0.25,
        )
        with self.assertRaises(ValueError):
            validate_args(args)
        args.batch_size = 8
        args.memory_fraction = 0
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
