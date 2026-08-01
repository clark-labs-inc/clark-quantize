import unittest

from scale_allocation_interactions import (
    advancement_decision,
    choice_storage,
    precision_map_storage,
)


class ScaleAllocationInteractionTests(unittest.TestCase):
    def test_choice_storage_tracks_calibration_and_physical_alignment(self):
        receipt = choice_storage({
            "label": "q4",
            "groups": 10,
            "centers": 3,
            "storage_bytes": 12,
        })
        self.assertEqual(receipt["payload_bytes"], 9)
        self.assertEqual(receipt["calibration_storage_bytes"], 12)
        self.assertEqual(receipt["physical_aligned_span_bytes"], 12)

    def test_map_requires_exact_surface_coverage(self):
        report = {
            "surfaces": [{
                "surface": "a",
                "choices": [{
                    "label": "fp16",
                    "groups": 2,
                    "centers": 2,
                    "storage_bytes": 4,
                }],
            }]
        }
        with self.assertRaises(ValueError):
            precision_map_storage(report, {})

    def test_smallest_candidate_must_pass_both_shards(self):
        candidates = {
            "small": {
                "budget_bytes": 10,
                "uniform_control": "q2",
                "storage": {
                    "calibration_storage_bytes": 10,
                    "physical_aligned_span_bytes": 10,
                },
            },
            "large": {
                "budget_bytes": 20,
                "uniform_control": "q4",
                "storage": {
                    "calibration_storage_bytes": 20,
                    "physical_aligned_span_bytes": 20,
                },
            },
        }
        shards = {
            "allocation_calibration": {"states": {
                "small": {"fragility": 0.8},
                "large": {"fragility": 0.5},
                "q2": {"fragility": 1.0},
                "q4": {"fragility": 1.0},
            }},
            "disjoint_confirmation": {"states": {
                "small": {"fragility": 0.95},
                "large": {"fragility": 0.5},
                "q2": {"fragility": 1.0},
                "q4": {"fragility": 1.0},
            }},
        }
        decision = advancement_decision(
            candidates, shards, minimum_relative_gain=0.10
        )
        self.assertFalse(decision["candidate_decisions"]["small"]["passes"])
        self.assertEqual(decision["selected_candidate"], "large")


if __name__ == "__main__":
    unittest.main()
