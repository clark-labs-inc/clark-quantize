import unittest

from decide_surface_group_confirmation import passes_gate


class BroadSurfaceGroupDecisionTests(unittest.TestCase):
    def test_shared_gate_accepts_positive_interval_and_guards(self):
        surface = {
            "paired_evidence": {
                "psnr_vs_ground_truth": {"lower_95": 0.01},
                "edge_psnr_vs_ground_truth": {"mean": 0.0},
                "psnr_vs_teacher": {"mean": -0.05},
            },
        }

        passed, failures = passes_gate(surface)

        self.assertTrue(passed)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
