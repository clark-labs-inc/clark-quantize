import unittest

from evaluate_surface_group_map import joint_gate


def evidence(gt_lower=0.01, edge_mean=0.0, teacher_mean=0.0):
    return {
        "psnr_vs_ground_truth": {"lower_95": gt_lower, "mean": 0.1},
        "edge_psnr_vs_ground_truth": {"lower_95": -0.1, "mean": edge_mean},
        "psnr_vs_teacher": {"lower_95": -0.1, "mean": teacher_mean},
    }


class SurfaceGroupMapTests(unittest.TestCase):
    def test_joint_gate_accepts_all_frozen_requirements(self):
        passed, failures = joint_gate(evidence(), 60000, 65536)

        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_joint_gate_reports_quality_and_size_failures(self):
        passed, failures = joint_gate(
            evidence(gt_lower=0.0, edge_mean=-0.03, teacher_mean=-0.2),
            70000,
            65536,
        )

        self.assertFalse(passed)
        self.assertEqual(len(failures), 4)


if __name__ == "__main__":
    unittest.main()
