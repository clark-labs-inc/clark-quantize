import unittest

from decide_surface_group_confirmation import (
    choose_surface_options,
    passes_gate,
)


def row(name, group_size, gt_mean, gt_lower, edge_mean, teacher_mean, cost):
    def evidence(mean, lower):
        return {
            "mean": mean,
            "median": mean,
            "lower_95": lower,
            "upper_95": mean + 0.1,
            "wins": 8,
            "ties": 0,
            "samples": 8,
        }

    return {
        "name": name,
        "paired_evidence": {
            "psnr_vs_ground_truth": evidence(gt_mean, gt_lower),
            "edge_psnr_vs_ground_truth": evidence(edge_mean, edge_mean - 0.1),
            "psnr_vs_teacher": evidence(teacher_mean, teacher_mean - 0.1),
        },
        "storage": {"scale_bytes_delta": cost},
        "candidate_group_size": group_size,
    }


class SurfaceGroupDecisionTests(unittest.TestCase):
    def test_gate_requires_positive_ground_truth_interval(self):
        passed, failures = passes_gate(
            row("surface", 2, 0.1, 0.0, 0.0, 0.0, 1000)
        )

        self.assertFalse(passed)
        self.assertIn("ground_truth_lower_95_not_positive", failures)

    def test_choice_uses_efficiency_and_budget(self):
        reports = {
            2: {"surfaces": [
                row("a", 2, 0.2, 0.1, 0.0, 0.0, 2000),
                row("b", 2, 0.2, 0.1, 0.0, 0.0, 2000),
            ]},
            6: {"surfaces": [
                row("a", 6, 0.15, 0.05, 0.0, 0.0, 500),
                row("b", 6, -0.1, -0.2, 0.0, 0.0, 500),
            ]},
        }

        selected, _audit, consumed = choose_surface_options(
            reports,
            ["a", "b"],
            2200,
        )

        self.assertEqual(
            [(item["name"], item["group_size"]) for item in selected],
            [("a", 6)],
        )
        self.assertEqual(consumed, 500)


if __name__ == "__main__":
    unittest.main()
