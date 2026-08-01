import unittest

import numpy as np

from evaluate_residual_gain import select_gain
from residual_gain import balanced_folds, residual_gain_output


def evidence(mean, lower=0.01):
    return {
        "paired": {
            "psnr_vs_ground_truth": {"mean": mean, "lower_95": lower},
            "edge_psnr_vs_ground_truth": {"mean": 0.0},
            "psnr_vs_teacher": {"mean": 0.0},
        },
        "ground_truth_by_source": {"a": 0.01, "b": 0.01},
    }


RULE = {
    "ground_truth_lower_95_min_db": 0.0,
    "edge_mean_min_db": -0.05,
    "teacher_mean_min_db": -0.1,
    "every_source_ground_truth_mean_min_db": -0.1,
}


class ResidualGainTests(unittest.TestCase):
    def test_gain_one_preserves_student(self):
        student = np.asarray([[[0.2, 0.4, 0.8]]], dtype=np.float32)
        bicubic = np.asarray([[[0.3, 0.3, 0.3]]], dtype=np.float32)
        np.testing.assert_array_equal(
            residual_gain_output(student, bicubic, 1.0), student
        )

    def test_balanced_folds_are_disjoint_and_repeatable(self):
        records = [
            {"source": source, "sha256": f"{source}-{index}"}
            for source in ("a", "b") for index in range(8)
        ]
        first = balanced_folds(
            records, screen_per_source=2, confirmation_per_source=3, seed=7
        )
        second = balanced_folds(
            records, screen_per_source=2, confirmation_per_source=3, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 4)
        self.assertEqual(len(first[1]), 6)
        self.assertFalse(
            {row["sha256"] for row in first[0]}
            & {row["sha256"] for row in first[1]}
        )

    def test_selector_uses_best_passing_gain(self):
        screen = {
            "1.0": {"gain": 1.0, "evidence": evidence(0.0, 0.0)},
            "1.025": {"gain": 1.025, "evidence": evidence(0.03)},
            "1.05": {"gain": 1.05, "evidence": evidence(0.05)},
        }
        self.assertEqual(select_gain(screen, RULE), 1.05)


if __name__ == "__main__":
    unittest.main()
