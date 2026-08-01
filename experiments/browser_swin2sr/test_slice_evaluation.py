import unittest

from slice_evaluation import slice_report


def sample(index):
    return {
        "sha256": f"{index:064x}",
        "source": "fixture",
        "split": "test",
        "language": "eng_Latn",
        "script": "Latn",
        "psnr_vs_ground_truth": 30.0 + index,
        "edge_psnr_vs_ground_truth": 31.0 + index,
        "psnr_vs_teacher": 40.0 + index,
        "bicubic_psnr_vs_ground_truth": 25.0,
        "teacher_psnr_vs_ground_truth": 32.0,
        "teacher_seconds": 0.1,
        "student_seconds": 0.2,
    }


class SliceEvaluationTests(unittest.TestCase):
    def test_recomputes_summary_for_prefix(self):
        report = {"state": "fixture", "samples": [sample(0), sample(1), sample(2)]}

        result = slice_report(report, 2)

        self.assertEqual(result["slice"]["samples"], 2)
        self.assertEqual(result["overall"]["samples"], 2)
        self.assertEqual(result["overall"]["psnr_vs_ground_truth"]["mean"], 30.5)
        self.assertEqual([row["sha256"] for row in result["samples"]], [
            f"{0:064x}",
            f"{1:064x}",
        ])

    def test_rejects_oversized_slice(self):
        with self.assertRaisesRegex(ValueError, "containing 1"):
            slice_report({"samples": [sample(0)]}, 2)


if __name__ == "__main__":
    unittest.main()
