import unittest

from analyze_surface_restore_sensitivity import (
    metric_delta,
    stratified_records,
)


class SurfaceRestoreSensitivityTests(unittest.TestCase):
    def test_stratified_records_balance_sources(self):
        records = [
            {
                "sha256": f"rico-{index}",
                "source": "rico",
                "split": "test",
            }
            for index in range(10)
        ] + [
            {
                "sha256": f"text-{index}",
                "source": "text",
                "script": f"script-{index % 3}",
            }
            for index in range(10)
        ]

        selected = stratified_records(records, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(
            sum(record["source"] == "rico" for record in selected),
            4,
        )
        self.assertEqual(
            sum(record["source"] == "text" for record in selected),
            4,
        )
        self.assertEqual(
            len({
                record.get("script")
                for record in selected
                if record["source"] == "text"
            }),
            3,
        )

    def test_metric_delta_uses_candidate_minus_baseline(self):
        baseline = {
            "psnr_vs_ground_truth": 30.0,
            "edge_psnr_vs_ground_truth": 31.0,
            "psnr_vs_teacher": 32.0,
        }
        candidate = {
            "psnr_vs_ground_truth": 30.5,
            "edge_psnr_vs_ground_truth": 30.75,
            "psnr_vs_teacher": 33.0,
        }

        self.assertEqual(metric_delta(candidate, baseline), {
            "psnr_vs_ground_truth": 0.5,
            "edge_psnr_vs_ground_truth": -0.25,
            "psnr_vs_teacher": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
