import unittest

from evaluate_restoration_islands import paired_deltas, restoration_groups


class RestorationIslandTests(unittest.TestCase):
    def test_groups_are_budgeted_and_semantic(self):
        ranking = [
            {
                "name": "encoder.stages.3.layers.5.attention.self.value",
                "weight_values": 10,
            },
            {
                "name": "encoder.stages.3.layers.1.attention.self.key",
                "weight_values": 10,
            },
            {
                "name": "encoder.stages.2.layers.1.attention.self.value",
                "weight_values": 380,
            },
        ]
        groups = restoration_groups(ranking)
        self.assertEqual(groups["top1_local_error"], [ranking[0]["name"]])
        self.assertEqual(groups["stage3_values"], [ranking[0]["name"]])
        self.assertEqual(
            groups["stage3_qkv"],
            [ranking[0]["name"], ranking[1]["name"]],
        )
        self.assertEqual(
            groups["top5pct_local_error"],
            [ranking[0]["name"], ranking[1]["name"]],
        )

    def test_paired_delta_preserves_source_regression(self):
        reference = {
            "a": {
                "psnr_vs_ground_truth": 10.0,
                "edge_psnr_vs_ground_truth": 20.0,
                "psnr_vs_teacher": 30.0,
            },
            "b": {
                "psnr_vs_ground_truth": 10.0,
                "edge_psnr_vs_ground_truth": 20.0,
                "psnr_vs_teacher": 30.0,
            },
        }
        candidate = [
            {
                "sha256": "a",
                "source": "rico",
                "psnr_vs_ground_truth": 9.0,
                "edge_psnr_vs_ground_truth": 21.0,
                "psnr_vs_teacher": 31.0,
            },
            {
                "sha256": "b",
                "source": "flores",
                "psnr_vs_ground_truth": 11.5,
                "edge_psnr_vs_ground_truth": 22.0,
                "psnr_vs_teacher": 33.0,
            },
        ]
        result = paired_deltas(candidate, reference)
        self.assertAlmostEqual(result["psnr_vs_ground_truth"], 0.25)
        self.assertAlmostEqual(result["by_source_psnr"]["rico"], -1.0)
        self.assertAlmostEqual(result["by_source_psnr"]["flores"], 1.5)


if __name__ == "__main__":
    unittest.main()
