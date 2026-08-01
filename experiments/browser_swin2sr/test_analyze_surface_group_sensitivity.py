import unittest

import numpy as np

from analyze_surface_group_sensitivity import paired_metric_evidence
from probe_sensitive_io_group_sizes import storage_receipt


class SurfaceGroupSensitivityTests(unittest.TestCase):
    def test_paired_metric_evidence_uses_sample_deltas(self):
        baseline = [
            {
                "psnr_vs_ground_truth": 10.0,
                "edge_psnr_vs_ground_truth": 20.0,
                "psnr_vs_teacher": 30.0,
            },
            {
                "psnr_vs_ground_truth": 12.0,
                "edge_psnr_vs_ground_truth": 22.0,
                "psnr_vs_teacher": 32.0,
            },
        ]
        candidate = [
            {
                "psnr_vs_ground_truth": 11.0,
                "edge_psnr_vs_ground_truth": 19.0,
                "psnr_vs_teacher": 30.5,
            },
            {
                "psnr_vs_ground_truth": 13.0,
                "edge_psnr_vs_ground_truth": 21.0,
                "psnr_vs_teacher": 32.5,
            },
        ]
        indices = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]])

        evidence = paired_metric_evidence(candidate, baseline, indices)

        self.assertEqual(evidence["psnr_vs_ground_truth"]["mean"], 1.0)
        self.assertEqual(evidence["psnr_vs_ground_truth"]["wins"], 2)
        self.assertEqual(evidence["edge_psnr_vs_ground_truth"]["mean"], -1.0)
        self.assertEqual(evidence["psnr_vs_teacher"]["samples"], 2)

    def test_group_two_storage_delta_is_exact(self):
        receipt = storage_receipt(
            values=3600,
            group_size=2,
            total_quantized_values=977604,
            reference_group_size=60,
        )

        self.assertEqual(receipt["scale_count"], 1800)
        self.assertGreater(
            receipt["whole_model_bit_delta_vs_reference"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
