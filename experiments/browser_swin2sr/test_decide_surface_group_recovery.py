import unittest

from decide_surface_group_recovery import decide


def sample(identity, gt, edge, teacher, source="rico"):
    return {
        "sha256": identity,
        "source": source,
        "psnr_vs_ground_truth": gt,
        "edge_psnr_vs_ground_truth": edge,
        "psnr_vs_teacher": teacher,
    }


class SurfaceGroupRecoveryDecisionTests(unittest.TestCase):
    def test_decision_selects_refined_when_every_gate_passes(self):
        study = {
            "selected_scale_bytes_delta": 4000,
            "decision_rule": {
                "bootstrap_seed": 7,
                "bootstrap_samples": 100,
                "ground_truth_lower_95_min_db": 0.0,
                "edge_mean_min_db": -0.1,
                "teacher_mean_min_db": -0.1,
                "every_source_mean_min_db": -0.1,
                "scale_byte_budget": 65536,
            },
        }
        control = {"samples": [
            sample(str(index), 10.0, 20.0, 30.0)
            for index in range(3040)
        ]}
        candidate = {"samples": [
            sample(str(index), 10.1, 20.1, 30.1)
            for index in range(3040)
        ]}

        result = decide(study, control, candidate)

        self.assertEqual(result["selected_arm"], "refined")
        self.assertTrue(all(result["gates"].values()))


if __name__ == "__main__":
    unittest.main()
