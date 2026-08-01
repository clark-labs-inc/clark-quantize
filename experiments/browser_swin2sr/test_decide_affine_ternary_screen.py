import unittest

from decide_affine_ternary_screen import decide


def study():
    return {
        "development_samples": 100,
        "pass_gate": {
            "ground_truth_delta_lower95_minimum": 0.0,
            "edge_delta_mean_minimum": 0.0,
            "teacher_delta_mean_minimum": 0.0,
        },
    }


def comparison(gt_lower=0.01, edge=0.01, teacher=0.01):
    def metric(mean, interval):
        return {
            "samples": 100,
            "candidate_minus_reference_mean": mean,
            "paired_bootstrap_95ci": interval,
        }
    return {"metrics": {
        "psnr_vs_ground_truth": metric(0.02, [gt_lower, 0.03]),
        "edge_psnr_vs_ground_truth": metric(edge, [-0.01, 0.03]),
        "psnr_vs_teacher": metric(teacher, [-0.01, 0.03]),
    }}


class AffineScreenDecisionTests(unittest.TestCase):
    def test_all_three_gates_are_required(self):
        self.assertTrue(decide(study(), comparison())["passed"])
        self.assertFalse(decide(study(), comparison(gt_lower=0.0))["passed"])
        self.assertFalse(decide(study(), comparison(edge=-0.001))["passed"])
        self.assertFalse(decide(study(), comparison(teacher=-0.001))["passed"])

    def test_pass_never_authorizes_promotion_or_final(self):
        result = decide(study(), comparison())
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["final_heldout_access_authorized"])
        self.assertFalse(result["runtime_implementation_authorized"])
        self.assertFalse(result["model_promotion_authorized"])

    def test_sample_mismatch_fails_closed(self):
        payload = comparison()
        payload["metrics"]["psnr_vs_teacher"]["samples"] = 99
        with self.assertRaisesRegex(ValueError, "sample count"):
            decide(study(), payload)
