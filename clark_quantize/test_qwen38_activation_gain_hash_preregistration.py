from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38ActivationGainHashPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_activation_gain_hash_residual_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_qwen38_27b_only(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])
        self.assertTrue(scope["calibration_and_validation_activations_disjoint"])

    def test_gain_and_control_receive_identical_training(self) -> None:
        training = self.prereg["training"]
        self.assertEqual(training["steps"], 12)
        self.assertEqual(training["candidate_and_control_training"], "identical")
        self.assertFalse(training["ternary_codes_and_scales_trainable"])
        self.assertFalse(training["routing_trainable"])

    def test_gate_requires_activation_gain_benefit_and_honest_rate(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreater(
            gate["minimum_relative_validation_error_reduction_vs_data_free_gain"],
            0,
        )
        self.assertGreater(
            gate["minimum_relative_validation_error_reduction_vs_trained_sign_only"],
            0,
        )
        self.assertLessEqual(gate["maximum_component_bits_per_weight"], 1.57)
        self.assertLessEqual(gate["maximum_kernel_identity_error"], 0.0005)

    def test_every_frozen_source_hash_matches(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertNotEqual(descriptor["sha256"], "PLACEHOLDER")
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                descriptor["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
