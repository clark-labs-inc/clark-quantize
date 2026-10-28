from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38GainHashPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_gain_hash_routing_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_only_qwen38_27b(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])
        self.assertTrue(scope["train_and_evaluation_rows_disjoint"])

    def test_gain_candidate_is_a_low_rate_sign_only_superset(self) -> None:
        candidate = self.prereg["candidate"]
        self.assertEqual(candidate["gain_bits"], 3)
        self.assertEqual(candidate["input_to_sketch_ratio"], 128)
        self.assertEqual(candidate["nonzero_per_group"], 64)
        self.assertIn("gain", candidate["representation"])

    def test_gate_requires_heldout_improvement_storage_and_identity(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreater(
            gate["minimum_aggregate_error_reduction_vs_sign_only"], 0
        )
        self.assertLessEqual(gate["maximum_component_bits_per_weight"], 1.57)
        self.assertLessEqual(gate["maximum_kernel_identity_error"], 0.0005)
        self.assertGreaterEqual(gate["minimum_used_gain_levels"], 2)

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
