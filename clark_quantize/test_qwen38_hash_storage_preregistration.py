from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38HashStoragePreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_hash_storage_budget_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_and_inventory_are_complete_qwen38(self) -> None:
        self.assertEqual(self.prereg["scope"]["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(self.prereg["scope"]["qwen_4b_compute_forbidden"])
        self.assertEqual(self.prereg["expected_inventory"]["tensor_count"], 1199)
        self.assertEqual(self.prereg["expected_inventory"]["total_parameters"], 27781427952)

    def test_gate_targets_base3_limit_without_dropping_tensors(self) -> None:
        gate = self.prereg["gate"]
        self.assertLessEqual(gate["maximum_complete_bundle_bits_per_total_parameter"], 1.585)
        self.assertGreaterEqual(gate["minimum_fixed_support_parameter_fraction"], 0.995)
        self.assertTrue(self.prereg["eligibility"]["no_tensor_excluded_from_denominator"])

    def test_claim_boundary_remains_structural(self) -> None:
        boundary = self.prereg["scope"]["claim_boundary"]
        self.assertIn("not a complete exported artifact", boundary)
        self.assertIn("whole-model fit", boundary)
        self.assertIn("scale-quality", boundary)

    def test_frozen_sources_match(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
