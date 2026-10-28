from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38HalfSupportLedgerPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_half_support_complete_ledger_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_and_complete_denominator_are_frozen(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])
        self.assertEqual(scope["complete_parameter_denominator"], 27781427952)
        self.assertEqual(
            self.prereg["expected_inventory"]["total_parameters"],
            27781427952,
        )

    def test_gate_is_whole_bundle_below_one_point_58(self) -> None:
        gate = self.prereg["gate"]
        self.assertLessEqual(
            gate["maximum_effective_bits_per_total_parameter"], 1.58
        )
        self.assertFalse(gate["required_route_sharing_assumption"])
        self.assertIn(
            "byte-accounting projection only",
            self.prereg["scope"]["claim_boundary"],
        )

    def test_tail_and_retained_populations_are_explicit(self) -> None:
        inventory = self.prereg["expected_inventory"]
        self.assertEqual(inventory["compressed_tail_tensors"], 27)
        self.assertGreater(inventory["retained_parameters"], 0)
        self.assertEqual(
            inventory["compressed_parameters"] + inventory["retained_parameters"],
            inventory["total_parameters"],
        )

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
