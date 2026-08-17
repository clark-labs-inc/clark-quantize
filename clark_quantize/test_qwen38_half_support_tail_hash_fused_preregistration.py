from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38HalfSupportTailHashFusedPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_half_support_tail_hash_fused_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_qwen38_27b_only(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])
        self.assertEqual(scope["affected_checkpoint_tensors"], 27)

    def test_real_vision_geometry_and_budget_are_frozen(self) -> None:
        case = self.prereg["case"]
        self.assertEqual((case["rows"], case["columns"]), (1152, 4304))
        self.assertEqual(case["tail_width"], 80)
        self.assertEqual(case["sketch_dim"], 34)
        self.assertEqual(case["raw_code_bits_per_weight"], 1.5)
        self.assertLessEqual(
            case["projected_complete_component_bits_per_weight"], 1.58
        )

    def test_gate_requires_low_overhead_and_exact_formula(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreaterEqual(
            gate["minimum_fused_throughput_fraction_of_tail_only"], 0.95
        )
        self.assertLessEqual(gate["maximum_absolute_error"], 0.0005)
        self.assertLessEqual(
            gate["maximum_complete_component_bits_per_weight"], 1.58
        )
        self.assertIn("T x plus A times R x", self.prereg["promotion_rules"][2])

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
