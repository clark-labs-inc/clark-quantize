from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38LearnedHashPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_learned_hash_routing_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_only_qwen38_27b(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])

    def test_train_and_evaluation_spans_are_disjoint(self) -> None:
        for tensor in self.prereg["scope"]["tensors"]:
            train_start, train_stop = tensor["train_row_span"]
            eval_start, eval_stop = tensor["evaluation_row_span"]
            self.assertLessEqual(train_stop, eval_start)
            self.assertLess(train_start, train_stop)
            self.assertLess(eval_start, eval_stop)

    def test_gate_up_share_one_route_and_down_is_separate(self) -> None:
        groups = self.prereg["scope"]["shared_routing_groups"]
        self.assertEqual([group["name"] for group in groups], ["gate_up", "down"])
        self.assertEqual(len(groups[0]["members"]), 2)
        self.assertEqual(len(groups[1]["members"]), 1)

    def test_gate_is_bounded_and_not_a_model_claim(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreater(gate["minimum_aggregate_error_reduction_vs_random"], 0)
        self.assertLessEqual(gate["maximum_component_bits_per_weight"], 1.56)
        self.assertIn(
            "cannot establish semi-lossless",
            self.prereg["promotion_rules"][-1],
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
