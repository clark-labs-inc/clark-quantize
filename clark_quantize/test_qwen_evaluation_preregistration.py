from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "experiments/qwen38_27b_spot/evaluation_preregistration.json"
TARGET_158 = ROOT / "experiments/qwen38_27b_spot/target_158_preregistration.json"


def load_script(name: str):
    path = ROOT / f"experiments/qwen38_27b_spot/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


QUALITY = load_script("behavioral_quality_gate")
RADIX = load_script("radix5_performance_gate")


class EvaluationPreregistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.value = json.loads(PREREGISTRATION.read_text())

    def test_quality_defaults_are_sealed(self) -> None:
        args = argparse.Namespace(max_new_tokens=24, max_ppl_ratio=1.5, max_mean_teacher_kl=1.0)
        QUALITY.validate_preregistration(self.value, args)
        args.max_ppl_ratio = 1.6
        with self.assertRaisesRegex(ValueError, "differ"):
            QUALITY.validate_preregistration(self.value, args)

    def test_radix_defaults_are_sealed(self) -> None:
        args = argparse.Namespace(trits=10_000_000, sample_rows=512)
        thresholds = RADIX.validate_preregistration(self.value, args)
        self.assertEqual(thresholds["minimum_encode_trits_per_second"], 50_000_000)
        args.trits += 1
        with self.assertRaisesRegex(ValueError, "differ"):
            RADIX.validate_preregistration(self.value, args)
        args.trits -= 1
        changed = json.loads(json.dumps(self.value))
        changed["radix5_performance"]["minimum_decode_trits_per_second"] = 1
        with self.assertRaisesRegex(ValueError, "differ"):
            RADIX.validate_preregistration(changed, args)

    def test_target_158_ledger_is_complete_and_self_consistent(self) -> None:
        target = json.loads(TARGET_158.read_text())
        inventory = target["fixed_inventory_expectations"]
        total = inventory["total_parameters"]
        rank2 = inventory["rank2_ternary_parameters"]
        retained = inventory["rank_not_2_bf16_parameters"]
        scales = inventory["per_row_fp16_scale_count"]
        self.assertEqual(rank2 + retained, total)
        computed = (rank2 * math.log2(3) + retained * 16 + scales * 16) / total
        self.assertAlmostEqual(
            computed,
            inventory["projected_complete_effective_bits_per_parameter"],
            places=14,
        )
        self.assertLessEqual(computed, target["promotion"]["maximum_complete_effective_bits_per_parameter"])
        self.assertTrue(target["promotion"]["does_not_authorize_full_pack"])
        self.assertIn("vision", target["parameter_scope"]["ternary"])
        self.assertIn("MTP", target["parameter_scope"]["ternary"])


if __name__ == "__main__":
    unittest.main()
