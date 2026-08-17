from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "experiments/qwen38_27b_spot/evaluation_preregistration.json"
TARGET_158 = ROOT / "experiments/qwen38_27b_spot/target_158_preregistration.json"
GEMV_PREREGISTRATION = (
    ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime/gemv_preregistration_v5.json"
)
RECURRENT_DELTA_PREREGISTRATION = (
    ROOT
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "qwen38_27b_recurrent_delta_avx2_preregistration_v1.json"
)


def load_script(name: str):
    path = ROOT / f"experiments/qwen38_27b_spot/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


QUALITY = load_script("behavioral_quality_gate")
RADIX = load_script("radix5_performance_gate")


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEMV = load_path(
    "benchmark_ternary_gemv",
    ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime/benchmark_ternary_gemv.py",
)


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

    def test_radix_sample_regeneration_is_explicitly_cpu(self) -> None:
        expected = (np.asarray([-1, 0, 1], dtype=np.int8), [{"name": "probe"}])
        with mock.patch.object(RADIX, "load_codes", return_value=expected) as load_codes:
            actual = RADIX.load_real_codes(Path("/model"), {"probes": []}, 512, 10_000_000)
        self.assertIs(actual, expected)
        self.assertEqual(load_codes.call_args.args[:4], (Path("/model"), {"probes": []}, 512, 10_000_000))
        self.assertEqual(load_codes.call_args.args[4].type, "cpu")

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

    def test_cpu_gemv_preregistration_seals_shape_and_promotion_scope(self) -> None:
        value = json.loads(GEMV_PREREGISTRATION.read_text())
        args = argparse.Namespace(
            tensor="model.language_model.layers.0.mlp.gate_proj.weight",
            rows=17408,
            columns=5120,
            warmups=2,
            repetitions=7,
            threads=[1, 2, 4],
            cache_eviction_bytes=268435456,
        )
        self.assertIs(GEMV.validate_preregistration(value, args), value)
        self.assertEqual(
            value["promotion"]["maximum_compute_image_bits_per_weight_including_fp16_scales"],
            (128 * 2 + 16) / 128,
        )
        self.assertTrue(value["promotion"]["does_not_authorize_breakthrough_claim"])
        args.rows -= 1
        with self.assertRaisesRegex(ValueError, "differ"):
            GEMV.validate_preregistration(value, args)

    def test_qwen38_recurrent_delta_gate_covers_full_state_stream(self) -> None:
        value = json.loads(RECURRENT_DELTA_PREREGISTRATION.read_text())
        scope = value["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertEqual(scope["value_heads"], 48)
        self.assertEqual(scope["head_dimension"], 128)
        self.assertEqual(scope["deltanet_layers"], 48)
        self.assertEqual(
            scope["state_bytes_all_layers"],
            scope["deltanet_layers"] * scope["state_bytes_per_layer"],
        )
        self.assertTrue(value["gate"]["require_exact_scalar_avx2_output"])
        self.assertTrue(value["gate"]["require_exact_scalar_avx2_final_state"])
        self.assertGreaterEqual(
            value["gate"][
                "minimum_full_stack_recurrence_only_tokens_per_second"
            ],
            2.0,
        )
        self.assertIn("CUTE", " ".join(value["promotion_rules"]))
        self.assertIn("MMLU", " ".join(value["promotion_rules"]))


if __name__ == "__main__":
    unittest.main()
