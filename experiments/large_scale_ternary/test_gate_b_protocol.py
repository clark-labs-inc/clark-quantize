from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b_protocol import (
    checkpoint_assignment_plan,
    compare_gate_b_reports,
    make_stratified_support_manifest,
    make_task_manifest,
    paired_accuracy_delta,
    seal_manifest,
    select_and_verify_rows,
    select_and_verify_support_rows,
    sha256_file,
    stratified_indices,
    validate_arm,
    verify_manifest,
    wilson_interval,
)
from gate_b_likelihood import aggregate_likelihood, causal_likelihood
from gate_b_model_arms import apply_ptq_ternary_dense


class GateBProtocolTests(unittest.TestCase):
    def test_stratified_sampling_is_balanced_and_deterministic(self) -> None:
        rows = (
            [{"subject": "large", "value": index} for index in range(20)]
            + [{"subject": "small", "value": index} for index in range(3)]
            + [{"subject": "medium", "value": index} for index in range(8)]
        )
        first = stratified_indices(rows, 9, 42, "subject")
        second = stratified_indices(rows, 9, 42, "subject")
        self.assertEqual(first, second)
        counts = {}
        for index in first:
            subject = rows[index]["subject"]
            counts[subject] = counts.get(subject, 0) + 1
        self.assertEqual(counts, {"large": 3, "medium": 3, "small": 3})

    def test_manifest_detects_dataset_drift(self) -> None:
        rows = [{"question": f"q{index}", "answer": index} for index in range(6)]
        task = make_task_manifest(
            rows,
            task="toy",
            dataset="org/toy",
            config="main",
            split="test",
            revision="abc123",
            fingerprint="fingerprint",
            limit=4,
            seed=7,
            shots=0,
        )
        manifest = seal_manifest([task], seed=7)
        verify_manifest(manifest)
        self.assertEqual(len(select_and_verify_rows(rows, task)), 4)
        changed = list(rows)
        selected_index = task["samples"][0]["index"]
        changed[selected_index] = {"question": "changed", "answer": -1}
        with self.assertRaisesRegex(ValueError, "dataset drift"):
            select_and_verify_rows(changed, task)

    def test_manifest_tampering_is_rejected(self) -> None:
        task = make_task_manifest(
            [{"x": 1}],
            task="toy",
            dataset="org/toy",
            config="main",
            split="test",
            revision="abc",
            fingerprint="fp",
            limit=1,
            seed=1,
            shots=0,
        )
        manifest = seal_manifest([task], 1)
        manifest["seed"] = 2
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_manifest(manifest)

    def test_support_examples_are_pinned_per_stratum(self) -> None:
        rows = [
            {"subject": subject, "question": f"{subject}-{index}"}
            for subject in ("a", "b")
            for index in range(4)
        ]
        support = make_stratified_support_manifest(
            rows,
            dataset="org/toy",
            config="all",
            split="dev",
            revision="abc",
            fingerprint="fp",
            shots_per_stratum=2,
            stratum_key="subject",
        )
        selected = select_and_verify_support_rows(rows, support)
        self.assertEqual(
            [row["question"] for row in selected["a"]], ["a-0", "a-1"]
        )
        changed = list(rows)
        changed[0] = {"subject": "a", "question": "changed"}
        with self.assertRaisesRegex(ValueError, "support dataset drift"):
            select_and_verify_support_rows(changed, support)

    def test_checkpoint_mapping_requires_complete_exact_state(self) -> None:
        model = {
            "block.weight": torch.empty(2, 3),
            "block.bias": torch.empty(2),
        }
        checkpoint = {
            "_orig_mod.block.weight_master": torch.empty(2, 3),
            "_orig_mod.block.bias": torch.empty(2),
        }
        self.assertEqual(
            checkpoint_assignment_plan(checkpoint, model),
            {
                "block.weight": "block.weight_master",
                "block.bias": "block.bias",
            },
        )
        with self.assertRaisesRegex(ValueError, "absent from checkpoint"):
            checkpoint_assignment_plan(
                {"block.weight_master": torch.empty(2, 3)}, model
            )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            checkpoint_assignment_plan(
                {
                    "block.weight_master": torch.empty(3, 3),
                    "block.bias": torch.empty(2),
                },
                model,
            )

    def test_arm_contract_is_unambiguous(self) -> None:
        validate_arm("bf16", None)
        validate_arm("qat_ternary", Path("checkpoint.pt"))
        validate_arm("ptq_ternary", None)
        with self.assertRaises(ValueError):
            validate_arm("bf16", Path("checkpoint.pt"))
        with self.assertRaises(ValueError):
            validate_arm("qat_ternary", None)

    def test_ptq_control_materializes_without_master_weights(self) -> None:
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(128, 2, bias=False)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 128)
                self.self_attn = Attention()

        model = Toy()
        metadata = apply_ptq_ternary_dense(model, dtype=torch.float32)
        self.assertEqual(metadata["ternary_modules"], ["self_attn.q_proj"])
        self.assertIsInstance(model.self_attn.q_proj, torch.nn.Linear)
        self.assertFalse(
            any(name.endswith("weight_master") for name, _ in model.named_parameters())
        )
        unique = set(model.self_attn.q_proj.weight.unique().tolist())
        self.assertLessEqual(len(unique), 3 * 2)

    def test_statistics_are_reproducible(self) -> None:
        interval = wilson_interval(50, 100)
        self.assertAlmostEqual(interval[0], 0.4038315, places=6)
        self.assertAlmostEqual(interval[1], 0.5961685, places=6)
        left = [
            {"sample_id": str(index), "correct": index < 7} for index in range(10)
        ]
        right = [
            {"sample_id": str(index), "correct": index < 5} for index in range(10)
        ]
        result = paired_accuracy_delta(
            left, right, seed=9, bootstrap_samples=1_000
        )
        self.assertAlmostEqual(result["accuracy_delta"], 0.2)
        self.assertEqual(
            result,
            paired_accuracy_delta(left, right, seed=9, bootstrap_samples=1_000),
        )

    def test_file_hash_streams_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"gate-b")
            self.assertEqual(
                sha256_file(path),
                "d32620c2747a5e973d55a7061475f64c2d2295f60947626407ab0f461a740e3f",
            )

    def test_report_comparison_requires_same_frozen_gate(self) -> None:
        def report(arm, accuracy, correct):
            return {
                "arm": arm,
                "model": {"id": "org/model", "revision": "abc"},
                "manifest": {"content_sha256": "sealed"},
                "results": [
                    {
                        "task": "toy",
                        "accuracy": accuracy,
                        "latency_ms": {"mean": 2.0},
                        "samples": [
                            {"sample_id": str(index), "correct": value}
                            for index, value in enumerate(correct)
                        ],
                    }
                ],
            }

        left = report("qat_ternary", 0.75, [True, True, True, False])
        right = report("bf16", 0.5, [True, True, False, False])
        comparison = compare_gate_b_reports(
            left, right, seed=3, bootstrap_samples=100
        )
        self.assertEqual(comparison["tasks"][0]["accuracy_delta"], 0.25)
        right["manifest"]["content_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "same sealed"):
            compare_gate_b_reports(left, right)

    def test_causal_likelihood_and_multilingual_aggregation(self) -> None:
        logits = torch.tensor(
            [
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ]
        )
        ids = torch.tensor([[0, 0, 1]])
        metrics = causal_likelihood(logits, ids, teacher_logits=logits)
        self.assertEqual(metrics["token_count"], 2)
        self.assertAlmostEqual(metrics["mean_teacher_kl"], 0.0, places=7)
        records = [
            {"stratum": "a", **metrics},
            {"stratum": "b", **metrics},
        ]
        aggregate = aggregate_likelihood(records)
        self.assertEqual(aggregate["documents"], 2)
        self.assertEqual(len(aggregate["language_strata"]), 2)
        self.assertAlmostEqual(aggregate["micro_mean_teacher_kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
