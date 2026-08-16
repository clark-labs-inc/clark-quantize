from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from clark_quantize.base3 import frame_metadata, pack_base3
from clark_quantize.qwen_base3_artifact import _sha256


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/behavioral_quality_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen38_behavioral_quality_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BehavioralQualityMathTests(unittest.TestCase):
    def test_causal_nll_matches_cross_entropy(self) -> None:
        logits = torch.tensor([
            [[3.0, 1.0, -2.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0]],
        ])
        ids = torch.tensor([[0, 1, 2]])
        total, count = MODULE.causal_nll(logits, ids)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), ids[:, 1:].reshape(-1), reduction="sum")
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, expected.item(), places=6)

    def test_teacher_student_kl_is_zero_for_equal_logits(self) -> None:
        logits = torch.randn(2, 4, 7, generator=torch.Generator().manual_seed(7))
        total, positions = MODULE.teacher_student_kl(logits, logits.clone())
        self.assertEqual(positions, 6)
        self.assertAlmostEqual(total, 0.0, places=6)

    def test_teacher_student_kl_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            MODULE.teacher_student_kl(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))

    def test_parameter_map_accepts_exact_and_qwen_wrapper_names(self) -> None:
        parameters = {
            "model.layers.0.mlp.up_proj.weight": object(),
            "lm_head.weight": object(),
        }
        result = MODULE.resolve_parameter_map(
            ("model.language_model.layers.0.mlp.up_proj.weight", "lm_head.weight"),
            parameters,
        )
        self.assertEqual(result, {
            "model.language_model.layers.0.mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
            "lm_head.weight": "lm_head.weight",
        })

    def test_parameter_map_fails_closed_on_missing_or_collision(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.resolve_parameter_map(("model.language_model.layers.0.mlp.up_proj.weight",), {})
        with self.assertRaisesRegex(ValueError, "map to"):
            MODULE.resolve_parameter_map(
                ("model.language_model.x", "model.x"),
                {"model.x": object()},
            )

    def test_artifact_identity_binds_all_pack_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            manifest_path = artifact / "codes.cq3d/manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest = {
                "identity": {"model_revision": "a" * 40, "index_sha256": "b" * 64},
                "source_sha256": "c" * 64,
                "extra": {"scale_source_sha256": "d" * 64},
                "ledger": {"trits": 9},
            }
            manifest_path.write_text(json.dumps(manifest))
            reader = type("Reader", (), {"manifest": manifest, "manifest_path": manifest_path})()
            receipt = {
                "artifact": str(artifact),
                "code_manifest_sha256": MODULE.file_sha256(manifest_path),
                "code_source_sha256": "c" * 64,
                "scale_source_sha256": "d" * 64,
                "identity": {"model_revision": "a" * 40, "index_sha256": "b" * 64},
                "code_ledger": {"trits": 9},
            }
            self.assertTrue(MODULE.validate_artifact_identity(reader, artifact, receipt)["valid"])
            receipt["scale_source_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "differs"):
                MODULE.validate_artifact_identity(reader, artifact, receipt)

    def test_generation_hash_is_canonical_and_sensitive(self) -> None:
        rows = [{"prompt": "p", "new_tokens": 1, "completion": "x"}]
        self.assertEqual(MODULE.generation_sha256(rows), MODULE.generation_sha256(list(rows)))
        changed = [{**rows[0], "completion": "y"}]
        self.assertNotEqual(MODULE.generation_sha256(rows), MODULE.generation_sha256(changed))

    def test_terminal_quality_is_valid_for_rescue_even_when_ptq_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "teacher.pt"
            cache.write_bytes(b"teacher")
            revision = "a" * 40
            index_hash = "b" * 64
            manifest_hash = "c" * 64
            code_hash = "d" * 64
            scale_hash = "e" * 64
            artifact = str(Path(temporary) / "artifact")
            pack = {
                "artifact": artifact,
                "code_manifest_sha256": manifest_hash,
                "code_source_sha256": code_hash,
                "scale_source_sha256": scale_hash,
                "identity": {"model_revision": revision, "index_sha256": index_hash},
                "code_ledger": {"trits": 9},
            }
            receipt = {
                "schema": "clark_qwen38_base3_behavioral_quality_v1",
                "model_revision": revision,
                "artifact_integrity": {
                    "valid": True,
                    "artifact": artifact,
                    "manifest_sha256": manifest_hash,
                    "code_source_sha256": code_hash,
                    "scale_source_sha256": scale_hash,
                    "model_revision": revision,
                    "index_sha256": index_hash,
                    "trits": 9,
                },
                "applied": {"parameters": 9, "tensors": 1, "batches": 1},
                "teacher_cache": {"path": str(cache), "sha256": MODULE.file_sha256(cache)},
                "bf16": {"likelihood": {"mean_nll": 1.0, "perplexity": 2.0}},
                "ternary": {"likelihood": {"mean_nll": 7.0, "perplexity": 1000.0}},
                "comparison": {
                    "perplexity_ratio": 500.0,
                    "mean_teacher_kl": 5.0,
                    "all_bf16_and_ternary_completions_nonempty": True,
                    "bf16_generations_deterministic": True,
                    "ternary_generations_deterministic": True,
                    "bf16_generation_sha256": "f" * 64,
                    "bf16_generation_repeat_sha256": "f" * 64,
                    "ternary_generation_sha256": "0" * 64,
                    "ternary_generation_repeat_sha256": "0" * 64,
                },
                "promotion": {"passed": False},
            }
            validated = MODULE.validate_terminal_quality_receipt(
                receipt, pack, teacher_cache=cache
            )
            self.assertTrue(validated["valid"])
            self.assertFalse(validated["promotion_passed"])

            receipt["comparison"]["mean_teacher_kl"] = float("nan")
            with self.assertRaisesRegex(ValueError, "not valid terminal science"):
                MODULE.validate_terminal_quality_receipt(receipt, pack, teacher_cache=cache)

    def test_parallel_apply_matches_exact_group_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = root / "codes.cq3d" / "chunks"
            scales_dir = root / "scales"
            chunks.mkdir(parents=True)
            scales_dir.mkdir()
            codes = np.array([[1, 0, -1, 1], [-1, 1, 0, 0]], dtype=np.int8)
            scales = np.array([[0.5, 2.0], [1.5, 0.25]], dtype="<f2")
            frame = pack_base3(codes.reshape(-1))
            (chunks / "00000000.cq3a").write_bytes(frame)
            scale_raw = scales.tobytes()
            (scales_dir / "00000000.f16").write_bytes(scale_raw)
            record = {
                "index": 0,
                "trits": codes.size,
                "raw_sha256": _sha256(codes.reshape(-1).tobytes()),
                "frame_sha256": _sha256(frame),
                "frame_metadata": frame_metadata(frame),
                "descriptor": {
                    "tensor": "weight",
                    "row_start": 0,
                    "row_stop": 2,
                    "tensor_shape": [2, 4],
                    "scale_file": "00000000.f16",
                    "scale_bytes": len(scale_raw),
                    "scale_sha256": _sha256(scale_raw),
                    "scale_shape": [2, 2],
                },
            }
            reader = type("Reader", (), {
                "root": root,
                "group_size": 2,
                "tensor_names": ("weight",),
                "sources": {"weight": {"shape": [2, 4]}},
                "records_by_tensor": {"weight": [record]},
            })()
            model = torch.nn.Linear(4, 2, bias=False, dtype=torch.float16)
            applied = MODULE.apply_artifact(model, reader, workers=1, max_in_flight=1)
            expected = torch.tensor([
                [0.5, 0.0, -2.0, 2.0],
                [-1.5, 1.5, 0.0, 0.0],
            ], dtype=torch.float16)
            self.assertTrue(torch.equal(model.weight, expected))
            self.assertEqual(applied, {"tensors": 1, "batches": 1, "parameters": 8})


if __name__ == "__main__":
    unittest.main()
