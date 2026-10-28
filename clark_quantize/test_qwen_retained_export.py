from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib
import tempfile
import unittest

from safetensors import safe_open
from safetensors.torch import save_file
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/export_retained_tensors.py"
SPEC = importlib.util.spec_from_file_location("qwen38_retained_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RetainedExportTests(unittest.TestCase):
    def test_verifies_pinned_source_shard_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / ".cache/huggingface/download"
            cache.mkdir(parents=True)
            shard = root / "one.safetensors"
            shard.write_bytes(b"pinned shard bytes")
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            (cache / "one.safetensors.metadata").write_text(f"revision\n{digest}\n123.0\n")
            receipt = MODULE.verify_source_shards(root, {shard.name}, "revision")
            self.assertEqual(receipt["shards"], 1)
            self.assertEqual(receipt["bytes"], shard.stat().st_size)
            shard.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                MODULE.verify_source_shards(root, {shard.name}, "revision")

    def test_exports_only_unmapped_tensors_with_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            mapped = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            retained_a = torch.arange(10, dtype=torch.bfloat16)
            retained_b = torch.randn(2, 3, dtype=torch.bfloat16)
            save_file({"mapped": mapped, "retained.a": retained_a}, model / "one.safetensors")
            save_file({"retained.b": retained_b}, model / "two.safetensors")
            (model / "model.safetensors.index.json").write_text(
                '{"weight_map":{"mapped":"one.safetensors",'
                '"retained.a":"one.safetensors","retained.b":"two.safetensors"}}'
            )
            output = root / "retained.safetensors"
            manifest, file_hash = MODULE.export_retained(
                model, {"mapped"}, output, model_revision="test-revision"
            )
            self.assertEqual(manifest["tensor_count"], 2)
            self.assertEqual(manifest["parameters"], 16)
            self.assertEqual(manifest["tensor_bytes"], 32)
            self.assertEqual(file_hash, MODULE.file_sha256(output))
            with safe_open(output, framework="pt", device="cpu") as handle:
                self.assertEqual(set(handle.keys()), {"retained.a", "retained.b"})
                torch.testing.assert_close(handle.get_tensor("retained.a"), retained_a)
                torch.testing.assert_close(handle.get_tensor("retained.b"), retained_b)

    def test_rejects_unknown_mapped_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, root / "one.safetensors")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"weight":"one.safetensors"}}'
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                MODULE.export_retained(root, {"missing"}, root / "out.safetensors", model_revision="test")


if __name__ == "__main__":
    unittest.main()
