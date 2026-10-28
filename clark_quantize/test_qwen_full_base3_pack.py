from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from clark_quantize.base3 import unpack_base3
from clark_quantize.qwen_base3_artifact import QwenBase3ArtifactReader
from clark_quantize.ternary import ternary_codes_scales


REPO = Path(__file__).resolve().parents[1]
PACKER = REPO / "experiments/qwen38_27b_spot/full_base3_pack.py"


class QwenFullBase3PackIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        generator = torch.Generator().manual_seed(20260815)
        self.weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(6, 4097, generator=generator).to(torch.bfloat16),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 4097, generator=generator).to(torch.bfloat16),
            "model.language_model.embed_tokens.weight": torch.randn(4, 4097, generator=generator).to(torch.bfloat16),
            "model.language_model.layers.0.input_layernorm.weight": torch.randn(259, generator=generator).to(torch.bfloat16),
            "model.visual.proj.weight": torch.randn(2, 8, generator=generator).to(torch.bfloat16),
        }
        shard = "model-00001-of-00001.safetensors"
        save_file(self.weights, self.model / shard)
        index = {
            "metadata": {"total_size": (self.model / shard).stat().st_size},
            "weight_map": {name: shard for name in self.weights},
        }
        index_path = self.model / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
        mapped_names = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
        ]
        mapped = sum(self.weights[name].numel() for name in mapped_names)
        total = sum(weight.numel() for weight in self.weights.values())
        scales = sum(
            weight.shape[0] * math.ceil(weight.shape[1] / 128)
            for name, weight in self.weights.items()
            if name in mapped_names
        )
        self.preflight = self.root / "preflight.json"
        self.preflight.write_text(json.dumps({
            "download_receipt": {
                "revision": "synthetic-qwen38-integration-v1",
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            },
            "parameter_ledger": {
                "total": total,
                "ternary_payload": mapped,
                "retained_high_precision": total - mapped,
                "fp16_scale_count": scales,
                "base3_payload_bytes": math.ceil((mapped * math.log2(3) + (total - mapped + scales) * 16) / 8),
            },
        }, sort_keys=True) + "\n")
        self.out = self.root / "artifact.cq3q"
        self.receipt = self.root / "receipt.json"
        self.progress = self.root / "progress.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self) -> list[str]:
        return [
            sys.executable,
            str(PACKER),
            "--model-dir", str(self.model),
            "--preflight", str(self.preflight),
            "--out", str(self.out),
            "--receipt", str(self.receipt),
            "--progress", str(self.progress),
            "--row-batch", "2",
            "--workers", "2",
            "--max-in-flight", "2",
            "--device", "cpu",
        ]

    def test_complete_export_exact_resume_and_corruption_rejection(self) -> None:
        first = subprocess.run(self.command(), check=False, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_receipt = json.loads(self.receipt.read_text())
        self.assertTrue(first_receipt["promotion"]["passed"], first.stdout)
        self.assertEqual(first_receipt["resume"]["initial_batches"], 0)
        self.assertGreater(first_receipt["resume"]["written_batches"], 0)
        self.assertFalse(self.progress.exists())
        complete_ledger = first_receipt["complete_model_ledger"]
        actual_artifact_bytes = sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file())
        self.assertEqual(complete_ledger["artifact_storage_bytes_excluding_retained_tensors"], actual_artifact_bytes)
        self.assertGreater(complete_ledger["artifact_metadata_bytes"], 0)
        self.assertEqual(
            complete_ledger["complete_transport_bytes_including_referenced_retained_tensors"],
            actual_artifact_bytes + complete_ledger["retained_high_precision_bytes"],
        )

        manifest = json.loads((self.out / "codes.cq3d/manifest.json").read_text())
        reader = QwenBase3ArtifactReader(self.out)
        validated = reader.validate()
        self.assertEqual(validated["trits"], first_receipt["code_ledger"]["trits"])
        self.assertEqual(validated["scales"] * 2, first_receipt["complete_model_ledger"]["fp16_scale_bytes"])
        decoded_trits = 0
        for record in manifest["chunks"]:
            descriptor = record["descriptor"]
            with safe_open(
                self.model / "model-00001-of-00001.safetensors", framework="pt", device="cpu"
            ) as handle:
                weight = handle.get_slice(descriptor["tensor"])[
                    descriptor["row_start"] : descriptor["row_stop"], :
                ].contiguous().float()
            expected_codes, expected_scales = ternary_codes_scales(weight, group=128, method="optimal")
            frame = (self.out / "codes.cq3d/chunks" / f"{record['index']:08d}.cq3a").read_bytes()
            np.testing.assert_array_equal(unpack_base3(frame), expected_codes.numpy().reshape(-1))
            scale_bytes = (self.out / "scales" / descriptor["scale_file"]).read_bytes()
            self.assertEqual(scale_bytes, expected_scales.numpy().astype("<f2", copy=False).tobytes())
            self.assertEqual(hashlib.sha256(scale_bytes).hexdigest(), descriptor["scale_sha256"])
            decoded_trits += record["trits"]
        self.assertEqual(decoded_trits, first_receipt["code_ledger"]["trits"])

        for name in reader.tensor_names:
            source = self.weights[name].float()
            expected_codes, expected_scales = ternary_codes_scales(source, group=128, method="optimal")
            padding = (-source.shape[1]) % 128
            padded = torch.nn.functional.pad(expected_codes, (0, padding)) if padding else expected_codes
            expected = (
                padded.reshape(source.shape[0], -1, 128).to(torch.float16)
                * expected_scales.unsqueeze(-1)
            ).reshape(source.shape[0], -1)[:, : source.shape[1]]
            torch.testing.assert_close(reader.tensor(name), expected, rtol=0, atol=0)

        subprocess.run(self.command(), check=True, text=True, capture_output=True)
        resumed = json.loads(self.receipt.read_text())
        self.assertEqual(resumed["resume"]["initial_batches"], len(manifest["chunks"]))
        self.assertEqual(resumed["resume"]["reused_batches"], len(manifest["chunks"]))
        self.assertEqual(resumed["resume"]["written_batches"], 0)

        scale_path = self.out / "scales" / manifest["chunks"][0]["descriptor"]["scale_file"]
        damaged = bytearray(scale_path.read_bytes())
        damaged[0] ^= 1
        scale_path.write_bytes(damaged)
        rejected = subprocess.run(self.command(), check=False, text=True, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("missing or changed", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
