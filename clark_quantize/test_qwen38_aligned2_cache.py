from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.base3_block203 import matrix_payload_bits, pack_rows
from clark_quantize.qwen38_aligned2_cache import (
    CACHE_MANIFEST_SCHEMA,
    CACHE_STATE_SCHEMA,
    Qwen38Aligned2CacheBuilder,
    Qwen38Aligned2CacheError,
)
from clark_quantize.qwen38_cq3_container import Qwen38CQ3Writer
from clark_quantize.scale_codebook import pack_nibbles
from clark_quantize.ternary_cpu import (
    native_block203_transcode_available,
    pack_group128_2bit,
)


@unittest.skipUnless(
    native_block203_transcode_available(),
    "native streaming block203 transcode is absent",
)
class Qwen38Aligned2CacheTests(unittest.TestCase):
    rows = 4
    columns = 256

    def fixture(self, root: Path) -> tuple[Path, list[np.ndarray]]:
        path = root / "qwen38-cache-source.cq3"
        codebook = np.linspace(0.0, 1.0, 16, dtype=np.float16)
        tensors = []
        codes_by_tensor = []
        parameter_count = 2 * self.rows * self.columns + 4
        with Qwen38CQ3Writer(
            path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "cache-test"},
            strategy={
                "name": "base3_group128_row_normalized_codebook4",
                "codec": "block203",
                "group_size": 128,
                "scale_bits": 4,
                "row_anchor_dtype": "F16",
                "scale_codebook_entries": 16,
                "scale_codebook_dtype": "F16",
            },
            parameter_count=parameter_count,
        ) as writer:
            for index in range(2):
                codes = (
                    (np.arange(self.rows * self.columns) + index) % 3 - 1
                ).astype(np.int8).reshape(self.rows, self.columns)
                codes_by_tensor.append(codes)
                groups = self.columns // 128
                scale_indices = (
                    np.arange(self.rows * groups, dtype=np.uint8) + index
                ) % 16
                anchors = np.linspace(
                    0.125, 1.0, self.rows, dtype=np.float16
                )
                packed_indices = pack_nibbles(scale_indices)
                prefix = f"tensor.{index:04d}"
                code = writer.add_section(
                    prefix + ".codes",
                    "ternary_block203",
                    pack_rows(codes),
                    metadata={
                        "rows": self.rows,
                        "columns": self.columns,
                        "trits": codes.size,
                        "payload_bits": matrix_payload_bits(
                            self.rows, self.columns
                        ),
                        "group_trits": 128,
                        "full_group_bits": 203,
                    },
                )
                scales = writer.add_section_parts(
                    prefix + ".scales",
                    "scale_indices_u4_anchors_f16_and_codebook_f16",
                    (
                        packed_indices,
                        anchors.astype("<f2").tobytes(),
                        codebook.astype("<f2").tobytes(),
                    ),
                    metadata={
                        "scale_count": scale_indices.size,
                        "index_bytes": len(packed_indices),
                        "codebook_bytes": codebook.nbytes,
                        "codebook_entries": 16,
                        "index_order": "low_nibble_first",
                        "row_count": self.rows,
                        "anchor_bytes": anchors.nbytes,
                        "factor_order": (
                            "indices_then_row_anchors_then_codebook"
                        ),
                        "decode": (
                            "float32(fp16_anchor) * float32(fp16_ratio)"
                        ),
                    },
                )
                tensors.append(
                    {
                        "name": (
                            f"model.language_model.layers.{index}."
                            "mlp.gate_proj.weight"
                        ),
                        "shape": [self.rows, self.columns],
                        "parameters": codes.size,
                        "source_dtype": "BF16",
                        "encoding": (
                            "base3_group128_row_normalized_codebook4"
                        ),
                        "sections": [code["id"], scales["id"]],
                    }
                )
            exact = writer.add_section(
                "tensor.0002.retained",
                "retained_exact_BF16",
                bytes(range(8)),
                metadata={"dtype": "BF16"},
            )
            tensors.append(
                {
                    "name": "model.language_model.layers.0.input_layernorm.weight",
                    "shape": [4],
                    "parameters": 4,
                    "source_dtype": "BF16",
                    "encoding": "retained_exact_BF16",
                    "sections": [exact["id"]],
                }
            )
            writer.finalize(tensors=tensors)
        return path, codes_by_tensor

    def test_build_resumes_and_publishes_exact_authenticated_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, codes = self.fixture(root)
            cache = root / "qwen38.a2"
            builder = Qwen38Aligned2CacheBuilder(source, cache)

            partial = builder.build(max_new_tensors=1)
            self.assertEqual(partial["schema"], CACHE_STATE_SCHEMA)
            self.assertFalse(partial["complete"])
            self.assertEqual(len(partial["completed"]), 1)
            self.assertFalse(builder.manifest_path.exists())

            manifest = builder.build(workers=2)
            self.assertEqual(manifest["schema"], CACHE_MANIFEST_SCHEMA)
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(manifest["tensors"]), 2)
            self.assertFalse(
                manifest["cache"]["full_int8_weight_matrix_materialized"]
            )
            ledger = manifest["identity"]["ledger"]
            self.assertEqual(ledger["ternary_parameters"], 2048)
            self.assertEqual(ledger["retained_exact_parameters"], 4)
            self.assertEqual(ledger["cache_payload_bytes"], 512)
            self.assertEqual(ledger["cache_alignment_padding_bytes"], 0)
            self.assertEqual(
                ledger["active_runtime_weight_bytes"], 512 + 88 + 8
            )
            content = cache.read_bytes()
            for record, expected_codes in zip(manifest["tensors"], codes):
                expected = pack_group128_2bit(expected_codes).tobytes()
                start = record["offset"]
                self.assertEqual(
                    content[start : start + record["bytes"]], expected
                )
            self.assertEqual(builder.build(), manifest)

    def test_committed_tensor_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            cache = root / "qwen38.a2"
            builder = Qwen38Aligned2CacheBuilder(source, cache)
            state = builder.build(max_new_tensors=1)
            with cache.open("r+b") as handle:
                handle.seek(state["completed"][0]["offset"])
                value = handle.read(1)
                handle.seek(state["completed"][0]["offset"])
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(
                Qwen38Aligned2CacheError, "checkpoint tensor"
            ):
                builder.build()

    def test_noncanonical_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            cache = root / "qwen38.a2"
            builder = Qwen38Aligned2CacheBuilder(source, cache)
            builder.build(max_new_tensors=1)
            state = json.loads(builder.state_path.read_text())
            builder.state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(
                Qwen38Aligned2CacheError, "non-canonical"
            ):
                builder.build()
