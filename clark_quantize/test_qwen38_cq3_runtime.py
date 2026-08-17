from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.base3_block203 import (
    matrix_payload_bits,
    pack_rows,
)
from clark_quantize.qwen38_cq3_container import (
    CQ3ContainerError,
    Qwen38CQ3Reader,
    Qwen38CQ3Writer,
)
from clark_quantize.qwen38_cq3_runtime import (
    CQ3RuntimeError,
    Qwen38CQ3RuntimeImage,
)
from clark_quantize.scale_codebook import pack_nibbles
from clark_quantize.ternary_cpu import native_available, quantize_q8


class Qwen38CQ3RuntimeTests(unittest.TestCase):
    rows = 8
    columns = 1024

    def fixture(self, root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
        path = root / "qwen38-row-normalized.cq3"
        codes = (
            np.arange(self.rows * self.columns, dtype=np.int64).reshape(
                self.rows, self.columns
            )
            % 3
            - 1
        ).astype(np.int8)
        groups = self.columns // 128
        indices = (
            np.arange(self.rows * groups, dtype=np.uint8) % 16
        ).astype(np.uint8)
        anchors = np.linspace(0.125, 1.0, self.rows, dtype=np.float16)
        codebook = np.linspace(0.0, 1.0, 16, dtype=np.float16)
        packed_indices = pack_nibbles(indices)
        exact_raw = bytes(range(8))
        with Qwen38CQ3Writer(
            path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "runtime-test"},
            strategy={
                "name": "base3_group128_row_normalized_codebook4",
                "codec": "block203",
                "group_size": 128,
                "scale_bits": 4,
                "row_anchor_dtype": "F16",
                "scale_codebook_entries": 16,
                "scale_codebook_dtype": "F16",
            },
            parameter_count=codes.size + 4,
        ) as writer:
            code = writer.add_section(
                "tensor.0000.codes",
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
                "tensor.0000.scales",
                "scale_indices_u4_anchors_f16_and_codebook_f16",
                (
                    packed_indices,
                    anchors.astype("<f2").tobytes(),
                    codebook.astype("<f2").tobytes(),
                ),
                metadata={
                    "scale_count": indices.size,
                    "index_bytes": len(packed_indices),
                    "codebook_bytes": codebook.nbytes,
                    "codebook_entries": 16,
                    "index_order": "low_nibble_first",
                    "row_count": self.rows,
                    "anchor_bytes": anchors.nbytes,
                    "factor_order": "indices_then_row_anchors_then_codebook",
                    "decode": "float32(fp16_anchor) * float32(fp16_ratio)",
                },
            )
            exact = writer.add_section(
                "tensor.0001.retained",
                "retained_exact_BF16",
                exact_raw,
                metadata={"dtype": "BF16"},
            )
            auxiliary = writer.add_section(
                "aux.0000",
                "auxiliary_file",
                b"{}\n",
                metadata={"name": "config.json"},
            )
            writer.finalize(
                tensors=[
                    {
                        "name": "model.layers.0.mlp.down_proj.weight",
                        "shape": [self.rows, self.columns],
                        "parameters": codes.size,
                        "source_dtype": "BF16",
                        "encoding": (
                            "base3_group128_row_normalized_codebook4"
                        ),
                        "sections": [code["id"], scales["id"]],
                    },
                    {
                        "name": "model.layers.0.input_layernorm.weight",
                        "shape": [4],
                        "parameters": 4,
                        "source_dtype": "BF16",
                        "encoding": "retained_exact_BF16",
                        "sections": [exact["id"]],
                    },
                ],
                auxiliary_files=[
                    {
                        "name": "config.json",
                        "section": auxiliary["id"],
                        "bytes": auxiliary["bytes"],
                        "sha256": auxiliary["sha256"],
                    }
                ],
            )
        return path, codes, indices, anchors

    def test_zero_copy_views_are_authenticated_and_survive_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, codes, indices, anchors = self.fixture(Path(temporary))
            image = Qwen38CQ3RuntimeImage(path)
            view = image.ternary_tensor(
                "model.layers.0.mlp.down_proj.weight"
            )
            self.assertEqual(view.shape, (self.rows, self.columns))
            self.assertEqual(view.packed_weights.nbytes, len(pack_rows(codes)))
            self.assertEqual(view.packed_scale_indices.nbytes, indices.size // 2)
            np.testing.assert_array_equal(view.row_anchors, anchors)
            self.assertFalse(view.packed_weights.flags.writeable)
            self.assertTrue(np.shares_memory(view.packed_weights, image._mapping))
            exact = image.exact_tensor(
                "model.layers.0.input_layernorm.weight"
            )
            self.assertEqual(exact.source_dtype, "BF16")
            self.assertEqual(exact.raw.tobytes(), bytes(range(8)))
            self.assertEqual(image.auxiliary_file("config.json").tobytes(), b"{}\n")
            receipt = image.verify_all()
            self.assertEqual(receipt["sections_verified"], 4)
            image.close()
            self.assertEqual(view.packed_weights[0], pack_rows(codes)[0])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                image.ternary_tensor(
                    "model.layers.0.mlp.down_proj.weight"
                )

    @unittest.skipUnless(native_available(), "native ternary CPU library is absent")
    def test_direct_mmap_gemv_matches_reference_and_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, codes, indices, anchors = self.fixture(Path(temporary))
            with Qwen38CQ3RuntimeImage(path) as image:
                view = image.ternary_tensor(
                    "model.layers.0.mlp.down_proj.weight"
                )
                activation = np.linspace(
                    -2.0, 3.0, self.columns, dtype=np.float32
                )
                activation_q8, activation_scales, activation_sums = quantize_q8(
                    activation
                )
                one_thread = view.gemv_q8(
                    activation_q8,
                    activation_scales,
                    activation_sums,
                    implementation="scalar_chunk16",
                )
                with ThreadPoolExecutor(max_workers=4) as executor:
                    four_threads = view.gemv_q8(
                        activation_q8,
                        activation_scales,
                        activation_sums,
                        implementation="scalar_chunk16",
                        threads=4,
                        executor=executor,
                    )
                groups = self.columns // 128
                dots = np.sum(
                    codes.reshape(self.rows, groups, 128).astype(np.int32)
                    * activation_q8.reshape(groups, 128).astype(np.int32),
                    axis=2,
                )
                ratios = view.ratio_codebook[indices].reshape(self.rows, groups)
                expected = np.sum(
                    dots.astype(np.float32)
                    * anchors.astype(np.float32)[:, None]
                    * ratios.astype(np.float32)
                    * activation_scales[None, :],
                    axis=1,
                    dtype=np.float32,
                )
                np.testing.assert_allclose(one_thread, expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(four_threads, one_thread)

    def test_payload_corruption_and_metadata_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _, _ = self.fixture(Path(temporary))
            with Qwen38CQ3Reader(path) as reader:
                offset = reader.sections["tensor.0000.codes"]["offset"]
            with path.open("r+b") as handle:
                handle.seek(offset)
                value = handle.read(1)
                handle.seek(offset)
                handle.write(bytes([value[0] ^ 1]))
            with Qwen38CQ3RuntimeImage(path) as image:
                with self.assertRaisesRegex(CQ3ContainerError, "SHA-256"):
                    image.ternary_tensor(
                        "model.layers.0.mlp.down_proj.weight"
                    )

    def test_nonternary_and_unknown_lookups_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _, _ = self.fixture(Path(temporary))
            with Qwen38CQ3RuntimeImage(path) as image:
                with self.assertRaisesRegex(CQ3RuntimeError, "not a row-normalized"):
                    image.ternary_tensor(
                        "model.layers.0.input_layernorm.weight"
                    )
                with self.assertRaisesRegex(KeyError, "unknown Qwen3.8"):
                    image.exact_tensor("missing.weight")


if __name__ == "__main__":
    unittest.main()
