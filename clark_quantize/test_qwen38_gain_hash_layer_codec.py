from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_fixed_support_codec import (
    pack_fixed_support_matrix,
    unpack_fixed_support_matrix,
)
from clark_quantize.qwen38_cq3_container import (
    Qwen38CQ3Reader,
    Qwen38CQ3Writer,
)
from clark_quantize.qwen38_gain_hash_layer_codec import (
    HEADER_BYTES,
    GainHashLayerComponents,
    deserialize_gain_hash_layer,
    gain_hash_layer_body_layout,
    gain_hash_layer_storage_bits_per_weight,
    materialize_mask64_gain_hash_layer,
    pack_gain_hash_layer,
)
from clark_quantize.qwen38_gain_hash_residual import gain_hash_storage_projection
from clark_quantize.qwen38_gain_route_codec import (
    pack_gain_hash_route_body,
    unpack_gain_hash_route_body,
)
from clark_quantize.qwen38_mask64_gain_hash_fused_cpu import (
    Mask64GainHashFusedCPULibrary,
    load_mask64_gain_hash_fused_cpu_library,
)
from clark_quantize.qwen38_mask64_packed_gain_hash_fused_cpu import (
    native_mask64_packed_gain_hash_fused_gemv_q8,
)


class Qwen38GainHashLayerCodecTests(unittest.TestCase):
    @staticmethod
    def make_case(seed: int = 38501) -> tuple[GainHashLayerComponents, np.ndarray]:
        generator = np.random.default_rng(seed)
        rows, columns, sketch_dim = 3, 256, 5
        codes = np.zeros((rows, columns), dtype=np.int8)
        for group in codes.reshape(-1, 128):
            support = generator.choice(128, size=64, replace=False)
            group[support] = generator.choice(
                np.array([-1, 1], dtype=np.int8), size=64
            )
        ternary = pack_fixed_support_matrix(
            codes, group_size=128, nonzero_per_group=64
        )
        buckets = generator.integers(
            0, sketch_dim, size=columns, dtype=np.uint16
        )
        signs = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=columns
        )
        gains = generator.integers(0, 8, size=columns, dtype=np.uint8)
        gain_codebook = np.sort(
            generator.uniform(0.0, 2.0, size=8).astype(np.float16)
        )
        route = pack_gain_hash_route_body(
            buckets,
            signs,
            gains,
            gain_codebook,
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        components = GainHashLayerComponents(
            ternary=ternary,
            scale_indices=generator.integers(
                0, 16, size=(rows, columns // 128), dtype=np.uint8
            ),
            row_anchors=generator.uniform(0.001, 0.02, size=rows).astype(
                np.float16
            ),
            ratio_codebook=np.sort(
                generator.uniform(0.5, 1.5, size=16).astype(np.float16)
            ),
            route=route,
            correction_codes=generator.integers(
                -7, 8, size=(rows, sketch_dim), dtype=np.int8
            ),
            correction_row_scales=generator.uniform(
                0.0001, 0.01, size=rows
            ).astype(np.float16),
        )
        return components, codes

    def test_complete_layer_roundtrip_is_exact_and_canonical(self) -> None:
        source, ternary_codes = self.make_case()
        body = pack_gain_hash_layer(source)
        restored = deserialize_gain_hash_layer(body)
        np.testing.assert_array_equal(
            unpack_fixed_support_matrix(restored.ternary), ternary_codes
        )
        np.testing.assert_array_equal(restored.scale_indices, source.scale_indices)
        np.testing.assert_array_equal(restored.row_anchors, source.row_anchors)
        np.testing.assert_array_equal(restored.ratio_codebook, source.ratio_codebook)
        np.testing.assert_array_equal(
            restored.correction_codes, source.correction_codes
        )
        np.testing.assert_array_equal(
            restored.correction_row_scales, source.correction_row_scales
        )
        self.assertEqual(pack_gain_hash_layer(restored), body)

    def test_body_bytes_match_gain_hash_storage_projection(self) -> None:
        for rows, columns, sketch_dim in (
            (17408, 5120, 40),
            (5120, 17408, 136),
        ):
            layout = gain_hash_layer_body_layout(
                rows=rows,
                columns=columns,
                sketch_dim=sketch_dim,
                gain_bits=3,
            )
            ledger = gain_hash_storage_projection(
                input_dim=columns,
                output_rows=rows,
                sketch_dim=sketch_dim,
                gain_bits=3,
                nonzero_per_group=64,
                group=128,
                header_bytes=HEADER_BYTES,
            )
            self.assertEqual(layout["body_bytes"], ledger["total_bytes"])
            self.assertEqual(
                gain_hash_layer_storage_bits_per_weight(
                    rows=rows, columns=columns, sketch_dim=sketch_dim
                ),
                ledger["bits_per_weight"],
            )

    def test_header_reserved_bytes_and_lengths_are_authenticated(self) -> None:
        source, _ = self.make_case()
        body = bytearray(pack_gain_hash_layer(source))
        body[HEADER_BYTES - 1] = 1
        with self.assertRaisesRegex(ValueError, "header"):
            deserialize_gain_hash_layer(body)
        body = bytearray(pack_gain_hash_layer(source))
        body[32] ^= 1
        with self.assertRaisesRegex(ValueError, "ledger"):
            deserialize_gain_hash_layer(body)

    def test_nonzero_ternary_and_int4_padding_are_rejected(self) -> None:
        source, _ = self.make_case()
        layout = gain_hash_layer_body_layout(
            rows=source.rows,
            columns=source.columns,
            sketch_dim=source.sketch_dim,
        )
        body = bytearray(pack_gain_hash_layer(source))
        ternary_last = HEADER_BYTES + layout["ternary_bytes"] - 1
        self.assertNotEqual(layout["ternary_bits"] % 8, 0)
        body[ternary_last] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            deserialize_gain_hash_layer(body)

        body = bytearray(pack_gain_hash_layer(source))
        correction_offset = HEADER_BYTES + sum(
            layout[name]
            for name in (
                "ternary_bytes",
                "scale_index_bytes",
                "row_anchor_bytes",
                "ratio_codebook_bytes",
                "route_body_bytes",
            )
        )
        body[correction_offset + (source.sketch_dim + 1) // 2 - 1] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding nibble"):
            deserialize_gain_hash_layer(body)

    def test_forbidden_negative_eight_and_negative_zero_are_rejected(self) -> None:
        source, _ = self.make_case()
        layout = gain_hash_layer_body_layout(
            rows=source.rows,
            columns=source.columns,
            sketch_dim=source.sketch_dim,
        )
        body = bytearray(pack_gain_hash_layer(source))
        correction_offset = HEADER_BYTES + sum(
            layout[name]
            for name in (
                "ternary_bytes",
                "scale_index_bytes",
                "row_anchor_bytes",
                "ratio_codebook_bytes",
                "route_body_bytes",
            )
        )
        body[correction_offset] = (body[correction_offset] & 0xF0) | 8
        with self.assertRaisesRegex(ValueError, "forbidden"):
            deserialize_gain_hash_layer(body)

        bad = replace(
            source,
            row_anchors=np.array([-0.0, 0.01, 0.02], dtype=np.float16),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            pack_gain_hash_layer(bad)

    def test_complete_layer_is_an_authenticated_cq3_tensor_section(self) -> None:
        source, _ = self.make_case()
        body = pack_gain_hash_layer(source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen38-gain-layer.cq3"
            with Qwen38CQ3Writer(
                path,
                model={"id": "Qwen/Qwen3.8-27B", "revision": "test-revision"},
                strategy={"name": "fixed-support-gain-hash-v1"},
                parameter_count=source.rows * source.columns,
            ) as writer:
                writer.add_section(
                    "layer0.gate.weight",
                    "gain_hash_layer_v1",
                    body,
                    metadata={
                        "rows": source.rows,
                        "columns": source.columns,
                        "sketch_dim": source.sketch_dim,
                    },
                )
                writer.finalize(
                    tensors=[
                        {
                            "name": "model.language_model.layers.0.mlp.gate_proj.weight",
                            "shape": [source.rows, source.columns],
                            "parameters": source.rows * source.columns,
                            "sections": ["layer0.gate.weight"],
                        }
                    ]
                )
            with Qwen38CQ3Reader(path) as reader:
                self.assertTrue(reader.verify_section("layer0.gate.weight"))
                restored_body = reader.read_section("layer0.gate.weight")
                self.assertEqual(restored_body, body)
                self.assertEqual(pack_gain_hash_layer(
                    deserialize_gain_hash_layer(restored_body)
                ), body)


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38GainHashLayerNativeMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"gain-layer{extension}"
        shared_flags = (
            ["-dynamiclib"]
            if platform.system() == "Darwin"
            else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                *shared_flags,
                str(repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"),
                str(repo / "clark_quantize/qwen38_mask64_gain_hash_fused_cpu.c"),
                "-o",
                str(output),
            ],
            check=True,
        )
        library = load_mask64_gain_hash_fused_cpu_library(output)
        if library is None:
            raise RuntimeError("gain-hash complete-layer native library did not load")
        cls.native: Mask64GainHashFusedCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_complete_body_materializes_into_fused_kernel_exactly(self) -> None:
        source, _ = Qwen38GainHashLayerCodecTests.make_case(38502)
        restored = deserialize_gain_hash_layer(pack_gain_hash_layer(source))
        compute = materialize_mask64_gain_hash_layer(
            restored, library=self.native.mask
        )
        generator = np.random.default_rng(38503)
        activation = generator.integers(
            -127, 128, size=source.columns, dtype=np.int8
        )
        activation_scales = generator.uniform(
            0.002, 0.03, size=source.columns // 128
        ).astype(np.float32)
        observed = native_mask64_packed_gain_hash_fused_gemv_q8(
            compute.ternary,
            compute.scale_indices,
            compute.row_anchors,
            compute.ratio_codebook,
            activation,
            activation_scales,
            compute.route,
            compute.correction_codes,
            compute.correction_row_scales,
            library=self.native,
            implementation="scalar",
        )
        ternary_codes = unpack_fixed_support_matrix(restored.ternary).astype(
            np.float32
        )
        activation_f32 = (
            activation.reshape(-1, 128).astype(np.float32)
            * activation_scales[:, None]
        ).reshape(-1)
        group_scales = (
            compute.row_anchors.astype(np.float32)[:, None]
            * compute.ratio_codebook.astype(np.float32)[compute.scale_indices]
        )
        base = (
            ternary_codes.reshape(source.rows, -1, 128)
            * group_scales[:, :, None]
            * activation_f32.reshape(1, -1, 128)
        ).sum(axis=(1, 2), dtype=np.float32)
        # Decode through the canonical route API to keep the reference
        # independent of the compute adapter's widened arrays.
        bucket_values, sign_values, gain_codes = unpack_gain_hash_route_body(
            restored.route
        )
        gain_values = restored.route.gain_codebook.astype(np.float32)[gain_codes]
        sketch = np.zeros(source.sketch_dim, dtype=np.float32)
        np.add.at(
            sketch,
            bucket_values,
            activation_f32 * sign_values.astype(np.float32) * gain_values,
        )
        correction = (
            compute.correction_codes.astype(np.float32)
            * compute.correction_row_scales.astype(np.float32)[:, None]
        ) @ sketch
        np.testing.assert_allclose(
            observed, base + correction, rtol=2e-5, atol=5e-4
        )


if __name__ == "__main__":
    unittest.main()
