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
    FixedSupportPayload,
    fixed_support_state_count,
    pack_fixed_support_matrix,
)
from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    native_pack_mask64_matrix,
)
from clark_quantize.qwen38_gain_hash_layer_codec import (
    GainHashLayerComponents,
    materialize_mask64_gain_hash_layer,
)
from clark_quantize.qwen38_gain_hash_layer_transcode import (
    GainHashLayerTranscodeCPULibrary,
    load_gain_hash_layer_transcode_cpu_library,
    native_materialize_mask64_gain_hash_layer,
    native_transcode_fixed_support_mask64,
)
from clark_quantize.qwen38_gain_route_codec import pack_gain_hash_route_body
from clark_quantize.qwen38_mask64_gain_hash_fused_cpu import (
    Mask64GainHashFusedCPULibrary,
    load_mask64_gain_hash_fused_cpu_library,
)


def random_fixed_support(
    generator: np.random.Generator, *, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        support = generator.choice(128, size=64, replace=False)
        group[support] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=64
        )
    return codes


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38GainHashLayerTranscodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"gain-layer-transcode{extension}"
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
                str(repo / "clark_quantize/qwen38_gain_hash_layer_transcode.c"),
                str(repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"),
                str(repo / "clark_quantize/qwen38_mask64_gain_hash_fused_cpu.c"),
                "-o",
                str(output),
            ],
            check=True,
        )
        transcode = load_gain_hash_layer_transcode_cpu_library(output)
        fused = load_mask64_gain_hash_fused_cpu_library(output)
        if transcode is None or fused is None:
            raise RuntimeError("fresh gain-hash transcode library did not load")
        cls.transcode: GainHashLayerTranscodeCPULibrary = transcode
        cls.fused: Mask64GainHashFusedCPULibrary = fused

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_native_transcode_is_byte_identical_to_dense_mask64_pack(self) -> None:
        generator = np.random.default_rng(38601)
        for rows, columns in ((1, 128), (7, 384), (9, 512)):
            codes = random_fixed_support(
                generator, rows=rows, columns=columns
            )
            stored = pack_fixed_support_matrix(
                codes, group_size=128, nonzero_per_group=64
            )
            observed = native_transcode_fixed_support_mask64(
                stored, library=self.transcode
            )
            expected = native_pack_mask64_matrix(
                codes, library=self.fused.mask
            )
            self.assertEqual(observed, expected)

    def test_complete_native_materialization_matches_dense_control(self) -> None:
        generator = np.random.default_rng(38602)
        rows, columns, sketch_dim = 5, 256, 7
        codes = random_fixed_support(generator, rows=rows, columns=columns)
        route = pack_gain_hash_route_body(
            generator.integers(0, sketch_dim, size=columns, dtype=np.uint16),
            generator.choice(
                np.array([-1, 1], dtype=np.int8), size=columns
            ),
            generator.integers(0, 8, size=columns, dtype=np.uint8),
            np.sort(generator.uniform(0.0, 2.0, size=8).astype(np.float16)),
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        components = GainHashLayerComponents(
            ternary=pack_fixed_support_matrix(
                codes, group_size=128, nonzero_per_group=64
            ),
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
        observed = native_materialize_mask64_gain_hash_layer(
            components, library=self.transcode
        )
        expected = materialize_mask64_gain_hash_layer(
            components, library=self.fused.mask
        )
        self.assertEqual(observed.ternary, expected.ternary)
        for observed_array, expected_array in (
            (observed.scale_indices, expected.scale_indices),
            (observed.row_anchors, expected.row_anchors),
            (observed.ratio_codebook, expected.ratio_codebook),
            (observed.correction_codes, expected.correction_codes),
            (
                observed.correction_row_scales,
                expected.correction_row_scales,
            ),
        ):
            np.testing.assert_array_equal(observed_array, expected_array)
        self.assertIs(observed.route, expected.route)

    def test_noncanonical_rank_padding_and_metadata_are_rejected(self) -> None:
        state_count = fixed_support_state_count(128, 64)
        invalid_rank = FixedSupportPayload(
            payload=state_count.to_bytes(24, "little"),
            bit_length=189,
            rows=1,
            columns=128,
            group_size=128,
            nonzero_per_group=64,
            group_width_bits=189,
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_transcode_fixed_support_mask64(
                invalid_rank, library=self.transcode
            )

        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :64] = 1
        encoded = pack_fixed_support_matrix(
            codes, group_size=128, nonzero_per_group=64
        )
        bad_padding = replace(
            encoded,
            payload=encoded.payload[:-1]
            + bytes((encoded.payload[-1] | 0x80,)),
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_transcode_fixed_support_mask64(
                bad_padding, library=self.transcode
            )
        for corrupted in (
            replace(encoded, group_size=64),
            replace(encoded, nonzero_per_group=63),
            replace(encoded, group_width_bits=188),
            replace(encoded, bit_length=188),
            replace(encoded, payload=encoded.payload[:-1]),
        ):
            with self.assertRaisesRegex(ValueError, "metadata"):
                native_transcode_fixed_support_mask64(
                    corrupted, library=self.transcode
                )


if __name__ == "__main__":
    unittest.main()
