from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    Mask64CPULibrary,
    load_mask64_cpu_library,
    mask64_hash_component_storage,
    native_mask64_gemv_q8_row_scale4,
    native_pack_mask64_matrix,
    native_unpack_mask64_matrix,
)


def random_codes(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        positions = generator.choice(128, size=64, replace=False)
        group[positions] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=64
        )
    return codes


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38FixedSupportMask64CPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        source = repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"mask64{extension}"
        shared_flags = (
            ["-dynamiclib"] if platform.system() == "Darwin"
            else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared_flags, str(source), "-o", str(output),
            ],
            check=True,
        )
        library = load_mask64_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh mask64 library did not load")
        cls.native: Mask64CPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_payload_is_exactly_one_point_five_bits_per_weight(self) -> None:
        generator = np.random.default_rng(158)
        codes = random_codes(generator, 7, 384)
        encoded = native_pack_mask64_matrix(codes, library=self.native)
        self.assertEqual(len(encoded.payload) * 8 / codes.size, 1.5)
        np.testing.assert_array_equal(
            native_unpack_mask64_matrix(encoded, library=self.native), codes
        )

    def test_canonical_support_and_sign_order_are_stable(self) -> None:
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, 0::2] = np.where(np.arange(64) & 1, 1, -1)
        encoded = native_pack_mask64_matrix(codes, library=self.native)
        self.assertEqual(encoded.payload[:16], bytes([0x55] * 16))
        self.assertEqual(encoded.payload[16:], bytes([0xAA] * 8))
        np.testing.assert_array_equal(
            native_unpack_mask64_matrix(encoded, library=self.native), codes
        )

    def test_bad_code_support_and_payload_are_rejected(self) -> None:
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :63] = 1
        with self.assertRaisesRegex(RuntimeError, "support"):
            native_pack_mask64_matrix(codes, library=self.native)
        codes[0, 63] = 2
        with self.assertRaisesRegex(ValueError, "lie in"):
            native_pack_mask64_matrix(codes, library=self.native)
        codes[0, 63] = 1
        encoded = native_pack_mask64_matrix(codes, library=self.native)
        corrupted = replace(
            encoded,
            payload=bytes([encoded.payload[0] & 0xFE]) + encoded.payload[1:],
        )
        with self.assertRaisesRegex(RuntimeError, "support"):
            native_unpack_mask64_matrix(corrupted, library=self.native)

    def test_scalar_gemv_matches_dequantized_numpy_oracle(self) -> None:
        generator = np.random.default_rng(27)
        rows, columns = 7, 256
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_mask64_matrix(codes, library=self.native)
        groups = columns // 128
        indices = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
        anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
        ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
        activation = generator.integers(-100, 101, size=columns, dtype=np.int8)
        activation_scales = generator.uniform(0.002, 0.03, size=groups).astype(
            np.float32
        )
        observed = native_mask64_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, activation_scales,
            library=self.native, implementation="scalar",
        )
        expected = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            for group in range(groups):
                start = group * 128
                dot = np.dot(
                    codes[row, start : start + 128].astype(np.int32),
                    activation[start : start + 128].astype(np.int32),
                )
                expected[row] += (
                    dot * np.float32(anchors[row])
                    * np.float32(ratios[indices[row, group]])
                    * activation_scales[group]
                )
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)

    def test_avx2_bmi2_matches_scalar_when_available(self) -> None:
        features = self.native.features()
        if not all(features.values()):
            self.skipTest("AVX2/FMA/F16C/BMI2 is unavailable")
        generator = np.random.default_rng(38)
        rows, columns = 19, 384
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_mask64_matrix(codes, library=self.native)
        groups = columns // 128
        indices = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
        anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
        ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
        activation = generator.integers(-127, 128, size=columns, dtype=np.int8)
        scales = generator.uniform(0.002, 0.03, size=groups).astype(np.float32)
        scalar = native_mask64_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, scales,
            library=self.native, implementation="scalar",
        )
        avx2 = native_mask64_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, scales,
            library=self.native, implementation="avx2_bmi2",
        )
        np.testing.assert_allclose(avx2, scalar, rtol=2e-6, atol=2e-6)

    def test_complete_mask64_hash_component_stays_below_one_point_57(self) -> None:
        gate = mask64_hash_component_storage(
            input_dim=5120, output_rows=17408, sketch_dim=40
        )
        down = mask64_hash_component_storage(
            input_dim=17408, output_rows=5120, sketch_dim=136
        )
        self.assertEqual(gate["ternary_code_bytes"] * 8 / (5120 * 17408), 1.5)
        self.assertGreater(gate["routing_bytes"], 0)
        self.assertLess(gate["bits_per_weight"], 1.57)
        self.assertLess(down["bits_per_weight"], 1.57)


if __name__ == "__main__":
    unittest.main()
