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
    native_pack_mask64_matrix,
)
from clark_quantize.qwen38_half_support_tail_cpu import (
    HalfSupportTailCPULibrary,
    load_half_support_tail_cpu_library,
    native_half_support_tail_gemv_q8_row_scale4,
    native_pack_half_support_tail_matrix,
    native_unpack_half_support_tail_matrix,
)


def random_codes(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for start in range(0, columns, 128):
        width = min(128, columns - start)
        for row in range(rows):
            positions = generator.choice(width, size=width // 2, replace=False)
            codes[row, start + positions] = generator.choice(
                np.array([-1, 1], dtype=np.int8), size=width // 2
            )
    return codes


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38HalfSupportTailCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        sources = [
            repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c",
            repo / "clark_quantize/qwen38_half_support_tail_cpu.c",
        ]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"half-support-tail{extension}"
        shared_flags = (
            ["-dynamiclib"] if platform.system() == "Darwin"
            else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared_flags, *(str(source) for source in sources),
                "-o", str(output),
            ],
            check=True,
        )
        library = load_half_support_tail_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh half-support tail library did not load")
        cls.native: HalfSupportTailCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_vision_tail_payload_is_exactly_one_point_five_bpw(self) -> None:
        generator = np.random.default_rng(158)
        codes = random_codes(generator, 3, 4304)
        encoded = native_pack_half_support_tail_matrix(codes, library=self.native)
        self.assertEqual(len(encoded.payload), 3 * 807)
        self.assertEqual(len(encoded.payload) * 8 / codes.size, 1.5)
        self.assertEqual(encoded.tail_width, 80)
        np.testing.assert_array_equal(
            native_unpack_half_support_tail_matrix(encoded, library=self.native),
            codes,
        )

    def test_full_group_layout_matches_mask64(self) -> None:
        generator = np.random.default_rng(27)
        codes = random_codes(generator, 1, 208)
        tail = native_pack_half_support_tail_matrix(codes, library=self.native)
        mask = native_pack_mask64_matrix(codes[:, :128], library=self.native.mask)
        self.assertEqual(tail.payload[:24], mask.payload)
        self.assertEqual(len(tail.payload[24:]), 15)

    def test_bad_support_and_padding_are_rejected(self) -> None:
        generator = np.random.default_rng(38)
        codes = random_codes(generator, 1, 130)
        encoded = native_pack_half_support_tail_matrix(codes, library=self.native)
        corrupted_support = replace(
            encoded,
            payload=encoded.payload[:24]
            + bytes([encoded.payload[24] ^ 1])
            + encoded.payload[25:],
        )
        with self.assertRaisesRegex(RuntimeError, "support"):
            native_unpack_half_support_tail_matrix(
                corrupted_support, library=self.native
            )
        corrupted_padding = replace(
            encoded,
            payload=encoded.payload[:-1] + bytes([encoded.payload[-1] | 0x80]),
        )
        with self.assertRaisesRegex(RuntimeError, "support"):
            native_unpack_half_support_tail_matrix(
                corrupted_padding, library=self.native
            )

    def test_scalar_tail_gemv_matches_numpy(self) -> None:
        generator = np.random.default_rng(350)
        rows, columns = 7, 208
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_half_support_tail_matrix(codes, library=self.native)
        groups = 2
        indices = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
        anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
        ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
        activation = generator.integers(-127, 128, size=columns, dtype=np.int8)
        scales = generator.uniform(0.002, 0.03, size=groups).astype(np.float32)
        observed = native_half_support_tail_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, scales,
            library=self.native, implementation="scalar",
        )
        expected = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            for group, start in enumerate((0, 128)):
                stop = min(start + 128, columns)
                dot = np.dot(
                    codes[row, start:stop].astype(np.int32),
                    activation[start:stop].astype(np.int32),
                )
                expected[row] += (
                    dot * np.float32(anchors[row])
                    * np.float32(ratios[indices[row, group]]) * scales[group]
                )
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)

    def test_vision_avx2_matches_scalar_when_available(self) -> None:
        if not all(self.native.features().values()):
            self.skipTest("AVX2/FMA/F16C/BMI2 is unavailable")
        generator = np.random.default_rng(4304)
        rows, columns = 5, 4304
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_half_support_tail_matrix(codes, library=self.native)
        groups = 34
        indices = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
        anchors = generator.uniform(0.001, 0.02, size=rows).astype(np.float16)
        ratios = generator.uniform(0.5, 1.5, size=16).astype(np.float16)
        activation = generator.integers(-127, 128, size=columns, dtype=np.int8)
        scales = generator.uniform(0.002, 0.03, size=groups).astype(np.float32)
        scalar = native_half_support_tail_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, scales,
            library=self.native, implementation="scalar",
        )
        avx2 = native_half_support_tail_gemv_q8_row_scale4(
            encoded, indices, anchors, ratios, activation, scales,
            library=self.native, implementation="avx2_bmi2",
        )
        np.testing.assert_allclose(avx2, scalar, rtol=3e-6, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
