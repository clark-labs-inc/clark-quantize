from __future__ import annotations

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
from clark_quantize.qwen38_mask64_hash_fused_cpu import (
    Mask64HashFusedCPULibrary,
    load_mask64_hash_fused_cpu_library,
    native_learned_projection_q8,
    native_mask64_hash_fused_gemv_q8,
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
class Qwen38Mask64HashFusedCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        sources = [
            repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c",
            repo / "clark_quantize/qwen38_mask64_hash_fused_cpu.c",
        ]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"mask64-hash{extension}"
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
        library = load_mask64_hash_fused_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh fused mask64/hash library did not load")
        cls.native: Mask64HashFusedCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def make_case(self) -> dict[str, np.ndarray | object]:
        generator = np.random.default_rng(38027)
        rows, columns, sketch_dim = 11, 384, 7
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_mask64_matrix(codes, library=self.native.mask)
        groups = columns // 128
        return {
            "generator": generator,
            "codes": codes,
            "encoded": encoded,
            "indices": generator.integers(0, 16, size=(rows, groups), dtype=np.uint8),
            "anchors": generator.uniform(0.001, 0.02, size=rows).astype(np.float16),
            "ratios": generator.uniform(0.5, 1.5, size=16).astype(np.float16),
            "activation": generator.integers(-127, 128, size=columns, dtype=np.int8),
            "activation_scales": generator.uniform(0.002, 0.03, size=groups).astype(np.float32),
            "buckets": generator.integers(0, sketch_dim, size=columns, dtype=np.uint16),
            "signs": generator.choice(np.array([-1, 1], dtype=np.int8), size=columns),
            "correction_codes": generator.integers(-7, 8, size=(rows, sketch_dim), dtype=np.int8),
            "correction_scales": generator.uniform(1e-4, 0.01, size=rows).astype(np.float16),
        }

    def test_q8_projection_matches_dequantized_numpy(self) -> None:
        case = self.make_case()
        activation = case["activation"]
        scales = case["activation_scales"]
        buckets = case["buckets"]
        signs = case["signs"]
        observed = native_learned_projection_q8(
            activation, scales, buckets, signs, sketch_dim=7,
            library=self.native,
        )
        expected = np.zeros(7, dtype=np.float32)
        values = activation.astype(np.float32) * np.repeat(scales, 128)
        np.add.at(expected, buckets, values * signs)
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)

    def test_scalar_fused_matches_explicit_numpy(self) -> None:
        case = self.make_case()
        observed = native_mask64_hash_fused_gemv_q8(
            case["encoded"], case["indices"], case["anchors"], case["ratios"],
            case["activation"], case["activation_scales"], case["buckets"],
            case["signs"], case["correction_codes"], case["correction_scales"],
            library=self.native, implementation="scalar",
        )
        codes = case["codes"]
        activation = case["activation"]
        scales = case["activation_scales"]
        expected = np.zeros(codes.shape[0], dtype=np.float32)
        for row in range(codes.shape[0]):
            for group in range(codes.shape[1] // 128):
                start = group * 128
                dot = np.dot(
                    codes[row, start : start + 128].astype(np.int32),
                    activation[start : start + 128].astype(np.int32),
                )
                expected[row] += (
                    dot * np.float32(case["anchors"][row])
                    * np.float32(case["ratios"][case["indices"][row, group]])
                    * scales[group]
                )
        sketch = np.zeros(case["correction_codes"].shape[1], dtype=np.float32)
        dequantized = activation.astype(np.float32) * np.repeat(scales, 128)
        np.add.at(sketch, case["buckets"], dequantized * case["signs"])
        expected += (
            case["correction_codes"].astype(np.float32) @ sketch
        ) * case["correction_scales"].astype(np.float32)
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)

    def test_avx2_fused_matches_scalar_when_available(self) -> None:
        if not all(self.native.features().values()):
            self.skipTest("AVX2/FMA/F16C/BMI2 is unavailable")
        case = self.make_case()
        arguments = (
            case["encoded"], case["indices"], case["anchors"], case["ratios"],
            case["activation"], case["activation_scales"], case["buckets"],
            case["signs"], case["correction_codes"], case["correction_scales"],
        )
        scalar = native_mask64_hash_fused_gemv_q8(
            *arguments, library=self.native, implementation="scalar"
        )
        avx2 = native_mask64_hash_fused_gemv_q8(
            *arguments, library=self.native, implementation="avx2_bmi2"
        )
        np.testing.assert_allclose(avx2, scalar, rtol=3e-6, atol=3e-6)

    def test_bad_route_and_int4_codes_are_rejected(self) -> None:
        case = self.make_case()
        bad_signs = case["signs"].copy()
        bad_signs[0] = 0
        with self.assertRaisesRegex(ValueError, "signs"):
            native_mask64_hash_fused_gemv_q8(
                case["encoded"], case["indices"], case["anchors"], case["ratios"],
                case["activation"], case["activation_scales"], case["buckets"],
                bad_signs, case["correction_codes"], case["correction_scales"],
                library=self.native,
            )
        bad_codes = case["correction_codes"].copy()
        bad_codes[0, 0] = -8
        with self.assertRaisesRegex(ValueError, r"\[-7, 7\]"):
            native_mask64_hash_fused_gemv_q8(
                case["encoded"], case["indices"], case["anchors"], case["ratios"],
                case["activation"], case["activation_scales"], case["buckets"],
                case["signs"], bad_codes, case["correction_scales"],
                library=self.native,
            )


if __name__ == "__main__":
    unittest.main()
