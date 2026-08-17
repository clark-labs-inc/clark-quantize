from __future__ import annotations

from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_half_support_tail_cpu import (
    native_pack_half_support_tail_matrix,
)
from clark_quantize.qwen38_half_support_tail_hash_fused_cpu import (
    HalfSupportTailHashFusedCPULibrary,
    load_half_support_tail_hash_fused_cpu_library,
    native_half_support_tail_hash_fused_gemv_q8,
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
class Qwen38HalfSupportTailHashFusedCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        sources = [
            repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c",
            repo / "clark_quantize/qwen38_half_support_tail_cpu.c",
            repo / "clark_quantize/qwen38_hash_residual_cpu.c",
            repo / "clark_quantize/qwen38_half_support_tail_hash_fused_cpu.c",
        ]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"half-support-tail-hash{extension}"
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
        library = load_half_support_tail_hash_fused_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh fused half-support/hash library did not load")
        cls.native: HalfSupportTailHashFusedCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def make_case(
        self, *, rows: int = 9, columns: int = 208, sketch_dim: int = 7
    ) -> dict[str, np.ndarray | object]:
        generator = np.random.default_rng(15827 + columns)
        codes = random_codes(generator, rows, columns)
        encoded = native_pack_half_support_tail_matrix(
            codes, library=self.native.tail
        )
        groups = (columns + 127) // 128
        return {
            "codes": codes,
            "encoded": encoded,
            "indices": generator.integers(
                0, 16, size=(rows, groups), dtype=np.uint8
            ),
            "anchors": generator.uniform(0.001, 0.02, size=rows).astype(np.float16),
            "ratios": generator.uniform(0.5, 1.5, size=16).astype(np.float16),
            "activation": generator.integers(
                -127, 128, size=columns, dtype=np.int8
            ),
            "activation_scales": generator.uniform(
                0.002, 0.03, size=groups
            ).astype(np.float32),
            "buckets": generator.integers(
                0, sketch_dim, size=columns, dtype=np.uint16
            ),
            "signs": generator.choice(
                np.array([-1, 1], dtype=np.int8), size=columns
            ),
            "correction_codes": generator.integers(
                -7, 8, size=(rows, sketch_dim), dtype=np.int8
            ),
            "correction_scales": generator.uniform(
                1e-4, 0.01, size=rows
            ).astype(np.float16),
        }

    def run_case(
        self, case: dict[str, np.ndarray | object], implementation: str
    ) -> np.ndarray:
        return native_half_support_tail_hash_fused_gemv_q8(
            case["encoded"], case["indices"], case["anchors"], case["ratios"],
            case["activation"], case["activation_scales"], case["buckets"],
            case["signs"], case["correction_codes"], case["correction_scales"],
            library=self.native, implementation=implementation,
        )

    def test_scalar_fused_matches_explicit_numpy_with_tail(self) -> None:
        case = self.make_case()
        observed = self.run_case(case, "scalar")
        codes = case["codes"]
        activation = case["activation"]
        activation_scales = case["activation_scales"]
        expected = np.zeros(codes.shape[0], dtype=np.float32)
        for row in range(codes.shape[0]):
            for group, start in enumerate(range(0, codes.shape[1], 128)):
                stop = min(start + 128, codes.shape[1])
                dot = np.dot(
                    codes[row, start:stop].astype(np.int32),
                    activation[start:stop].astype(np.int32),
                )
                expected[row] += (
                    dot * np.float32(case["anchors"][row])
                    * np.float32(case["ratios"][case["indices"][row, group]])
                    * activation_scales[group]
                )
        dequantized = activation.astype(np.float32) * np.repeat(
            activation_scales, 128
        )[: activation.size]
        sketch = np.zeros(case["correction_codes"].shape[1], dtype=np.float32)
        np.add.at(sketch, case["buckets"], dequantized * case["signs"])
        expected += (
            case["correction_codes"].astype(np.float32) @ sketch
        ) * case["correction_scales"].astype(np.float32)
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)

    def test_vision_shape_avx2_matches_scalar_when_available(self) -> None:
        if not all(self.native.features().values()):
            self.skipTest("AVX2/FMA/F16C/BMI2 is unavailable")
        case = self.make_case(rows=13, columns=4304, sketch_dim=34)
        scalar = self.run_case(case, "scalar")
        avx2 = self.run_case(case, "avx2_bmi2")
        np.testing.assert_allclose(avx2, scalar, rtol=3e-6, atol=3e-6)

    def test_bad_route_and_int4_codes_are_rejected(self) -> None:
        case = self.make_case()
        bad_signs = case["signs"].copy()
        bad_signs[0] = 0
        case_with_bad_signs = {**case, "signs": bad_signs}
        with self.assertRaisesRegex(ValueError, "signs"):
            self.run_case(case_with_bad_signs, "scalar")
        bad_buckets = case["buckets"].copy()
        bad_buckets[0] = case["correction_codes"].shape[1]
        case_with_bad_buckets = {**case, "buckets": bad_buckets}
        with self.assertRaisesRegex(ValueError, "bucket"):
            self.run_case(case_with_bad_buckets, "scalar")
        bad_codes = case["correction_codes"].copy()
        bad_codes[0, 0] = -8
        case_with_bad_codes = {**case, "correction_codes": bad_codes}
        with self.assertRaisesRegex(ValueError, r"\[-7, 7\]"):
            self.run_case(case_with_bad_codes, "scalar")


if __name__ == "__main__":
    unittest.main()
