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
from clark_quantize.qwen38_gain_route_codec import pack_gain_hash_route_body
from clark_quantize.qwen38_mask64_gain_hash_fused_cpu import (
    Mask64GainHashFusedCPULibrary,
    load_mask64_gain_hash_fused_cpu_library,
    native_gain_projection_q8,
    native_mask64_gain_hash_fused_gemv_q8,
)
from clark_quantize.qwen38_mask64_packed_gain_hash_fused_cpu import (
    native_mask64_packed_gain_hash_fused_gemv_q8,
    native_packed_gain_projection_q8,
)


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38Mask64PackedGainHashFusedCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"packed-gain-hash{extension}"
        shared_flags = (
            ["-dynamiclib"] if platform.system() == "Darwin"
            else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared_flags,
                str(repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"),
                str(repo / "clark_quantize/qwen38_mask64_gain_hash_fused_cpu.c"),
                "-o", str(output),
            ],
            check=True,
        )
        library = load_mask64_gain_hash_fused_cpu_library(output)
        if library is None:
            raise RuntimeError("packed gain-route fused library did not load")
        cls.native: Mask64GainHashFusedCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def make_case(self):
        generator = np.random.default_rng(38227)
        rows, columns, sketch_dim = 9, 384, 7
        codes = np.zeros((rows, columns), dtype=np.int8)
        for group in codes.reshape(-1, 128):
            positions = generator.choice(128, size=64, replace=False)
            group[positions] = generator.choice(
                np.array([-1, 1], dtype=np.int8), size=64
            )
        encoded = native_pack_mask64_matrix(codes, library=self.native.mask)
        groups = columns // 128
        buckets = generator.integers(
            0, sketch_dim, size=columns, dtype=np.uint16
        )
        signs = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=columns
        )
        gain_codes = generator.integers(0, 8, size=columns, dtype=np.uint8)
        gain_codebook = np.sort(
            generator.uniform(0.25, 2.0, size=8).astype(np.float16)
        )
        route = pack_gain_hash_route_body(
            buckets,
            signs,
            gain_codes,
            gain_codebook,
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        return {
            "encoded": encoded,
            "indices": generator.integers(
                0, 16, size=(rows, groups), dtype=np.uint8
            ),
            "anchors": generator.uniform(0.001, 0.02, size=rows).astype(np.float16),
            "ratios": np.sort(
                generator.uniform(0.5, 1.5, size=16).astype(np.float16)
            ),
            "activation": generator.integers(-127, 128, size=columns, dtype=np.int8),
            "activation_scales": generator.uniform(0.002, 0.03, size=groups).astype(np.float32),
            "buckets": buckets,
            "signs": signs,
            "gain_codes": gain_codes,
            "gain_codebook": gain_codebook,
            "route": route,
            "correction_codes": generator.integers(
                -7, 8, size=(rows, sketch_dim), dtype=np.int8
            ),
            "correction_scales": generator.uniform(1e-4, 0.01, size=rows).astype(np.float16),
        }

    def test_packed_projection_is_identical_to_expanded_projection(self) -> None:
        case = self.make_case()
        packed = native_packed_gain_projection_q8(
            case["activation"],
            case["activation_scales"],
            case["route"],
            library=self.native,
        )
        expanded = native_gain_projection_q8(
            case["activation"],
            case["activation_scales"],
            case["buckets"],
            case["signs"],
            case["gain_codes"],
            case["gain_codebook"],
            sketch_dim=case["route"].sketch_dim,
            library=self.native,
        )
        np.testing.assert_array_equal(packed, expanded)

    def test_packed_fused_call_is_identical_to_expanded_call(self) -> None:
        case = self.make_case()
        common = (
            case["encoded"], case["indices"], case["anchors"], case["ratios"],
            case["activation"], case["activation_scales"],
        )
        packed = native_mask64_packed_gain_hash_fused_gemv_q8(
            *common,
            case["route"],
            case["correction_codes"],
            case["correction_scales"],
            library=self.native,
        )
        expanded = native_mask64_gain_hash_fused_gemv_q8(
            *common,
            case["buckets"],
            case["signs"],
            case["gain_codes"],
            case["gain_codebook"],
            case["correction_codes"],
            case["correction_scales"],
            library=self.native,
        )
        np.testing.assert_array_equal(packed, expanded)

    def test_packed_route_shape_mismatch_is_rejected(self) -> None:
        case = self.make_case()
        bad_codes = case["correction_codes"][:, :-1]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            native_mask64_packed_gain_hash_fused_gemv_q8(
                case["encoded"], case["indices"], case["anchors"], case["ratios"],
                case["activation"], case["activation_scales"], case["route"],
                bad_codes, case["correction_scales"], library=self.native,
            )


if __name__ == "__main__":
    unittest.main()
