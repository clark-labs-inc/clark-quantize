from __future__ import annotations

import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from clark_quantize.qwen38_hash_residual import (
    expand_sketch_rows,
    sketch_rows,
    sparse_jl_projection,
)
from clark_quantize.qwen38_hash_residual_cpu import (
    HashResidualCPULibrary,
    load_hash_residual_cpu_library,
    native_hash_int4_gemv_f32,
    native_hash_projection_f32,
    native_hash_residual_f32,
    pack_signed_int4_rows,
)


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38HashResidualCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        source = repo / "clark_quantize/qwen38_hash_residual_cpu.c"
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"hash-residual{extension}"
        shared_flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared_flags, str(source), "-lm", "-o", str(output),
            ],
            check=True,
        )
        library = load_hash_residual_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh hash-residual library did not load")
        cls.native: HashResidualCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_signed_int4_packing_is_row_aligned_and_canonical(self) -> None:
        codes = np.array(
            [[-7, -1, 0, 1, 7], [7, 1, 0, -1, -7]], dtype=np.int8
        )
        packed = pack_signed_int4_rows(codes)
        np.testing.assert_array_equal(
            packed,
            np.array([[0xF9, 0x10, 0x07], [0x17, 0xF0, 0x09]], dtype=np.uint8),
        )
        self.assertEqual(int(packed[0, -1] >> 4), 0)

    def test_native_projection_matches_frozen_clark_hash_algebra(self) -> None:
        projection = sparse_jl_projection(65, 11, hashes_per_input=4, seed=38)
        generator = torch.Generator().manual_seed(158)
        values = torch.randn(65, generator=generator, dtype=torch.float32)
        observed = native_hash_projection_f32(
            values.numpy(),
            projection.buckets.numpy(),
            projection.signs.numpy(),
            sketch_dim=projection.sketch_dim,
            library=self.native,
        )
        expected = sketch_rows(values[None, :], projection)[0].numpy()
        np.testing.assert_array_equal(observed, expected)

    def test_native_int4_scalar_matches_exported_coefficient_algebra(self) -> None:
        generator = np.random.default_rng(27)
        for columns in (1, 15, 16, 17, 40, 80, 160):
            codes = generator.integers(-7, 8, size=(19, columns), dtype=np.int8)
            scales = generator.uniform(1e-4, 0.25, size=19).astype(np.float16)
            sketch = generator.normal(size=columns).astype(np.float32)
            observed = native_hash_int4_gemv_f32(
                pack_signed_int4_rows(codes),
                scales,
                sketch,
                sketch_dim=columns,
                library=self.native,
                implementation="scalar",
            )
            expected = (codes.astype(np.float32) @ sketch) * scales.astype(np.float32)
            np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)

    def test_native_runtime_matches_expanded_dense_correction(self) -> None:
        projection = sparse_jl_projection(128, 16, hashes_per_input=4, seed=6840346601621575288)
        generator = np.random.default_rng(158)
        values = generator.normal(size=128).astype(np.float32)
        codes = generator.integers(-7, 8, size=(23, 16), dtype=np.int8)
        scales = generator.uniform(1e-4, 0.1, size=23).astype(np.float16)
        coefficients = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
        sketch = native_hash_projection_f32(
            values,
            projection.buckets.numpy(),
            projection.signs.numpy(),
            sketch_dim=16,
            library=self.native,
        )
        observed = native_hash_int4_gemv_f32(
            pack_signed_int4_rows(codes),
            scales,
            sketch,
            sketch_dim=16,
            library=self.native,
            implementation="scalar",
        )
        expanded = expand_sketch_rows(torch.from_numpy(coefficients), projection)
        expected = torch.from_numpy(values) @ expanded.T
        np.testing.assert_allclose(observed, expected.numpy(), rtol=2e-5, atol=2e-5)
        fused = native_hash_residual_f32(
            values,
            projection.buckets.numpy(),
            projection.signs.numpy(),
            pack_signed_int4_rows(codes),
            scales,
            sketch_dim=16,
            library=self.native,
            implementation="scalar",
        )
        np.testing.assert_allclose(fused, expected.numpy(), rtol=2e-5, atol=2e-5)
        base = generator.normal(size=23).astype(np.float32)
        added = native_hash_residual_f32(
            values,
            projection.buckets.numpy(),
            projection.signs.numpy(),
            pack_signed_int4_rows(codes),
            scales,
            sketch_dim=16,
            library=self.native,
            implementation="scalar",
            base_output=base,
        )
        np.testing.assert_allclose(added, base + expected.numpy(), rtol=2e-5, atol=2e-5)

    def test_safe_native_entry_rejects_noncanonical_nibbles(self) -> None:
        packed = np.array([[0x08]], dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_hash_int4_gemv_f32(
                packed,
                np.array([1.0], dtype=np.float16),
                np.array([1.0], dtype=np.float32),
                sketch_dim=1,
                library=self.native,
                implementation="scalar",
            )
        packed = np.array([[0x10]], dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_hash_int4_gemv_f32(
                packed,
                np.array([1.0], dtype=np.float16),
                np.array([1.0], dtype=np.float32),
                sketch_dim=1,
                library=self.native,
                implementation="scalar",
            )

    def test_avx2_matches_scalar_when_available(self) -> None:
        features = self.native.features()
        if not (features["avx2"] and features["fma"]):
            self.skipTest("AVX2/FMA is unavailable")
        generator = np.random.default_rng(585)
        codes = generator.integers(-7, 8, size=(67, 80), dtype=np.int8)
        scales = generator.uniform(1e-4, 0.2, size=67).astype(np.float16)
        sketch = generator.normal(size=80).astype(np.float32)
        packed = pack_signed_int4_rows(codes)
        scalar = native_hash_int4_gemv_f32(
            packed, scales, sketch, sketch_dim=80, library=self.native,
            implementation="scalar", trusted=True,
        )
        avx2 = native_hash_int4_gemv_f32(
            packed, scales, sketch, sketch_dim=80, library=self.native,
            implementation="avx2", trusted=True,
        )
        np.testing.assert_allclose(avx2, scalar, rtol=2e-6, atol=2e-6)
        projection = sparse_jl_projection(128, 80, hashes_per_input=4, seed=585)
        values = generator.normal(size=128).astype(np.float32)
        base = generator.normal(size=67).astype(np.float32)
        scalar_fused = native_hash_residual_f32(
            values, projection.buckets.numpy(), projection.signs.numpy(), packed,
            scales, sketch_dim=80, library=self.native, implementation="scalar",
            trusted=True, base_output=base,
        )
        avx2_fused = native_hash_residual_f32(
            values, projection.buckets.numpy(), projection.signs.numpy(), packed,
            scales, sketch_dim=80, library=self.native, implementation="avx2",
            trusted=True, base_output=base,
        )
        np.testing.assert_allclose(avx2_fused, scalar_fused, rtol=2e-6, atol=2e-6)

    def test_projection_normalization_is_exact_clark_hash_scale(self) -> None:
        projection = sparse_jl_projection(32, 7, hashes_per_input=4, seed=1)
        values = np.ones(32, dtype=np.float32)
        observed = native_hash_projection_f32(
            values,
            projection.buckets.numpy(),
            projection.signs.numpy(),
            sketch_dim=7,
            library=self.native,
        )
        raw = np.zeros(7, dtype=np.float32)
        for repetition in range(4):
            np.add.at(raw, projection.buckets[repetition].numpy(), projection.signs[repetition].numpy())
        np.testing.assert_array_equal(observed, raw / np.float32(math.sqrt(4)))


if __name__ == "__main__":
    unittest.main()
