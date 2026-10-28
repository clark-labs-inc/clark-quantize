from __future__ import annotations

from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from clark_quantize.qwen38_calibration_span_residual import (
    Int8ResidualAdapter,
    load_int8_adapter,
)
from clark_quantize.qwen38_calibration_residual_cpu import (
    CalibrationResidualCPULibrary,
    load_calibration_residual_cpu_library,
    native_calibration_residual_f32,
)


def synthetic_adapter(seed: int, input_dim: int = 40, output_dim: int = 32) -> Int8ResidualAdapter:
    generator = torch.Generator().manual_seed(seed)
    return Int8ResidualAdapter(
        left_codes=torch.randint(-127, 128, (input_dim, 16), generator=generator, dtype=torch.int8),
        left_scales=torch.rand(16, generator=generator, dtype=torch.float32).mul(0.002).to(torch.float16),
        right_codes=torch.randint(-127, 128, (16, output_dim), generator=generator, dtype=torch.int8),
        right_scales=torch.rand(16, generator=generator, dtype=torch.float32).mul(0.002).to(torch.float16),
        bias=torch.randn(output_dim, generator=generator, dtype=torch.float32).mul(0.01).to(torch.float16),
        ridge_fraction=0.01,
    )


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38CalibrationResidualCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        source = repo / "clark_quantize/qwen38_calibration_residual_cpu.c"
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"calibration-residual{extension}"
        shared = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared, str(source), "-lm", "-o", str(output),
            ],
            check=True,
        )
        library = load_calibration_residual_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh calibration-residual library did not load")
        cls.native: CalibrationResidualCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_scalar_matches_unmaterialized_trained_algebra(self) -> None:
        adapter = synthetic_adapter(37)
        generator = np.random.default_rng(38)
        values = generator.normal(size=40).astype(np.float32)
        base = generator.normal(size=32).astype(np.float32)
        observed = native_calibration_residual_f32(
            values, adapter, library=self.native, implementation="scalar"
        )
        expected = adapter.correction(torch.from_numpy(values)[None, :])[0].numpy()
        np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-5)
        added = native_calibration_residual_f32(
            values, adapter, library=self.native, implementation="scalar", base_output=base
        )
        np.testing.assert_allclose(added, base + expected, rtol=3e-5, atol=3e-5)

    def test_scalar_executes_exact_v37_layer0_payload(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        payload = repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
            "qwen38_27b_sensitivity_gated_residual_v37/artifacts/layer-0.csr"
        )
        adapter = load_int8_adapter(
            payload, input_features=5120, output_features=5120, rank=16,
            ridge_fraction=0.01,
        )
        values = np.random.default_rng(158).normal(size=5120).astype(np.float32)
        observed = native_calibration_residual_f32(
            values, adapter, library=self.native, implementation="scalar"
        )
        expected = adapter.correction(torch.from_numpy(values)[None, :])[0].numpy()
        np.testing.assert_allclose(observed, expected, rtol=6e-5, atol=5e-4)

    def test_safe_entry_rejects_nonfinite_input(self) -> None:
        adapter = synthetic_adapter(39)
        values = np.zeros(40, dtype=np.float32)
        values[3] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            native_calibration_residual_f32(
                values, adapter, library=self.native, implementation="scalar"
            )

    def test_avx2_matches_scalar_when_available(self) -> None:
        if not all(self.native.features().values()):
            self.skipTest("AVX2/FMA/F16C is unavailable")
        adapter = synthetic_adapter(40, input_dim=5120, output_dim=5120)
        generator = np.random.default_rng(41)
        values = generator.normal(size=5120).astype(np.float32)
        base = generator.normal(size=5120).astype(np.float32)
        scalar = native_calibration_residual_f32(
            values, adapter, library=self.native, implementation="scalar", base_output=base,
        )
        avx2 = native_calibration_residual_f32(
            values, adapter, library=self.native, implementation="avx2", base_output=base,
        )
        np.testing.assert_allclose(avx2, scalar, rtol=6e-5, atol=5e-4)

    def test_source_has_no_high_precision_factor_materialization(self) -> None:
        source = (Path(__file__).with_name("qwen38_calibration_residual_cpu.c")).read_text()
        self.assertNotIn("malloc(", source)
        self.assertNotIn("calloc(", source)
        self.assertIn("float *rank_scratch", source)
        self.assertIn("_mm256_cvtepi8_epi32", source)
        self.assertIn("_mm256_cvtph_ps", source)


if __name__ == "__main__":
    unittest.main()
