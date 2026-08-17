from __future__ import annotations

import unittest

import numpy as np

from clark_quantize import ternary_cpu


class TernaryCpuPackingTests(unittest.TestCase):
    def test_aligned2_roundtrip_including_partial_bytes(self) -> None:
        generator = np.random.default_rng(20260817)
        for count in range(33):
            codes = generator.integers(-1, 2, size=count, dtype=np.int8)
            packed = ternary_cpu.pack_aligned2(codes)
            np.testing.assert_array_equal(ternary_cpu.unpack_aligned2(packed, count), codes)
            self.assertEqual(packed.size, (count + 3) // 4)

    def test_aligned2_rejects_invalid_codes_and_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, "ternary|code"):
            ternary_cpu.pack_aligned2(np.asarray([-1, 2], dtype=np.int8))
        packed = ternary_cpu.pack_aligned2(np.asarray([0], dtype=np.int8))
        packed[0] = (packed[0] & 0x3f) | 0xc0
        with self.assertRaisesRegex(ValueError, "canonical|code"):
            ternary_cpu.unpack_aligned2(packed, 1)

    def test_group128_compute_image_roundtrip(self) -> None:
        generator = np.random.default_rng(20260817)
        codes = generator.integers(-1, 2, size=128 * 3, dtype=np.int8)
        packed = ternary_cpu.pack_group128_2bit(codes)
        self.assertEqual(packed.size, codes.size // 4)
        np.testing.assert_array_equal(
            ternary_cpu.unpack_group128_2bit(packed, codes.size),
            codes,
        )

    def test_group128_radix5_roundtrip_and_canonical_padding(self) -> None:
        generator = np.random.default_rng(20260817)
        codes = generator.integers(-1, 2, size=128 * 3, dtype=np.int8)
        packed = ternary_cpu.pack_group128_radix5(codes)
        self.assertEqual(packed.size, 26 * 3)
        np.testing.assert_array_equal(
            ternary_cpu.unpack_group128_radix5(packed, codes.size),
            codes,
        )
        bad = packed.copy()
        bad[25] = np.uint8(int(bad[25]) + 27)
        with self.assertRaisesRegex(ValueError, "canonical|code"):
            ternary_cpu.unpack_group128_radix5(bad, codes.size)


@unittest.skipUnless(ternary_cpu.native_available(), "Clark ternary CPU library is not built")
class TernaryCpuNativeTests(unittest.TestCase):
    def fixture(self, rows: int = 17, columns: int = 256):
        generator = np.random.default_rng(20260817)
        codes = generator.integers(-1, 2, size=(rows, columns), dtype=np.int8)
        scales = generator.uniform(0.001, 0.25, size=(rows, columns // 128)).astype(np.float16)
        activation = generator.normal(size=columns).astype(np.float32)
        quantized, activation_scales, sums = ternary_cpu.quantize_q8(activation)
        packed = np.stack([ternary_cpu.pack_group128_2bit(row) for row in codes])
        return codes, scales, quantized, activation_scales, sums, packed

    def test_q8_quantization_contract(self) -> None:
        activation = np.concatenate([
            np.zeros(128, dtype=np.float32),
            np.linspace(-2.0, 2.0, 128, dtype=np.float32),
        ])
        quantized, scales, sums = ternary_cpu.quantize_q8(activation)
        self.assertTrue(np.all(quantized[:128] == 0))
        self.assertEqual(scales[0], np.float32(1.0))
        self.assertEqual(sums[0], 0)
        self.assertEqual(int(quantized[128]), -127)
        self.assertEqual(int(quantized[-1]), 127)
        self.assertEqual(int(sums[1]), int(quantized[128:].astype(np.int32).sum()))
        activation[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ternary_cpu.quantize_q8(activation)

    def test_scalar_matches_groupwise_integer_oracle(self) -> None:
        codes, scales, quantized, activation_scales, sums, packed = self.fixture()
        actual = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="scalar",
        )
        expected = np.empty(codes.shape[0], dtype=np.float32)
        for row in range(codes.shape[0]):
            total = np.float32(0.0)
            for group in range(codes.shape[1] // 128):
                start = group * 128
                dot = int(np.dot(
                    codes[row, start:start + 128].astype(np.int32),
                    quantized[start:start + 128].astype(np.int32),
                ))
                contribution = np.float32(dot) * np.float32(scales[row, group])
                contribution = contribution * activation_scales[group]
                total = np.float32(total + contribution)
            expected[row] = total
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_radix5_scalar_matches_aligned_scalar(self) -> None:
        codes, scales, quantized, activation_scales, sums, packed = self.fixture()
        aligned = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="scalar",
        )
        packed_radix5 = np.stack([
            ternary_cpu.pack_group128_radix5(row) for row in codes
        ])
        expanded = ternary_cpu.expand_q8_radix5(quantized)
        radix5 = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="scalar",
        )
        np.testing.assert_allclose(radix5, aligned, rtol=1e-6, atol=1e-6)

    def test_stream_control_reads_every_byte(self) -> None:
        values = np.arange(1_000_003, dtype=np.uint8)
        expected = int(values.astype(np.uint64).sum())
        self.assertEqual(
            ternary_cpu.memory_read_sum(values, implementation="scalar"),
            expected,
        )
        if ternary_cpu.cpu_features()["avx2"]:
            self.assertEqual(
                ternary_cpu.memory_read_sum(values, implementation="avx2", threads=4),
                expected,
            )

    def test_rejects_group_scratch_larger_than_qwen_contract(self) -> None:
        groups = 257
        columns = groups * 128
        with self.assertRaisesRegex(ValueError, "shape"):
            ternary_cpu.gemv_q8_group128_2bit(
                np.zeros((1, columns // 4), dtype=np.uint8),
                np.ones((1, groups), dtype=np.float16),
                np.zeros(columns, dtype=np.int8),
                np.ones(groups, dtype=np.float32),
                np.zeros(groups, dtype=np.int32),
                implementation="scalar",
            )

    @unittest.skipUnless(ternary_cpu.cpu_features()["avx2"], "AVX2 is unavailable")
    def test_avx2_matches_scalar_and_is_thread_deterministic(self) -> None:
        codes, scales, quantized, activation_scales, sums, packed = self.fixture(rows=31, columns=512)
        scalar = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="scalar",
        )
        avx2 = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="avx2",
        )
        threaded = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="avx2", threads=4,
        )
        np.testing.assert_allclose(avx2, scalar, rtol=1e-5, atol=5e-5)
        np.testing.assert_array_equal(threaded, avx2)

        packed_radix5 = np.stack([
            ternary_cpu.pack_group128_radix5(row) for row in codes
        ])
        expanded = ternary_cpu.expand_q8_radix5(quantized)
        radix5 = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2",
        )
        radix5_threaded = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2", threads=4,
        )
        radix5_nogather = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather",
        )
        radix5_nogather_threaded = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather", threads=4,
        )
        radix5_static = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather_static",
        )
        radix5_static_threaded = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather_static", threads=4,
        )
        np.testing.assert_allclose(radix5, scalar, rtol=1e-5, atol=5e-5)
        np.testing.assert_array_equal(radix5_threaded, radix5)
        np.testing.assert_array_equal(radix5_nogather, radix5)
        np.testing.assert_array_equal(radix5_nogather_threaded, radix5_nogather)
        np.testing.assert_array_equal(radix5_static, radix5)
        np.testing.assert_array_equal(radix5_static_threaded, radix5_static)


if __name__ == "__main__":
    unittest.main()
