from __future__ import annotations

import unittest

import numpy as np

from clark_quantize import ternary_cpu
from clark_quantize.base3_block203 import (
    matrix_payload_bits,
    matrix_payload_bytes,
    pack_row,
    pack_rows,
    row_payload_bits,
    row_payload_bytes,
)
from clark_quantize.qwen38_row_scale_codebook import (
    reconstruct_row_normalized_scales,
)
from clark_quantize.scale_codebook import pack_nibbles


class TernaryCpuPackingTests(unittest.TestCase):
    def test_block203_is_byte_identical_to_canonical_python(self) -> None:
        generator = np.random.default_rng(20260817)
        counts = (*range(129), 129, 255, 256, 4097, 5120)
        for count in counts:
            cases = (
                np.full(count, -1, dtype=np.int8),
                np.zeros(count, dtype=np.int8),
                np.ones(count, dtype=np.int8),
                generator.integers(-1, 2, size=count, dtype=np.int8),
            )
            for codes in cases:
                actual = ternary_cpu.pack_block203(codes)
                self.assertEqual(
                    ternary_cpu.block203_size(count),
                    (row_payload_bits(count), row_payload_bytes(count)),
                )
                self.assertEqual(actual.tobytes(), pack_row(codes))
                np.testing.assert_array_equal(
                    ternary_cpu.unpack_block203(actual, count), codes,
                )

    def test_block203_rejects_invalid_and_noncanonical_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "ternary|code"):
            ternary_cpu.pack_block203(np.asarray([-2], dtype=np.int8))
        with self.assertRaisesRegex(ValueError, "ternary|code"):
            ternary_cpu.pack_block203(np.asarray([0.5], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "ternary|code"):
            ternary_cpu.pack_block203(np.asarray([255], dtype=np.uint8))
        noncanonical = np.frombuffer(
            (3**128).to_bytes(row_payload_bytes(128), "little"),
            dtype=np.uint8,
        ).copy()
        with self.assertRaisesRegex(ValueError, "canonical|code"):
            ternary_cpu.unpack_block203(noncanonical, 128)
        padding = ternary_cpu.pack_block203(np.asarray([0], dtype=np.int8))
        padding[-1] |= np.uint8(0x80)
        with self.assertRaisesRegex(ValueError, "canonical|code|padding"):
            ternary_cpu.unpack_block203(padding, 1)

    def test_block203_matrix_is_byte_identical_with_per_row_group_resets(self) -> None:
        generator = np.random.default_rng(20260817)
        for rows, columns in ((3, 1), (5, 2), (3, 5), (2, 127), (3, 128),
                              (4, 129), (2, 255), (3, 256), (2, 4097)):
            codes = generator.integers(-1, 2, size=(rows, columns), dtype=np.int8)
            packed = ternary_cpu.pack_block203_rows(codes)
            self.assertEqual(packed.tobytes(), pack_rows(codes))
            self.assertEqual(
                ternary_cpu.block203_matrix_size(rows, columns),
                (matrix_payload_bits(rows, columns), matrix_payload_bytes(rows, columns)),
            )
            np.testing.assert_array_equal(
                ternary_cpu.unpack_block203_rows(packed, rows, columns), codes,
            )
        self.assertEqual(ternary_cpu.pack_block203_rows(
            np.zeros((3, 1), dtype=np.int8)
        ).size, 1)

    def test_block203_matrix_rejects_padding_and_bad_shapes(self) -> None:
        packed = ternary_cpu.pack_block203_rows(np.zeros((3, 1), dtype=np.int8))
        packed[0] |= np.uint8(0x80)
        with self.assertRaisesRegex(ValueError, "canonical|code|padding"):
            ternary_cpu.unpack_block203_rows(packed, 3, 1)
        with self.assertRaises(ValueError):
            ternary_cpu.pack_block203_rows(np.zeros(3, dtype=np.int8))
        noncanonical_value = (3**128) << row_payload_bits(128)
        noncanonical = np.frombuffer(
            noncanonical_value.to_bytes(matrix_payload_bytes(2, 128), "little"),
            dtype=np.uint8,
        ).copy()
        with self.assertRaisesRegex(ValueError, "canonical|code"):
            ternary_cpu.unpack_block203_rows(noncanonical, 2, 128)

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

    def test_radix5_plane_activation_layout(self) -> None:
        activation = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        expanded = ternary_cpu.expand_q8_radix5_planes(activation).reshape(2, 5, 32)
        for group in range(2):
            for position in range(128):
                self.assertEqual(
                    int(expanded[group, position % 5, position // 5]),
                    int(activation[group * 128 + position]),
                )
            self.assertTrue(np.all(expanded[group, :, 26:] == 0))


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

    def test_native_swiglu_contract(self) -> None:
        gate = np.asarray([-4.0, -0.5, 0.0, 0.5, 4.0], dtype=np.float32)
        up = np.asarray([2.0, -3.0, 1.0, 0.25, -0.5], dtype=np.float32)
        expected = (gate / (np.float32(1.0) + np.exp(-gate))) * up
        np.testing.assert_allclose(
            ternary_cpu.swiglu_f32(gate, up), expected,
            rtol=1e-6, atol=1e-7,
        )
        gate[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ternary_cpu.swiglu_f32(gate, up)

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

    def test_row_scale4_native_decode_matches_frozen_fp32_contract(self) -> None:
        generator = np.random.default_rng(20260817)
        rows, groups = 19, 37
        anchors = generator.lognormal(-2.0, 0.8, size=rows).astype(np.float16)
        codebook = np.concatenate(
            (
                np.zeros(1, dtype=np.float16),
                np.geomspace(0.4, 2.5, 15, dtype=np.float32).astype(np.float16),
            )
        )
        codes = generator.integers(0, 16, size=(rows, groups), dtype=np.uint8)
        packed = np.frombuffer(pack_nibbles(codes), dtype=np.uint8).copy()
        expected = reconstruct_row_normalized_scales(anchors, codebook, codes)
        scalar = ternary_cpu.decode_row_scale4(
            packed, anchors, codebook, groups, implementation="scalar"
        )
        np.testing.assert_array_equal(scalar, expected)
        if ternary_cpu.cpu_features()["avx2"]:
            avx2 = ternary_cpu.decode_row_scale4(
                packed, anchors, codebook, groups, implementation="avx2"
            )
            np.testing.assert_array_equal(avx2, scalar)

    def test_row_scale4_gemv_matches_fp32_product_oracle(self) -> None:
        codes, _, quantized, activation_scales, sums, packed_weights = self.fixture(
            rows=19, columns=512
        )
        generator = np.random.default_rng(88)
        anchors = generator.uniform(0.02, 0.2, size=codes.shape[0]).astype(np.float16)
        codebook = np.concatenate(
            (
                np.zeros(1, dtype=np.float16),
                np.linspace(0.5, 1.5, 15, dtype=np.float32).astype(np.float16),
            )
        )
        scale_codes = generator.integers(
            0, 16, size=(codes.shape[0], codes.shape[1] // 128), dtype=np.uint8
        )
        packed_scale_codes = np.frombuffer(
            pack_nibbles(scale_codes), dtype=np.uint8
        ).copy()
        materialized = reconstruct_row_normalized_scales(
            anchors, codebook, scale_codes
        )
        row_scalar = ternary_cpu.gemv_q8_group128_2bit_row_scale4(
            packed_weights,
            packed_scale_codes,
            anchors,
            codebook,
            quantized,
            activation_scales,
            sums,
            implementation="scalar",
        )
        oracle = np.empty(codes.shape[0], dtype=np.float32)
        for row in range(codes.shape[0]):
            total = np.float32(0.0)
            for group in range(codes.shape[1] // 128):
                start = group * 128
                dot = int(np.dot(
                    codes[row, start:start + 128].astype(np.int32),
                    quantized[start:start + 128].astype(np.int32),
                ))
                contribution = np.float32(dot) * materialized[row, group]
                contribution = contribution * activation_scales[group]
                total = np.float32(total + contribution)
            oracle[row] = total
        np.testing.assert_allclose(row_scalar, oracle, rtol=1e-6, atol=1e-6)
        block203_weights = ternary_cpu.pack_block203_rows(codes)
        block203_scalar = ternary_cpu.gemv_q8_group128_block203_row_scale4(
            block203_weights,
            packed_scale_codes,
            anchors,
            codebook,
            quantized,
            activation_scales,
            sums,
            implementation="scalar",
        )
        np.testing.assert_allclose(block203_scalar, row_scalar, rtol=1e-6, atol=1e-6)
        block203_constdiv_scalar = ternary_cpu.gemv_q8_group128_block203_row_scale4(
            block203_weights,
            packed_scale_codes,
            anchors,
            codebook,
            quantized,
            activation_scales,
            sums,
            implementation="scalar_constdiv",
        )
        np.testing.assert_array_equal(block203_constdiv_scalar, block203_scalar)
        block203_chunk16_scalar = ternary_cpu.gemv_q8_group128_block203_row_scale4(
            block203_weights,
            packed_scale_codes,
            anchors,
            codebook,
            quantized,
            activation_scales,
            sums,
            implementation="scalar_chunk16",
        )
        np.testing.assert_array_equal(block203_chunk16_scalar, block203_scalar)
        noncanonical_block203 = block203_weights.copy()
        payload_bits, _ = ternary_cpu.block203_matrix_size(*codes.shape)
        unused_bits = noncanonical_block203.nbytes * 8 - payload_bits
        self.assertGreater(unused_bits, 0)
        noncanonical_block203[-1] |= np.uint8(1 << (8 - unused_bits))
        with self.assertRaisesRegex(ValueError, "code"):
            ternary_cpu.gemv_q8_group128_block203_row_scale4(
                noncanonical_block203,
                packed_scale_codes,
                anchors,
                codebook,
                quantized,
                activation_scales,
                sums,
                implementation="scalar",
                validated=True,
            )
        if ternary_cpu.cpu_features()["avx2"]:
            row_avx2 = ternary_cpu.gemv_q8_group128_2bit_row_scale4(
                packed_weights,
                packed_scale_codes,
                anchors,
                codebook,
                quantized,
                activation_scales,
                sums,
                implementation="avx2",
            )
            np.testing.assert_allclose(row_avx2, row_scalar, rtol=1e-5, atol=5e-5)
            block203_avx2 = ternary_cpu.gemv_q8_group128_block203_row_scale4(
                block203_weights,
                packed_scale_codes,
                anchors,
                codebook,
                quantized,
                activation_scales,
                sums,
                implementation="avx2",
            )
            np.testing.assert_allclose(
                block203_avx2, block203_scalar, rtol=1e-5, atol=5e-5
            )
            block203_constdiv_avx2 = (
                ternary_cpu.gemv_q8_group128_block203_row_scale4(
                    block203_weights,
                    packed_scale_codes,
                    anchors,
                    codebook,
                    quantized,
                    activation_scales,
                    sums,
                    implementation="avx2_constdiv",
                )
            )
            np.testing.assert_array_equal(block203_constdiv_avx2, block203_avx2)
            block203_chunk16_avx2 = (
                ternary_cpu.gemv_q8_group128_block203_row_scale4(
                    block203_weights,
                    packed_scale_codes,
                    anchors,
                    codebook,
                    quantized,
                    activation_scales,
                    sums,
                    implementation="avx2_chunk16",
                )
            )
            np.testing.assert_array_equal(block203_chunk16_avx2, block203_avx2)

    def test_row_scale4_rejects_noncanonical_payloads(self) -> None:
        anchors = np.ones(1, dtype=np.float16)
        codebook = np.ones(16, dtype=np.float16)
        codebook[0] = 0
        packed = np.frombuffer(pack_nibbles(np.zeros(3, dtype=np.uint8)), dtype=np.uint8).copy()
        packed[-1] |= np.uint8(0x10)
        with self.assertRaisesRegex(ValueError, "canonical|code"):
            ternary_cpu.decode_row_scale4(packed, anchors, codebook, 3)
        packed[-1] &= np.uint8(0x0F)
        codebook[0] = np.float16(1.0)
        with self.assertRaisesRegex(ValueError, "canonical"):
            ternary_cpu.decode_row_scale4(packed, anchors, codebook, 3)

    def test_block203_row_scale4_rejects_out_of_range_canonical_bits(self) -> None:
        _, weight_bytes = ternary_cpu.block203_matrix_size(1, 128)
        weights = np.full(weight_bytes, 0xFF, dtype=np.uint8)
        weights[-1] = np.uint8(0x07)
        indices = np.zeros(1, dtype=np.uint8)
        anchors = np.ones(1, dtype=np.float16)
        codebook = np.zeros(16, dtype=np.float16)
        activation = np.zeros(128, dtype=np.int8)
        scales = np.ones(1, dtype=np.float32)
        sums = np.zeros(1, dtype=np.int32)
        for implementation in ("scalar", "scalar_constdiv", "scalar_chunk16"):
            with self.subTest(implementation=implementation):
                with self.assertRaisesRegex(ValueError, "code"):
                    ternary_cpu.gemv_q8_group128_block203_row_scale4(
                        weights,
                        indices,
                        anchors,
                        codebook,
                        activation,
                        scales,
                        sums,
                        implementation=implementation,
                    )

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
        if ternary_cpu.cpu_features()["neon"]:
            self.assertEqual(
                ternary_cpu.memory_read_sum(values, implementation="neon", threads=4),
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
        expanded_planes = ternary_cpu.expand_q8_radix5_planes(quantized)
        radix5_planes = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded_planes, activation_scales, sums,
            implementation="avx2_planes",
        )
        radix5_planes_threaded = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded_planes, activation_scales, sums,
            implementation="avx2_planes", threads=4,
        )
        radix5_pair = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather_pair",
        )
        radix5_pair_threaded = ternary_cpu.gemv_q8_group128_radix5(
            packed_radix5, scales, expanded, activation_scales, sums,
            implementation="avx2_nogather_pair", threads=4,
        )
        np.testing.assert_allclose(radix5, scalar, rtol=1e-5, atol=5e-5)
        np.testing.assert_array_equal(radix5_threaded, radix5)
        np.testing.assert_array_equal(radix5_nogather, radix5)
        np.testing.assert_array_equal(radix5_nogather_threaded, radix5_nogather)
        np.testing.assert_array_equal(radix5_static, radix5)
        np.testing.assert_array_equal(radix5_static_threaded, radix5_static)
        np.testing.assert_allclose(radix5_planes, radix5, rtol=1e-5, atol=5e-5)
        np.testing.assert_array_equal(radix5_planes_threaded, radix5_planes)
        np.testing.assert_array_equal(radix5_pair, radix5)
        np.testing.assert_array_equal(radix5_pair_threaded, radix5_pair)

    @unittest.skipUnless(
        ternary_cpu.cpu_features()["dotprod"], "ARM dot-product is unavailable"
    )
    def test_arm_dotprod_matches_scalar_and_is_thread_deterministic(self) -> None:
        codes, scales, quantized, activation_scales, sums, packed = self.fixture(
            rows=31, columns=512
        )
        scalar = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="scalar",
        )
        arm = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="arm_dotprod",
        )
        threaded = ternary_cpu.gemv_q8_group128_2bit(
            packed, scales, quantized, activation_scales, sums,
            implementation="arm_dotprod", threads=4,
        )
        np.testing.assert_allclose(arm, scalar, rtol=1e-5, atol=5e-5)
        np.testing.assert_array_equal(threaded, arm)

    @unittest.skipUnless(
        ternary_cpu.cpu_features()["dotprod"], "ARM dot-product is unavailable"
    )
    def test_arm_fused_swiglu_matches_two_projection_contract(self) -> None:
        _, gate_scales, quantized, activation_scales, sums, gate_packed = (
            self.fixture(rows=32, columns=512)
        )
        _, up_scales, _, _, _, up_packed = self.fixture(rows=32, columns=512)
        gate = ternary_cpu.gemv_q8_group128_2bit(
            gate_packed, gate_scales, quantized, activation_scales, sums,
            implementation="arm_dotprod",
        )
        up = ternary_cpu.gemv_q8_group128_2bit(
            up_packed, up_scales, quantized, activation_scales, sums,
            implementation="arm_dotprod",
        )
        expected = ternary_cpu.swiglu_f32(gate, up)
        fused = ternary_cpu.gemv_q8_group128_2bit_swiglu(
            gate_packed, gate_scales, up_packed, up_scales,
            quantized, activation_scales, sums,
            implementation="arm_dotprod",
        )
        threaded = ternary_cpu.gemv_q8_group128_2bit_swiglu(
            gate_packed, gate_scales, up_packed, up_scales,
            quantized, activation_scales, sums,
            implementation="arm_dotprod", threads=4,
        )
        np.testing.assert_array_equal(fused, expected)
        np.testing.assert_array_equal(threaded, fused)


if __name__ == "__main__":
    unittest.main()
