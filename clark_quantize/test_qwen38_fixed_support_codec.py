from __future__ import annotations

from itertools import combinations, product
import unittest

import numpy as np

from clark_quantize.qwen38_fixed_support_codec import (
    FixedSupportPayload,
    decode_fixed_support_group,
    encode_fixed_support_group,
    exact_hash_residual_component_bits_per_weight,
    fixed_support_state_count,
    fixed_support_width_bits,
    pack_fixed_support_matrix,
    rank_support,
    unrank_support,
    unpack_fixed_support_matrix,
)


class Qwen38FixedSupportCodecTests(unittest.TestCase):
    def test_small_state_space_is_bijective_and_dense(self) -> None:
        group_size = 5
        support_size = 2
        observed = set()
        for positions in combinations(range(group_size), support_size):
            rank = rank_support(positions, group_size)
            self.assertEqual(unrank_support(rank, group_size, support_size), positions)
            for signs in product((-1, 1), repeat=support_size):
                codes = np.zeros(group_size, dtype=np.int8)
                codes[list(positions)] = signs
                encoded = encode_fixed_support_group(codes, support_size)
                observed.add(encoded)
                np.testing.assert_array_equal(
                    decode_fixed_support_group(encoded, group_size, support_size), codes
                )
        self.assertEqual(observed, set(range(fixed_support_state_count(5, 2))))

    def test_qwen_candidate_widths_are_exact(self) -> None:
        self.assertEqual(fixed_support_width_bits(128, 64), 189)
        self.assertEqual(fixed_support_width_bits(128, 60), 184)
        self.assertEqual(fixed_support_width_bits(128, 54), 176)

    def test_random_qwen_groups_round_trip(self) -> None:
        generator = np.random.default_rng(158)
        for support_size in (54, 60, 64):
            for _ in range(20):
                positions = generator.choice(128, size=support_size, replace=False)
                codes = np.zeros(128, dtype=np.int8)
                codes[positions] = generator.choice(
                    np.array([-1, 1], dtype=np.int8), size=support_size
                )
                encoded = encode_fixed_support_group(codes, support_size)
                np.testing.assert_array_equal(
                    decode_fixed_support_group(encoded, 128, support_size), codes
                )

    def test_bit_contiguous_matrix_round_trip_has_exact_length(self) -> None:
        generator = np.random.default_rng(27)
        codes = np.zeros((7, 256), dtype=np.int8)
        for group in codes.reshape(-1, 128):
            positions = generator.choice(128, size=60, replace=False)
            group[positions] = generator.choice(
                np.array([-1, 1], dtype=np.int8), size=60
            )
        payload = pack_fixed_support_matrix(codes, nonzero_per_group=60)
        self.assertEqual(payload.bit_length, 7 * 2 * 184)
        self.assertEqual(len(payload.payload), (payload.bit_length + 7) // 8)
        np.testing.assert_array_equal(unpack_fixed_support_matrix(payload), codes)

    def test_noncanonical_groups_and_padding_are_rejected(self) -> None:
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :59] = 1
        with self.assertRaisesRegex(ValueError, "wrong nonzero"):
            pack_fixed_support_matrix(codes, nonzero_per_group=60)
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            decode_fixed_support_group(fixed_support_state_count(128, 60), 128, 60)
        with self.assertRaisesRegex(ValueError, "lie in"):
            pack_fixed_support_matrix(
                np.full((1, 128), 255, dtype=np.uint8), nonzero_per_group=64
            )
        valid = np.zeros((1, 128), dtype=np.int8)
        valid[0, :64] = 1
        payload = pack_fixed_support_matrix(valid, nonzero_per_group=64)
        corrupted = FixedSupportPayload(
            **{
                **payload.__dict__,
                "payload": payload.payload[:-1] + bytes([payload.payload[-1] | 0x80]),
            }
        )
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_fixed_support_matrix(corrupted)

    def test_exact_component_rate_creates_hash_correction_headroom(self) -> None:
        cases = [
            (5120, 40, 64, 1.55),
            (5120, 80, 60, 1.55),
            (5120, 160, 54, 1.55),
            (17408, 136, 64, 1.55),
            (17408, 272, 60, 1.55),
            (17408, 544, 54, 1.55),
        ]
        for input_dim, sketch_dim, support, maximum in cases:
            rate = exact_hash_residual_component_bits_per_weight(
                input_dim=input_dim,
                sketch_dim=sketch_dim,
                nonzero_per_group=support,
            )
            self.assertLess(rate, maximum)
            self.assertGreater(rate, 1.50)


if __name__ == "__main__":
    unittest.main()
