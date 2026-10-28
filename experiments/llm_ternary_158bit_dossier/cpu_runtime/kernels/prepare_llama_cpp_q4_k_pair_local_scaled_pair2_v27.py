#!/usr/bin/env python3
"""Prepare the v27 exact Q4_K pair-local-scaled two-row AVX2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import prepare_llama_cpp_q4_k_streamed_decode_pair2_v26 as streamed_v26


FROZEN_PARENT_SHA256 = streamed_v26.FROZEN_PARENT_SHA256
OLD_HELPER = streamed_v26.NEW_HELPER
NEW_HELPER = "ggml_gemv_q4_K_8x8_q8_K_pair_local_scaled_pair2_tile"


PAIR_LOCAL_SCALED_TEMPLATE = r"""template <int RAW_OFFSET, int LHS_LOW_OFFSET, int LHS_HIGH_OFFSET, int SHUFFLE_0, int SHUFFLE_1>
static inline __attribute__((always_inline)) void ggml_q4_K_accumulate_scaled_raw_pair_pair2(
        const uint8_t * GGML_RESTRICT rhs_base,
        const int8_t * GGML_RESTRICT lhs_0_base,
        const int8_t * GGML_RESTRICT lhs_1_base,
        const __m256i scales_0,
        const __m256i scales_1,
        __m256i & scaled_0,
        __m256i & scaled_1) {
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i rhs_raw_0123 =
        _mm256_loadu_si256((const __m256i *) (rhs_base + RAW_OFFSET));
    const __m256i rhs_raw_4567 =
        _mm256_loadu_si256((const __m256i *) (rhs_base + RAW_OFFSET + 32));

    {
        const __m256i rhs_0123 = _mm256_and_si256(rhs_raw_0123, m4b);
        const __m256i rhs_4567 = _mm256_and_si256(rhs_raw_4567, m4b);
        __m256i lhs = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_0_base + LHS_LOW_OFFSET)));
        lhs = _mm256_permute2f128_si256(lhs, lhs, 0);
        const __m256i low = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_0)),
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_1)));
        __m256i contribution = _mm256_madd_epi16(low, scales_0);

        const __m256i rhs_high_0123 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0123, 4), m4b);
        const __m256i rhs_high_4567 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_4567, 4), m4b);
        lhs = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_0_base + LHS_HIGH_OFFSET)));
        lhs = _mm256_permute2f128_si256(lhs, lhs, 0);
        const __m256i high = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(rhs_high_0123, _mm256_shuffle_epi32(rhs_high_4567, 177), 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_0)),
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_high_0123, 177), rhs_high_4567, 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_1)));
        contribution = _mm256_add_epi32(contribution, _mm256_madd_epi16(high, scales_1));
        scaled_0 = _mm256_add_epi32(scaled_0, contribution);
    }

    {
        const __m256i rhs_0123 = _mm256_and_si256(rhs_raw_0123, m4b);
        const __m256i rhs_4567 = _mm256_and_si256(rhs_raw_4567, m4b);
        __m256i lhs = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_1_base + LHS_LOW_OFFSET)));
        lhs = _mm256_permute2f128_si256(lhs, lhs, 0);
        const __m256i low = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_0)),
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_1)));
        __m256i contribution = _mm256_madd_epi16(low, scales_0);

        const __m256i rhs_high_0123 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0123, 4), m4b);
        const __m256i rhs_high_4567 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_4567, 4), m4b);
        lhs = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_1_base + LHS_HIGH_OFFSET)));
        lhs = _mm256_permute2f128_si256(lhs, lhs, 0);
        const __m256i high = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(rhs_high_0123, _mm256_shuffle_epi32(rhs_high_4567, 177), 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_0)),
            _mm256_maddubs_epi16(
                _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_high_0123, 177), rhs_high_4567, 170),
                _mm256_shuffle_epi32(lhs, SHUFFLE_1)));
        contribution = _mm256_add_epi32(contribution, _mm256_madd_epi16(high, scales_1));
        scaled_1 = _mm256_add_epi32(scaled_1, contribution);
    }
}

"""


PAIR_LOCAL_PREFIX = """                    const uint8_t * rhs_base = b_ptr[b].qs + sb * 256;
                    const int8_t * lhs_0_base = a_ptr_0[b].qs + sb * 64;
                    const int8_t * lhs_1_base = a_ptr_1[b].qs + sb * 64;
                    __m256i iacc_sb_0 = _mm256_setzero_si256();
                    __m256i iacc_sb_1 = _mm256_setzero_si256();

"""


PAIR_LOCAL_SUFFIX = """
                    ggml_q4_K_accumulate_scaled_raw_pair_pair2<0, 0, 32, 0, 85>(
                        rhs_base, lhs_0_base, lhs_1_base, scales_0, scales_1,
                        iacc_sb_0, iacc_sb_1);
                    ggml_q4_K_accumulate_scaled_raw_pair_pair2<64, 0, 32, 170, 255>(
                        rhs_base, lhs_0_base, lhs_1_base, scales_0, scales_1,
                        iacc_sb_0, iacc_sb_1);
                    ggml_q4_K_accumulate_scaled_raw_pair_pair2<128, 16, 48, 0, 85>(
                        rhs_base, lhs_0_base, lhs_1_base, scales_0, scales_1,
                        iacc_sb_0, iacc_sb_1);
                    ggml_q4_K_accumulate_scaled_raw_pair_pair2<192, 16, 48, 170, 255>(
                        rhs_base, lhs_0_base, lhs_1_base, scales_0, scales_1,
                        iacc_sb_0, iacc_sb_1);
                    const __m256i iacc_min_sb_0 = _mm256_madd_epi16(
                        _mm256_shuffle_epi32(q8s_0, 0), mins_01);
                    const __m256i iacc_min_sb_1 = _mm256_madd_epi16(
                        _mm256_shuffle_epi32(q8s_1, 0), mins_01);
                    q8s_0 = _mm256_bsrli_epi128(q8s_0, 4);
                    q8s_1 = _mm256_bsrli_epi128(q8s_1, 4);
                    iacc_b_0 = _mm256_add_epi32(iacc_b_0, iacc_sb_0);
                    iacc_b_1 = _mm256_add_epi32(iacc_b_1, iacc_sb_1);
                    iacc_min_b_0 = _mm256_add_epi32(iacc_min_b_0, iacc_min_sb_0);
                    iacc_min_b_1 = _mm256_add_epi32(iacc_min_b_1, iacc_min_sb_1);
"""


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    observed = text.count(old)
    if observed != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, observed {observed}")
    return text.replace(old, new)


def transform(parent: str) -> str:
    candidate = streamed_v26.transform(parent)
    candidate = replace_exact(
        candidate,
        streamed_v26.STREAM_RAW_PAIR_TEMPLATE,
        PAIR_LOCAL_SCALED_TEMPLATE,
        1,
        "raw-pair helper template",
    )
    candidate = replace_exact(
        candidate,
        streamed_v26.STREAMED_DOT_PREFIX,
        PAIR_LOCAL_PREFIX,
        1,
        "dot prefix",
    )
    candidate = replace_exact(
        candidate,
        streamed_v26.STREAMED_DOT_SUFFIX,
        PAIR_LOCAL_SUFFIX,
        1,
        "dot suffix",
    )
    candidate = replace_exact(candidate, OLD_HELPER, NEW_HELPER, 3, "pair2 helper name")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=FROZEN_PARENT_SHA256)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"parent SHA-256 mismatch: expected {args.expected_sha256}, observed {source_sha256}"
        )
    candidate_bytes = transform(source_bytes.decode("utf-8")).encode("utf-8")
    temporary = args.output.with_name(args.output.name + ".v27.tmp")
    temporary.write_bytes(candidate_bytes)
    os.chmod(temporary, args.source.stat().st_mode)
    os.replace(temporary, args.output)
    print(
        "{"
        f'"source_sha256":"{source_sha256}",'
        f'"candidate_sha256":"{hashlib.sha256(candidate_bytes).hexdigest()}",'
        f'"helper":"{NEW_HELPER}",'
        '"persistent_int16_dot_accumulators":0,'
        '"persistent_int32_dot_accumulators":2,'
        '"rows_per_pass":2'
        "}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
