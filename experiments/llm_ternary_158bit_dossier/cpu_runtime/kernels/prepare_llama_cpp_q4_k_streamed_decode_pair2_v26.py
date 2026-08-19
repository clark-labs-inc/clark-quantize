#!/usr/bin/env python3
"""Prepare the v26 exact Q4_K streamed-decode two-row AVX2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import prepare_llama_cpp_q4_k_register_pressure_pair2_v25 as pair2_v25


FROZEN_PARENT_SHA256 = pair2_v25.FROZEN_PARENT_SHA256
OLD_HELPER = pair2_v25.HELPER_NAME
NEW_HELPER = "ggml_gemv_q4_K_8x8_q8_K_streamed_pair2_tile"


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    observed = text.count(old)
    if observed != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, observed {observed}")
    return text.replace(old, new)


STREAM_RAW_PAIR_TEMPLATE = r"""template <int RAW_OFFSET, int LHS_LOW_OFFSET, int LHS_HIGH_OFFSET, int SHUFFLE_0, int SHUFFLE_1>
static inline __attribute__((always_inline)) void ggml_q4_K_stream_raw_pair_pair2(
        const uint8_t * GGML_RESTRICT rhs_base,
        const int8_t * GGML_RESTRICT lhs_0_base,
        const int8_t * GGML_RESTRICT lhs_1_base,
        __m256i & low_0,
        __m256i & high_0,
        __m256i & low_1,
        __m256i & high_1) {
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i rhs_raw_0123 =
        _mm256_loadu_si256((const __m256i *) (rhs_base + RAW_OFFSET));
    const __m256i rhs_raw_4567 =
        _mm256_loadu_si256((const __m256i *) (rhs_base + RAW_OFFSET + 32));

    {
        const __m256i rhs_0123 = _mm256_and_si256(rhs_raw_0123, m4b);
        const __m256i rhs_4567 = _mm256_and_si256(rhs_raw_4567, m4b);
        __m256i lhs_0 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_0_base + LHS_LOW_OFFSET)));
        lhs_0 = _mm256_permute2f128_si256(lhs_0, lhs_0, 0);
        low_0 = _mm256_add_epi16(low_0, _mm256_maddubs_epi16(
            _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
            _mm256_shuffle_epi32(lhs_0, SHUFFLE_0)));
        low_0 = _mm256_add_epi16(low_0, _mm256_maddubs_epi16(
            _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
            _mm256_shuffle_epi32(lhs_0, SHUFFLE_1)));

        __m256i lhs_1 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_1_base + LHS_LOW_OFFSET)));
        lhs_1 = _mm256_permute2f128_si256(lhs_1, lhs_1, 0);
        low_1 = _mm256_add_epi16(low_1, _mm256_maddubs_epi16(
            _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
            _mm256_shuffle_epi32(lhs_1, SHUFFLE_0)));
        low_1 = _mm256_add_epi16(low_1, _mm256_maddubs_epi16(
            _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
            _mm256_shuffle_epi32(lhs_1, SHUFFLE_1)));
    }

    {
        const __m256i rhs_0123 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0123, 4), m4b);
        const __m256i rhs_4567 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_4567, 4), m4b);
        __m256i lhs_0 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_0_base + LHS_HIGH_OFFSET)));
        lhs_0 = _mm256_permute2f128_si256(lhs_0, lhs_0, 0);
        high_0 = _mm256_add_epi16(high_0, _mm256_maddubs_epi16(
            _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
            _mm256_shuffle_epi32(lhs_0, SHUFFLE_0)));
        high_0 = _mm256_add_epi16(high_0, _mm256_maddubs_epi16(
            _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
            _mm256_shuffle_epi32(lhs_0, SHUFFLE_1)));

        __m256i lhs_1 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *) (lhs_1_base + LHS_HIGH_OFFSET)));
        lhs_1 = _mm256_permute2f128_si256(lhs_1, lhs_1, 0);
        high_1 = _mm256_add_epi16(high_1, _mm256_maddubs_epi16(
            _mm256_blend_epi32(rhs_0123, _mm256_shuffle_epi32(rhs_4567, 177), 170),
            _mm256_shuffle_epi32(lhs_1, SHUFFLE_0)));
        high_1 = _mm256_add_epi16(high_1, _mm256_maddubs_epi16(
            _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123, 177), rhs_4567, 170),
            _mm256_shuffle_epi32(lhs_1, SHUFFLE_1)));
    }
}

"""


FP_ACCUMULATORS = """        __m256 acc_rows[2] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
        };
        __m256 acc_min_rows[2] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
        };
"""

NAMED_FP_ACCUMULATORS = """        __m256 acc_row_0 = _mm256_setzero_ps();
        __m256 acc_row_1 = _mm256_setzero_ps();
        __m256 acc_min_row_0 = _mm256_setzero_ps();
        __m256 acc_min_row_1 = _mm256_setzero_ps();
"""

INTEGER_ACCUMULATORS = """                __m256i iacc_b[2] = {
                    _mm256_setzero_si256(), _mm256_setzero_si256(),
                };
                __m256i iacc_min_b[2] = {
                    _mm256_setzero_si256(), _mm256_setzero_si256(),
                };
                __m256i q8s[2];
                for (int64_t y = 0; y < nr; ++y) {
                    const block_q8_K * a_ptr = a_ptr_start + y * nb;
                    const __m256i q8sums =
                        _mm256_loadu_si256((const __m256i *) a_ptr[b].bsums);
                    q8s[y] = _mm256_castsi128_si256(_mm_hadd_epi16(
                        _mm256_castsi256_si128(q8sums),
                        _mm256_extracti128_si256(q8sums, 1)));
                    q8s[y] = _mm256_permute2f128_si256(q8s[y], q8s[y], 0);
                }
"""

NAMED_INTEGER_ACCUMULATORS = """                __m256i iacc_b_0 = _mm256_setzero_si256();
                __m256i iacc_b_1 = _mm256_setzero_si256();
                __m256i iacc_min_b_0 = _mm256_setzero_si256();
                __m256i iacc_min_b_1 = _mm256_setzero_si256();
                const block_q8_K * a_ptr_0 = a_ptr_start;
                const block_q8_K * a_ptr_1 = a_ptr_start + nb;
                const __m256i q8sums_0 =
                    _mm256_loadu_si256((const __m256i *) a_ptr_0[b].bsums);
                const __m256i q8sums_1 =
                    _mm256_loadu_si256((const __m256i *) a_ptr_1[b].bsums);
                __m256i q8s_0 = _mm256_castsi128_si256(_mm_hadd_epi16(
                    _mm256_castsi256_si128(q8sums_0),
                    _mm256_extracti128_si256(q8sums_0, 1)));
                __m256i q8s_1 = _mm256_castsi128_si256(_mm_hadd_epi16(
                    _mm256_castsi256_si128(q8sums_1),
                    _mm256_extracti128_si256(q8sums_1, 1)));
                q8s_0 = _mm256_permute2f128_si256(q8s_0, q8s_0, 0);
                q8s_1 = _mm256_permute2f128_si256(q8s_1, q8s_1, 0);
"""


STREAMED_DOT_PREFIX = """                    const uint8_t * rhs_base = b_ptr[b].qs + sb * 256;
                    const int8_t * lhs_0_base = a_ptr_0[b].qs + sb * 64;
                    const int8_t * lhs_1_base = a_ptr_1[b].qs + sb * 64;
                    __m256i iacc_low_0 = _mm256_setzero_si256();
                    __m256i iacc_high_0 = _mm256_setzero_si256();
                    __m256i iacc_low_1 = _mm256_setzero_si256();
                    __m256i iacc_high_1 = _mm256_setzero_si256();

                    ggml_q4_K_stream_raw_pair_pair2<0, 0, 32, 0, 85>(
                        rhs_base, lhs_0_base, lhs_1_base,
                        iacc_low_0, iacc_high_0, iacc_low_1, iacc_high_1);
                    ggml_q4_K_stream_raw_pair_pair2<64, 0, 32, 170, 255>(
                        rhs_base, lhs_0_base, lhs_1_base,
                        iacc_low_0, iacc_high_0, iacc_low_1, iacc_high_1);
                    ggml_q4_K_stream_raw_pair_pair2<128, 16, 48, 0, 85>(
                        rhs_base, lhs_0_base, lhs_1_base,
                        iacc_low_0, iacc_high_0, iacc_low_1, iacc_high_1);
                    ggml_q4_K_stream_raw_pair_pair2<192, 16, 48, 170, 255>(
                        rhs_base, lhs_0_base, lhs_1_base,
                        iacc_low_0, iacc_high_0, iacc_low_1, iacc_high_1);

"""


STREAMED_DOT_SUFFIX = """
                    const __m256i iacc_sb_0 = _mm256_add_epi32(
                        _mm256_madd_epi16(iacc_low_0, scales_0),
                        _mm256_madd_epi16(iacc_high_0, scales_1));
                    const __m256i iacc_sb_1 = _mm256_add_epi32(
                        _mm256_madd_epi16(iacc_low_1, scales_0),
                        _mm256_madd_epi16(iacc_high_1, scales_1));
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


OLD_FP_UPDATE = """                // Multiply-Add with scale values for the complete super block
                for (int64_t y = 0; y < nr; ++y) {
                    const block_q8_K * a_ptr = a_ptr_start + y * nb;
                    const __m256 row_scale_f32 = _mm256_set1_ps(a_ptr[b].d);
                    acc_rows[y] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_b[y]),
                        _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_rows[y]);
                    acc_min_rows[y] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_min_b[y]),
                        _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows[y]);
                }
"""

NEW_FP_UPDATE = """                // Preserve each row's original per-super-block FP accumulation order.
                const __m256 row_scale_f32_0 = _mm256_set1_ps(a_ptr_0[b].d);
                const __m256 row_scale_f32_1 = _mm256_set1_ps(a_ptr_1[b].d);
                acc_row_0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_b_0),
                    _mm256_mul_ps(col_scale_f32, row_scale_f32_0), acc_row_0);
                acc_row_1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_b_1),
                    _mm256_mul_ps(col_scale_f32, row_scale_f32_1), acc_row_1);
                acc_min_row_0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_min_b_0),
                    _mm256_mul_ps(col_dmin_f32, row_scale_f32_0), acc_min_row_0);
                acc_min_row_1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_min_b_1),
                    _mm256_mul_ps(col_dmin_f32, row_scale_f32_1), acc_min_row_1);
"""

OLD_STORE = """            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            for (int64_t y = 0; y < nr; ++y) {
                acc_rows[y] = _mm256_permutevar8x32_ps(acc_rows[y], finalpermutemask);
                _mm256_storeu_ps(
                    s + y * bs + x * 8,
                    _mm256_sub_ps(acc_rows[y], acc_min_rows[y]));
            }
"""

NEW_STORE = """            // Accumulated output values permuted into the existing output order.
            acc_row_0 = _mm256_permutevar8x32_ps(acc_row_0, finalpermutemask);
            acc_row_1 = _mm256_permutevar8x32_ps(acc_row_1, finalpermutemask);
            _mm256_storeu_ps(s + x * 8, _mm256_sub_ps(acc_row_0, acc_min_row_0));
            _mm256_storeu_ps(s + bs + x * 8, _mm256_sub_ps(acc_row_1, acc_min_row_1));
"""


def transform(parent: str) -> str:
    candidate = pair2_v25.transform(parent).replace(OLD_HELPER, NEW_HELPER)
    helper_start = candidate.find(f"static __attribute__((noinline)) void {NEW_HELPER}")
    helper_end = candidate.find("#endif\n\nvoid ggml_gemv_q4_K_8x8_q8_K(", helper_start)
    if helper_start < 0 or helper_end < 0:
        raise RuntimeError("generated pair2 helper boundaries not found")
    helper = candidate[helper_start:helper_end]

    helper = replace_exact(helper, FP_ACCUMULATORS, NAMED_FP_ACCUMULATORS, 1, "FP accumulators")
    helper = replace_exact(
        helper,
        INTEGER_ACCUMULATORS,
        NAMED_INTEGER_ACCUMULATORS,
        1,
        "integer accumulators",
    )
    helper = replace_exact(
        helper,
        "    // Mask to extract nibbles from bytes\n    const __m256i m4b = _mm256_set1_epi8(0x0F);\n\n",
        "",
        1,
        "obsolete helper nibble mask",
    )

    inner_start_marker = "                    // Load the eight block_q4_K for two sub blocks"
    inner_end_marker = "\n                }\n\n                // Multiply-Add with scale values for the complete super block"
    inner_start = helper.find(inner_start_marker)
    inner_end = helper.find(inner_end_marker, inner_start)
    if inner_start < 0 or inner_end < 0:
        raise RuntimeError("pair2 inner dot region not found")
    old_inner = helper[inner_start:inner_end]
    scale_start = old_inner.find("                    uint32_t utmp_0[4], utmp_1[4];")
    scale_end = old_inner.find("                    // The Q4_K decode above is shared.", scale_start)
    if scale_start < 0 or scale_end < 0:
        raise RuntimeError("Q4_K scale decode region not found")
    scale_decode = old_inner[scale_start:scale_end]
    new_inner = STREAMED_DOT_PREFIX + scale_decode + STREAMED_DOT_SUFFIX
    helper = helper[:inner_start] + new_inner + helper[inner_end:]

    helper = replace_exact(helper, OLD_FP_UPDATE, NEW_FP_UPDATE, 1, "FP update")
    helper = replace_exact(helper, OLD_STORE, NEW_STORE, 1, "output stores")
    candidate = candidate[:helper_start] + helper + candidate[helper_end:]

    template_insertion = candidate.find(f"static __attribute__((noinline)) void {NEW_HELPER}")
    candidate = candidate[:template_insertion] + STREAM_RAW_PAIR_TEMPLATE + candidate[template_insertion:]
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
    temporary = args.output.with_name(args.output.name + ".v26.tmp")
    temporary.write_bytes(candidate_bytes)
    os.chmod(temporary, args.source.stat().st_mode)
    os.replace(temporary, args.output)
    print(
        "{"
        f'"source_sha256":"{source_sha256}",'
        f'"candidate_sha256":"{hashlib.sha256(candidate_bytes).hexdigest()}",'
        f'"helper":"{NEW_HELPER}",'
        '"raw_vectors_streamed_per_step":2,'
        '"rows_per_pass":2'
        "}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
