#!/usr/bin/env python3
"""Prepare the v29 exact Q4_K out-of-line row two-row AVX2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import prepare_llama_cpp_q4_k_row_serial_scaled_pair2_v28 as row_serial_v28


pair_local_v27 = row_serial_v28.pair_local_v27
FROZEN_PARENT_SHA256 = pair_local_v27.FROZEN_PARENT_SHA256
OLD_HELPER = pair_local_v27.NEW_HELPER
NEW_HELPER = "ggml_gemv_q4_K_8x8_q8_K_out_of_line_row_pair2_tile"
ROW_MICROKERNEL = "ggml_q4_K_accumulate_scaled_four_pairs_row"


OUT_OF_LINE_ROW_TEMPLATE = row_serial_v28.ROW_SERIAL_SCALED_TEMPLATE + r"""static __attribute__((noinline)) __m256i ggml_q4_K_accumulate_scaled_four_pairs_row(
        const uint8_t * GGML_RESTRICT rhs_base,
        const int8_t * GGML_RESTRICT lhs_base,
        const __m256i scales_0,
        const __m256i scales_1) {
    __m256i scaled = _mm256_setzero_si256();
    ggml_q4_K_accumulate_scaled_raw_pair_row<0, 0, 32, 0, 85>(
        rhs_base, lhs_base, scales_0, scales_1, scaled);
    ggml_q4_K_accumulate_scaled_raw_pair_row<64, 0, 32, 170, 255>(
        rhs_base, lhs_base, scales_0, scales_1, scaled);
    ggml_q4_K_accumulate_scaled_raw_pair_row<128, 16, 48, 0, 85>(
        rhs_base, lhs_base, scales_0, scales_1, scaled);
    ggml_q4_K_accumulate_scaled_raw_pair_row<192, 16, 48, 170, 255>(
        rhs_base, lhs_base, scales_0, scales_1, scaled);
    return scaled;
}

"""


OUT_OF_LINE_PREFIX = """                    const uint8_t * rhs_base = b_ptr[b].qs + sb * 256;
                    const int8_t * lhs_0_base = a_ptr_0[b].qs + sb * 64;
                    const int8_t * lhs_1_base = a_ptr_1[b].qs + sb * 64;

"""


OUT_OF_LINE_SUFFIX = """
                    const __m256i iacc_sb_0 = ggml_q4_K_accumulate_scaled_four_pairs_row(
                        rhs_base, lhs_0_base, scales_0, scales_1);
                    const __m256i iacc_sb_1 = ggml_q4_K_accumulate_scaled_four_pairs_row(
                        rhs_base, lhs_1_base, scales_0, scales_1);
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
    candidate = pair_local_v27.transform(parent)
    candidate = replace_exact(
        candidate,
        pair_local_v27.PAIR_LOCAL_SCALED_TEMPLATE,
        OUT_OF_LINE_ROW_TEMPLATE,
        1,
        "pair-local helper template",
    )
    candidate = replace_exact(
        candidate,
        pair_local_v27.PAIR_LOCAL_PREFIX,
        OUT_OF_LINE_PREFIX,
        1,
        "dot prefix",
    )
    candidate = replace_exact(
        candidate,
        pair_local_v27.PAIR_LOCAL_SUFFIX,
        OUT_OF_LINE_SUFFIX,
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
    temporary = args.output.with_name(args.output.name + ".v29.tmp")
    temporary.write_bytes(candidate_bytes)
    os.chmod(temporary, args.source.stat().st_mode)
    os.replace(temporary, args.output)
    print(
        "{"
        f'"source_sha256":"{source_sha256}",'
        f'"candidate_sha256":"{hashlib.sha256(candidate_bytes).hexdigest()}",'
        f'"tile_helper":"{NEW_HELPER}",'
        f'"row_microkernel":"{ROW_MICROKERNEL}",'
        '"row_microkernel_calls_per_superblock":2'
        "}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
