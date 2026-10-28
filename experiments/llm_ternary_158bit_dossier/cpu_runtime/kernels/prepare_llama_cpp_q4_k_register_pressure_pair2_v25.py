#!/usr/bin/env python3
"""Prepare the frozen v25 Q4_K AVX2 pair-two-row candidate.

The parent kernel is retained byte-for-byte for nr=1,2,3.  For nr=4 this
transform inserts a no-inline copy whose row count and output-column count are
compile-time constants (2 and 8), shrinks only its row accumulator arrays, and
calls it twice inside each output-column tile.  The duplicate weight decode is
therefore cache-local rather than a second whole-matrix scan.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


FROZEN_PARENT_SHA256 = "c171a06d5a5d5d5221a6167c7039b759e38efd073dca7ba33a46afa8ed6b1ec2"

FUNCTION_HEADER = (
    "void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, "
    "const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {"
)
NEXT_FUNCTION_HEADER = "void ggml_gemv_iq4_nl_8x8_q8_0("
HELPER_NAME = "ggml_gemv_q4_K_8x8_q8_K_pair2_tile"


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    observed = text.count(old)
    if observed != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, observed {observed}")
    return text.replace(old, new)


def transform(parent: str) -> str:
    function_start = parent.find(FUNCTION_HEADER)
    if function_start < 0:
        raise RuntimeError("Q4_K function header not found")
    function_end = parent.find(NEXT_FUNCTION_HEADER, function_start)
    if function_end < 0:
        raise RuntimeError("function following Q4_K kernel not found")

    original = parent[function_start:function_end]
    helper_header = (
        "static __attribute__((noinline)) void "
        f"{HELPER_NAME}(int n, float * GGML_RESTRICT s, size_t bs, "
        "const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {\n"
        "    constexpr int nr = 2;\n"
        "    constexpr int nc = 8;"
    )
    helper = replace_exact(original, FUNCTION_HEADER, helper_header, 1, "helper header")
    for declaration in ("acc_rows", "acc_min_rows", "iacc_b", "iacc_min_b"):
        helper = replace_exact(
            helper,
            f"{declaration}[4]",
            f"{declaration}[2]",
            1,
            f"{declaration} extent",
        )
    helper = replace_exact(helper, "q8s[4]", "q8s[2]", 1, "q8s extent")
    helper = replace_exact(
        helper,
        "_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),",
        "_mm256_setzero_ps(), _mm256_setzero_ps(),",
        2,
        "FP accumulator initializers",
    )
    helper = replace_exact(
        helper,
        "_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),",
        "_mm256_setzero_si256(), _mm256_setzero_si256(),",
        2,
        "integer accumulator initializers",
    )

    dispatch_needle = """    assert(nr > 0 && nr <= 4);

    // Decode each Q4_K weight block once and reuse it across all ragged
"""
    dispatch = f"""    assert(nr > 0 && nr <= 4);

#if defined(__GNUC__) || defined(__clang__)
    if (nr == 4) {{
        // Two compile-time two-row passes run consecutively for every output
        // tile.  The second pass immediately revisits the same approximately
        // 23 KiB weight tile while halving the live accumulator set.
        for (int64_t x = 0; x < nc / 8; ++x) {{
            const block_q4_Kx8 * b_ptr = b_ptr_start + x * b_nb;
            {HELPER_NAME}(n, s + x * 8, bs, b_ptr, a_ptr_start);
            {HELPER_NAME}(n, s + 2 * bs + x * 8, bs, b_ptr, a_ptr_start + 2 * b_nb);
        }}
        return;
    }}
#endif

    // Decode each Q4_K weight block once and reuse it across all ragged
"""
    original_with_dispatch = replace_exact(
        original,
        dispatch_needle,
        dispatch,
        1,
        "nr=4 dispatch insertion",
    )
    helper = (
        "#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))\n"
        + helper
        + "#endif\n\n"
    )
    transformed = (
        parent[:function_start]
        + helper
        + original_with_dispatch
        + parent[function_end:]
    )
    return transformed


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
    candidate = transform(source_bytes.decode("utf-8"))
    candidate_bytes = candidate.encode("utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".v25.tmp")
    temporary.write_bytes(candidate_bytes)
    os.chmod(temporary, args.source.stat().st_mode)
    os.replace(temporary, args.output)

    print(
        "{"
        f'"source_sha256":"{source_sha256}",'
        f'"candidate_sha256":"{hashlib.sha256(candidate_bytes).hexdigest()}",'
        f'"helper":"{HELPER_NAME}",'
        '"nr4_pair_passes":2,'
        '"rows_per_pass":2'
        "}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
