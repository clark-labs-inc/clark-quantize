#include "repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & name, const char * message,
                       int block = -1, int row = -1, int index = -1) {
    std::fprintf(stderr,
                 "q8_k_dual_layout_producer_v24: %s: %s "
                 "(block=%d row=%d index=%d)\n",
                 name.c_str(), message, block, row, index);
    std::exit(1);
}

void verify_row_bsums(const std::string & name,
                      const std::vector<block_q8_K> & rows,
                      int nb) {
    for (int row = 0; row < 4; ++row) {
        for (int block = 0; block < nb; ++block) {
            const block_q8_K & current = rows[row * nb + block];
            for (int group16 = 0; group16 < QK_K / 16; ++group16) {
                int sum = 0;
                for (int lane = 0; lane < 16; ++lane) {
                    sum += current.qs[group16 * 16 + lane];
                }
                if (current.bsums[group16] != sum) {
                    fail(name, "row-major bsum does not match codes",
                         block, row, group16);
                }
            }
        }
    }
}

void compare_case(const std::string & name, const std::vector<float> & input,
                  bool require_consistent_bsums = true) {
    if (input.size() % (4 * QK_K) != 0) {
        fail(name, "input size is not four complete Q8_K rows");
    }

    const int nb = static_cast<int>(input.size() / (4 * QK_K));
    const int64_t k = static_cast<int64_t>(nb) * QK_K;
    const std::vector<float> before = input;

    std::vector<block_q8_Kx4> expected_native(nb);
    std::vector<block_q8_K> expected_rows(4 * nb);
    ggml_quantize_mat_q8_K_4x8(input.data(), expected_native.data(), k);
    ggml_repack_q8_K_4x8_to_q8_K_rows(
        expected_native.data(), expected_rows.data(), k);

    std::vector<block_q8_Kx4> actual_native(nb);
    std::vector<block_q8_K> actual_rows(4 * nb);
    ggml_quantize_mat_q8_K_4x8_dual(
        input.data(), actual_native.data(), actual_rows.data(), k);

    if (std::memcmp(input.data(), before.data(), input.size() * sizeof(float)) != 0) {
        fail(name, "source F32 bytes changed");
    }
    if (std::memcmp(actual_native.data(), expected_native.data(),
                    expected_native.size() * sizeof(expected_native[0])) != 0) {
        fail(name, "native output differs from parent producer");
    }
    if (std::memcmp(actual_rows.data(), expected_rows.data(),
                    expected_rows.size() * sizeof(expected_rows[0])) != 0) {
        fail(name, "row-major output differs from parent plus deinterleave");
    }
    if (require_consistent_bsums) {
        verify_row_bsums(name, actual_rows, nb);
    }
}

std::vector<float> one_block() {
    return std::vector<float>(4 * QK_K);
}

void run_special_cases() {
    compare_case("all_zero", one_block());

    auto extrema = one_block();
    for (size_t i = 0; i < extrema.size(); ++i) {
        static constexpr float values[] = {
            -127.0f, -17.0f, -1.0f, -0.0f, 0.0f, 1.0f, 19.0f, 127.0f};
        extrema[i] = values[i % 8];
    }
    compare_case("signed_extrema", extrema);

    auto alternating = one_block();
    for (size_t i = 0; i < alternating.size(); ++i) {
        alternating[i] = (i & 1) ? 3.25f : -3.25f;
    }
    compare_case("alternating_signs", alternating);

    auto repeated_maxima = one_block();
    for (int row = 0; row < 4; ++row) {
        for (int i = 0; i < QK_K; ++i) {
            repeated_maxima[row * QK_K + i] =
                (i % 31 == 0) ? ((i & 1) ? 11.0f : -11.0f)
                              : static_cast<float>((i % 13) - 6);
        }
    }
    compare_case("repeated_opposite_signed_maxima", repeated_maxima);

    auto distinct_scales = one_block();
    for (int row = 0; row < 4; ++row) {
        const float scale = std::ldexp(1.0f, row - 2);
        for (int i = 0; i < QK_K; ++i) {
            distinct_scales[row * QK_K + i] =
                scale * static_cast<float>((i % 29) - 14);
        }
    }
    compare_case("distinct_per_row_scales", distinct_scales);

    auto nonfinite = one_block();
    nonfinite[0] = std::numeric_limits<float>::infinity();
    nonfinite[QK_K + 1] = -std::numeric_limits<float>::infinity();
    nonfinite[2 * QK_K + 2] = std::numeric_limits<float>::quiet_NaN();
    nonfinite[3 * QK_K + 3] = -std::numeric_limits<float>::quiet_NaN();
    compare_case("nonfinite_parent_equivalence", nonfinite, false);
}

} // namespace

int main() {
    static_assert(sizeof(block_q8_Kx4) == 4 * sizeof(block_q8_K));

    constexpr int nb = 1024;
    std::vector<float> seeded(static_cast<size_t>(4 * nb * QK_K));
    std::mt19937 rng(0x38'27'24u);
    std::uniform_real_distribution<float> distribution(-17.0f, 19.0f);

    for (size_t i = 0; i < seeded.size(); ++i) {
        float value = distribution(rng);
        if (i % 257 == 0) {
            value = 0.0f;
        } else if (i % 521 == 0) {
            value = (i & 1) ? 127.0f : -127.0f;
        } else if (i % 1031 == 0) {
            value = (i & 1) ? 0.000001f : -0.000001f;
        }
        seeded[i] = value;
    }

    compare_case("seeded", seeded);
    run_special_cases();

    std::printf(
        "{\"schema\":\"q8_k_dual_layout_producer_v24_test\","
        "\"seeded_blocks\":%d,\"special_cases\":6,"
        "\"source_f32_unchanged\":true,"
        "\"native_matches_parent_bit_exact\":true,"
        "\"row_major_matches_parent_bridge_bit_exact\":true,"
        "\"row_major_bsums_match_codes\":true}\n",
        nb);
    return 0;
}
