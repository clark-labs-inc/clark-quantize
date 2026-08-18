#include "repack.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

[[noreturn]] void fail(const char * message, int block = -1, int row = -1, int index = -1) {
    std::fprintf(stderr,
                 "q8_k_x4_deinterleave_v21: %s (block=%d row=%d index=%d)\n",
                 message, block, row, index);
    std::exit(1);
}

void verify_mapping(const std::vector<block_q8_Kx4> & src,
                    const std::vector<block_q8_K> & dst,
                    bool require_consistent_bsums) {
    const int nb = static_cast<int>(src.size());
    if (dst.size() != static_cast<size_t>(4 * nb)) {
        fail("destination size mismatch");
    }

    for (int b = 0; b < nb; ++b) {
        for (int row = 0; row < 4; ++row) {
            const block_q8_K & out = dst[row * nb + b];
            if (std::memcmp(&out.d, &src[b].d[row], sizeof(out.d)) != 0) {
                fail("scale bits changed", b, row);
            }

            for (int k = 0; k < QK_K; ++k) {
                const int packed = (k / 8) * 32 + row * 8 + (k % 8);
                if (out.qs[k] != src[b].qs[packed]) {
                    fail("quant code mapping mismatch", b, row, k);
                }
            }

            for (int group16 = 0; group16 < QK_K / 16; ++group16) {
                const int tiled = (group16 / 4) * 16 + row * 4 + (group16 % 4);
                const int16_t expected = src[b].bsums[tiled];
                if (out.bsums[group16] != expected) {
                    fail("bsums mapping mismatch", b, row, group16);
                }
                if (require_consistent_bsums) {
                    int sum = 0;
                    for (int lane = 0; lane < 16; ++lane) {
                        sum += out.qs[group16 * 16 + lane];
                    }
                    if (out.bsums[group16] != sum) {
                        fail("deinterleaved bsum does not match codes", b, row, group16);
                    }
                }
            }
        }
    }
}

void verify_inverse_roundtrip(const std::vector<block_q8_Kx4> & src,
                              const std::vector<block_q8_K> & rows) {
    const int nb = static_cast<int>(src.size());
    std::vector<block_q8_Kx4> reconstructed(nb);

    for (int b = 0; b < nb; ++b) {
        for (int row = 0; row < 4; ++row) {
            const block_q8_K & in = rows[row * nb + b];
            reconstructed[b].d[row] = in.d;
            for (int group16 = 0; group16 < QK_K / 16; ++group16) {
                const int tiled = (group16 / 4) * 16 + row * 4 + (group16 % 4);
                reconstructed[b].bsums[tiled] = in.bsums[group16];
            }
            for (int group8 = 0; group8 < QK_K / 8; ++group8) {
                std::memcpy(reconstructed[b].qs + group8 * 32 + row * 8,
                            in.qs + group8 * 8, 8);
            }
        }
    }

    if (std::memcmp(src.data(), reconstructed.data(), src.size() * sizeof(src[0])) != 0) {
        fail("inverse roundtrip changed packed bytes");
    }
}

void run_raw_special_cases() {
    std::vector<block_q8_Kx4> src(3);
    std::memset(src.data(), 0, src.size() * sizeof(src[0]));

    for (int row = 0; row < 4; ++row) {
        src[1].d[row] = (row + 1) * 0.03125f;
        src[2].d[row] = -(row + 1) * 1.75f;
    }
    for (int b = 1; b < 3; ++b) {
        for (int packed = 0; packed < QK_K * 4; ++packed) {
            static constexpr int8_t extrema[] = {-128, -127, -1, 0, 1, 126, 127};
            src[b].qs[packed] = extrema[(packed + 3 * b) % 7];
        }
        for (int row = 0; row < 4; ++row) {
            for (int group16 = 0; group16 < QK_K / 16; ++group16) {
                const int tiled = (group16 / 4) * 16 + row * 4 + (group16 % 4);
                src[b].bsums[tiled] =
                    static_cast<int16_t>(-1200 + b * 300 + row * 64 + group16);
            }
        }
    }

    const auto before = src;
    std::vector<block_q8_K> rows(src.size() * 4);
    ggml_repack_q8_K_4x8_to_q8_K_rows(src.data(), rows.data(),
                                      static_cast<int64_t>(src.size()) * QK_K);
    if (std::memcmp(src.data(), before.data(), src.size() * sizeof(src[0])) != 0) {
        fail("raw source mutated");
    }
    verify_mapping(src, rows, false);
    verify_inverse_roundtrip(src, rows);
}

} // namespace

int main() {
    static_assert(sizeof(block_q8_Kx4) == 4 * sizeof(block_q8_K));

    constexpr int nb = 1024;
    constexpr int64_t k = static_cast<int64_t>(nb) * QK_K;
    std::vector<float> input(static_cast<size_t>(4 * k));
    std::mt19937 rng(0x38'27'21u);
    std::uniform_real_distribution<float> distribution(-17.0f, 19.0f);

    for (size_t i = 0; i < input.size(); ++i) {
        float value = distribution(rng);
        if (i % 257 == 0) {
            value = 0.0f;
        } else if (i % 521 == 0) {
            value = (i & 1) ? 127.0f : -127.0f;
        } else if (i % 1031 == 0) {
            value = (i & 1) ? 0.000001f : -0.000001f;
        }
        input[i] = value;
    }

    std::vector<block_q8_Kx4> packed(nb);
    ggml_quantize_mat_q8_K_4x8(input.data(), packed.data(), k);
    const auto packed_before = packed;

    std::vector<block_q8_K> rows(4 * nb);
    ggml_repack_q8_K_4x8_to_q8_K_rows(packed.data(), rows.data(), k);

    if (std::memcmp(packed.data(), packed_before.data(), packed.size() * sizeof(packed[0])) != 0) {
        fail("native source mutated");
    }
    verify_mapping(packed, rows, true);
    verify_inverse_roundtrip(packed, rows);
    run_raw_special_cases();

    std::printf(
        "{\"schema\":\"q8_k_x4_deinterleave_v21_test\","
        "\"seeded_native_blocks\":%d,\"raw_special_blocks\":3,"
        "\"source_unchanged\":true,\"mapping_bit_exact\":true,"
        "\"bsums_match_codes\":true,\"inverse_roundtrip_bit_exact\":true}\n",
        nb);
    return 0;
}
