// Fused AFF3_X16 research kernel for Qwen/Qwen3.8-27B on AVX2 AMD CPUs.
//
// The persistent artifact remains canonical base-3.  This file consumes the
// losslessly transcoded two-bit compute image frozen in the v17 protocol.  It
// never reconstructs a floating-point weight matrix or an expanded matrix of
// per-group alpha/mu values.

#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#define CQ3_RESTRICT __restrict__
#else
#define CQ3_RESTRICT
#endif

namespace {

constexpr int kGroupSize = 128;
constexpr int kPackedBytes = 32;
constexpr int kChunks = 8;

struct Aff3RowGroup {
    uint8_t qs[kPackedBytes];
    uint8_t scale_indices;
};
static_assert(sizeof(Aff3RowGroup) == 33, "AFF3 row/group layout changed");

struct Aff3X8Group {
    uint8_t scale_indices[8];
    uint8_t qs[8 * kPackedBytes];
};
static_assert(sizeof(Aff3X8Group) == 264, "AFF3_X8 layout changed");

struct Aff3X16Group {
    uint8_t scale_indices[16];
    uint8_t qs[16 * kPackedBytes];
};
static_assert(sizeof(Aff3X16Group) == 528, "AFF3_X16 layout changed");

inline bool codes_are_canonical(const uint8_t * packed) {
    for (int index = 0; index < kPackedBytes; ++index) {
        const uint8_t value = packed[index];
        if ((value & 3U) == 3U || ((value >> 2U) & 3U) == 3U ||
                ((value >> 4U) & 3U) == 3U || ((value >> 6U) & 3U) == 3U) {
            return false;
        }
    }
    return true;
}

inline float half_to_float(uint16_t value) {
    return _cvtsh_ss(value);
}

inline uint8_t alpha_index(uint8_t combined) {
    return combined & 0x0fU;
}

inline uint8_t mu_index(uint8_t combined) {
    return combined >> 4U;
}

inline __m256i chunk_dot(
        __m256i packed,
        __m256i values0,
        __m256i values1,
        __m256i values2,
        __m256i values3) {
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i digits0 = _mm256_and_si256(packed, mask);
    const __m256i digits1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
    const __m256i digits2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    const __m256i digits3 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
    const __m256i products01 = _mm256_add_epi16(
        _mm256_maddubs_epi16(digits0, values0),
        _mm256_maddubs_epi16(digits1, values1));
    const __m256i products23 = _mm256_add_epi16(
        _mm256_maddubs_epi16(digits2, values2),
        _mm256_maddubs_epi16(digits3, values3));
    return _mm256_madd_epi16(_mm256_add_epi16(products01, products23), ones);
}

inline void load_activation_words(
        const int8_t * activation,
        int chunk,
        __m256i * values0,
        __m256i * values1,
        __m256i * values2,
        __m256i * values3) {
    int32_t word0;
    int32_t word1;
    int32_t word2;
    int32_t word3;
    std::memcpy(&word0, activation + chunk * 4 + 0, 4);
    std::memcpy(&word1, activation + chunk * 4 + 32, 4);
    std::memcpy(&word2, activation + chunk * 4 + 64, 4);
    std::memcpy(&word3, activation + chunk * 4 + 96, 4);
    *values0 = _mm256_set1_epi32(word0);
    *values1 = _mm256_set1_epi32(word1);
    *values2 = _mm256_set1_epi32(word2);
    *values3 = _mm256_set1_epi32(word3);
}

inline __m256 gather_ratios(const float * ratios, __m128i bytes) {
    return _mm256_i32gather_ps(ratios, _mm256_cvtepu8_epi32(bytes), 4);
}

inline void split_scale_indices(
        const uint8_t * combined,
        __m256 * alpha_ratio0,
        __m256 * alpha_ratio1,
        __m256 * mu_ratio0,
        __m256 * mu_ratio1,
        const float * alpha_ratios,
        const float * mu_ratios) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(combined));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i alpha = _mm_and_si128(packed, nibble_mask);
    const __m128i mu = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
    *alpha_ratio0 = gather_ratios(alpha_ratios, alpha);
    *alpha_ratio1 = gather_ratios(alpha_ratios, _mm_srli_si128(alpha, 8));
    *mu_ratio0 = gather_ratios(mu_ratios, mu);
    *mu_ratio1 = gather_ratios(mu_ratios, _mm_srli_si128(mu, 8));
}

inline void decode_ratio_tables(
        const uint16_t * alpha_codebook,
        const uint16_t * mu_codebook,
        float * alpha_ratios,
        float * mu_ratios) {
    _mm256_storeu_ps(alpha_ratios + 0, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha_codebook + 0))));
    _mm256_storeu_ps(alpha_ratios + 8, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha_codebook + 8))));
    _mm256_storeu_ps(mu_ratios + 0, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(mu_codebook + 0))));
    _mm256_storeu_ps(mu_ratios + 8, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(mu_codebook + 8))));
}

inline float horizontal_sum(__m256 values) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(values),
        _mm256_extractf128_ps(values, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline int32_t dot_ordinary_group(const uint8_t * packed, const int8_t * activation) {
    const __m256i weights = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed));
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i digits0 = _mm256_and_si256(weights, mask);
    const __m256i digits1 = _mm256_and_si256(_mm256_srli_epi16(weights, 2), mask);
    const __m256i digits2 = _mm256_and_si256(_mm256_srli_epi16(weights, 4), mask);
    const __m256i digits3 = _mm256_and_si256(_mm256_srli_epi16(weights, 6), mask);
    const __m256i products01 = _mm256_add_epi16(
        _mm256_maddubs_epi16(digits0, _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(activation + 0))),
        _mm256_maddubs_epi16(digits1, _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(activation + 32))));
    const __m256i products23 = _mm256_add_epi16(
        _mm256_maddubs_epi16(digits2, _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(activation + 64))),
        _mm256_maddubs_epi16(digits3, _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(activation + 96))));
    const __m256i sum32 = _mm256_madd_epi16(
        _mm256_add_epi16(products01, products23), ones);
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), sum32);
    int32_t total = 0;
    for (int lane = 0; lane < 8; ++lane) {
        total += lanes[lane];
    }
    return total;
}

}  // namespace

extern "C" {

size_t cq3_aff3_row_group_size_v17() {
    return sizeof(Aff3RowGroup);
}

size_t cq3_aff3_x8_group_size_v17() {
    return sizeof(Aff3X8Group);
}

size_t cq3_aff3_x16_group_size_v17() {
    return sizeof(Aff3X16Group);
}

int cq3_aff3_repack_x8_v17(
        const void * CQ3_RESTRICT source_pointer,
        void * CQ3_RESTRICT destination_pointer,
        int rows,
        int groups) {
    if (source_pointer == nullptr || destination_pointer == nullptr || rows <= 0 ||
            rows % 8 != 0 || groups <= 0) {
        return 1;
    }
    const auto * source = static_cast<const Aff3RowGroup *>(source_pointer);
    auto * destination = static_cast<Aff3X8Group *>(destination_pointer);
    for (int row = 0; row < rows; row += 8) {
        for (int group = 0; group < groups; ++group) {
            auto & output = destination[(row / 8) * groups + group];
            for (int lane = 0; lane < 8; ++lane) {
                const auto & input = source[(row + lane) * groups + group];
                if (!codes_are_canonical(input.qs)) {
                    return 3;
                }
                output.scale_indices[lane] = input.scale_indices;
            }
            for (int chunk = 0; chunk < kChunks; ++chunk) {
                for (int lane = 0; lane < 8; ++lane) {
                    const auto & input = source[(row + lane) * groups + group];
                    std::memcpy(output.qs + chunk * 32 + lane * 4,
                        input.qs + chunk * 4, 4);
                }
            }
        }
    }
    return 0;
}

int cq3_aff3_repack_x16_v17(
        const void * CQ3_RESTRICT source_pointer,
        void * CQ3_RESTRICT destination_pointer,
        int rows,
        int groups) {
    if (source_pointer == nullptr || destination_pointer == nullptr || rows <= 0 ||
            rows % 16 != 0 || groups <= 0) {
        return 1;
    }
    const auto * source = static_cast<const Aff3RowGroup *>(source_pointer);
    auto * destination = static_cast<Aff3X16Group *>(destination_pointer);
    for (int row = 0; row < rows; row += 16) {
        for (int group = 0; group < groups; ++group) {
            auto & output = destination[(row / 16) * groups + group];
            for (int lane = 0; lane < 16; ++lane) {
                const auto & input = source[(row + lane) * groups + group];
                if (!codes_are_canonical(input.qs)) {
                    return 3;
                }
                output.scale_indices[lane] = input.scale_indices;
            }
            for (int chunk = 0; chunk < kChunks; ++chunk) {
                for (int lane = 0; lane < 16; ++lane) {
                    const auto & input = source[(row + lane) * groups + group];
                    std::memcpy(output.qs + chunk * 64 + lane * 4,
                        input.qs + chunk * 4, 4);
                }
            }
        }
    }
    return 0;
}

int cq3_aff3_gemv_ordinary_v17(
        const void * CQ3_RESTRICT weights_pointer,
        const uint16_t * CQ3_RESTRICT alpha_anchors,
        const uint16_t * CQ3_RESTRICT mu_anchors,
        const uint16_t * CQ3_RESTRICT alpha_codebook,
        const uint16_t * CQ3_RESTRICT mu_codebook,
        const int8_t * CQ3_RESTRICT activation,
        const float * CQ3_RESTRICT activation_scales,
        const int32_t * CQ3_RESTRICT activation_sums,
        int rows,
        int columns,
        float * CQ3_RESTRICT output) {
    if (weights_pointer == nullptr || alpha_anchors == nullptr || mu_anchors == nullptr ||
            alpha_codebook == nullptr || mu_codebook == nullptr || activation == nullptr ||
            activation_scales == nullptr || activation_sums == nullptr || output == nullptr ||
            rows <= 0 || columns <= 0 || columns % kGroupSize != 0) {
        return 1;
    }
    const int groups = columns / kGroupSize;
    const auto * weights = static_cast<const Aff3RowGroup *>(weights_pointer);
    float alpha_ratios[16];
    float mu_ratios[16];
    decode_ratio_tables(alpha_codebook, mu_codebook, alpha_ratios, mu_ratios);
    for (int row = 0; row < rows; ++row) {
        const float alpha_anchor_value = half_to_float(alpha_anchors[row]);
        const float mu_anchor_value = half_to_float(mu_anchors[row]);
        float total = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const auto & weight = weights[row * groups + group];
            const int32_t dot = dot_ordinary_group(
                weight.qs, activation + group * kGroupSize) - activation_sums[group];
            const float alpha = alpha_anchor_value *
                alpha_ratios[alpha_index(weight.scale_indices)];
            const float mu = mu_anchor_value * mu_ratios[mu_index(weight.scale_indices)];
            const float affine = std::fma(static_cast<float>(dot), alpha,
                static_cast<float>(activation_sums[group]) * mu);
            total = std::fma(affine, activation_scales[group], total);
        }
        output[row] = total;
    }
    return 0;
}

int cq3_aff3_gemv_x8_v17(
        const void * CQ3_RESTRICT weights_pointer,
        const uint16_t * CQ3_RESTRICT alpha_anchors,
        const uint16_t * CQ3_RESTRICT mu_anchors,
        const uint16_t * CQ3_RESTRICT alpha_codebook,
        const uint16_t * CQ3_RESTRICT mu_codebook,
        const int8_t * CQ3_RESTRICT activation,
        const float * CQ3_RESTRICT activation_scales,
        const int32_t * CQ3_RESTRICT activation_sums,
        int rows,
        int columns,
        float * CQ3_RESTRICT output) {
    if (weights_pointer == nullptr || alpha_anchors == nullptr || mu_anchors == nullptr ||
            alpha_codebook == nullptr || mu_codebook == nullptr || activation == nullptr ||
            activation_scales == nullptr || activation_sums == nullptr || output == nullptr ||
            rows <= 0 || rows % 8 != 0 || columns <= 0 || columns % kGroupSize != 0) {
        return 1;
    }
    const int groups = columns / kGroupSize;
    const auto * weights = static_cast<const Aff3X8Group *>(weights_pointer);
    float alpha_ratios[16];
    float mu_ratios[16];
    decode_ratio_tables(alpha_codebook, mu_codebook, alpha_ratios, mu_ratios);
    for (int row = 0; row < rows; row += 8) {
        const __m256 alpha_anchor_values = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(alpha_anchors + row)));
        const __m256 mu_anchor_values = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(mu_anchors + row)));
        __m256 total = _mm256_setzero_ps();
        for (int group = 0; group < groups; ++group) {
            const auto & weight = weights[(row / 8) * groups + group];
            __m256i dot = _mm256_setzero_si256();
            const int8_t * group_activation = activation + group * kGroupSize;
            for (int chunk = 0; chunk < kChunks; ++chunk) {
                __m256i values0;
                __m256i values1;
                __m256i values2;
                __m256i values3;
                load_activation_words(group_activation, chunk,
                    &values0, &values1, &values2, &values3);
                dot = _mm256_add_epi32(dot, chunk_dot(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                        weight.qs + chunk * 32)),
                    values0, values1, values2, values3));
            }
            dot = _mm256_sub_epi32(dot, _mm256_set1_epi32(activation_sums[group]));
            const __m128i indices = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(weight.scale_indices));
            const __m128i nibble_mask = _mm_set1_epi8(0x0f);
            const __m256 alpha = _mm256_mul_ps(alpha_anchor_values,
                gather_ratios(alpha_ratios, _mm_and_si128(indices, nibble_mask)));
            const __m256 mu = _mm256_mul_ps(mu_anchor_values,
                gather_ratios(mu_ratios, _mm_and_si128(
                    _mm_srli_epi16(indices, 4), nibble_mask)));
            const __m256 affine = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(dot), alpha,
                _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(activation_sums[group])), mu));
            total = _mm256_fmadd_ps(
                affine, _mm256_set1_ps(activation_scales[group]), total);
        }
        _mm256_storeu_ps(output + row, total);
    }
    return 0;
}

int cq3_aff3_gemv_x16_v17(
        const void * CQ3_RESTRICT weights_pointer,
        const uint16_t * CQ3_RESTRICT alpha_anchors,
        const uint16_t * CQ3_RESTRICT mu_anchors,
        const uint16_t * CQ3_RESTRICT alpha_codebook,
        const uint16_t * CQ3_RESTRICT mu_codebook,
        const int8_t * CQ3_RESTRICT activation,
        const float * CQ3_RESTRICT activation_scales,
        const int32_t * CQ3_RESTRICT activation_sums,
        int rows,
        int columns,
        float * CQ3_RESTRICT output) {
    if (weights_pointer == nullptr || alpha_anchors == nullptr || mu_anchors == nullptr ||
            alpha_codebook == nullptr || mu_codebook == nullptr || activation == nullptr ||
            activation_scales == nullptr || activation_sums == nullptr || output == nullptr ||
            rows <= 0 || rows % 16 != 0 || columns <= 0 || columns % kGroupSize != 0) {
        return 1;
    }
    const int groups = columns / kGroupSize;
    const auto * weights = static_cast<const Aff3X16Group *>(weights_pointer);
    float alpha_ratios[16];
    float mu_ratios[16];
    decode_ratio_tables(alpha_codebook, mu_codebook, alpha_ratios, mu_ratios);
    for (int row = 0; row < rows; row += 16) {
        const __m256 alpha_anchor0 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(alpha_anchors + row + 0)));
        const __m256 alpha_anchor1 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(alpha_anchors + row + 8)));
        const __m256 mu_anchor0 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(mu_anchors + row + 0)));
        const __m256 mu_anchor1 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(mu_anchors + row + 8)));
        __m256 total0 = _mm256_setzero_ps();
        __m256 total1 = _mm256_setzero_ps();
        for (int group = 0; group < groups; ++group) {
            const auto & weight = weights[(row / 16) * groups + group];
            __m256i dot0 = _mm256_setzero_si256();
            __m256i dot1 = _mm256_setzero_si256();
            const int8_t * group_activation = activation + group * kGroupSize;
            for (int chunk = 0; chunk < kChunks; ++chunk) {
                __m256i values0;
                __m256i values1;
                __m256i values2;
                __m256i values3;
                load_activation_words(group_activation, chunk,
                    &values0, &values1, &values2, &values3);
                const uint8_t * packed = weight.qs + chunk * 64;
                dot0 = _mm256_add_epi32(dot0, chunk_dot(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + 0)),
                    values0, values1, values2, values3));
                dot1 = _mm256_add_epi32(dot1, chunk_dot(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + 32)),
                    values0, values1, values2, values3));
            }
            const __m256i sum32 = _mm256_set1_epi32(activation_sums[group]);
            dot0 = _mm256_sub_epi32(dot0, sum32);
            dot1 = _mm256_sub_epi32(dot1, sum32);
            __m256 alpha_ratio0;
            __m256 alpha_ratio1;
            __m256 mu_ratio0;
            __m256 mu_ratio1;
            split_scale_indices(weight.scale_indices,
                &alpha_ratio0, &alpha_ratio1, &mu_ratio0, &mu_ratio1,
                alpha_ratios, mu_ratios);
            const __m256 alpha0 = _mm256_mul_ps(alpha_anchor0, alpha_ratio0);
            const __m256 alpha1 = _mm256_mul_ps(alpha_anchor1, alpha_ratio1);
            const __m256 mu0 = _mm256_mul_ps(mu_anchor0, mu_ratio0);
            const __m256 mu1 = _mm256_mul_ps(mu_anchor1, mu_ratio1);
            const __m256 sumf = _mm256_set1_ps(static_cast<float>(activation_sums[group]));
            const __m256 affine0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(dot0), alpha0, _mm256_mul_ps(sumf, mu0));
            const __m256 affine1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(dot1), alpha1, _mm256_mul_ps(sumf, mu1));
            const __m256 activation_scale = _mm256_set1_ps(activation_scales[group]);
            total0 = _mm256_fmadd_ps(affine0, activation_scale, total0);
            total1 = _mm256_fmadd_ps(affine1, activation_scale, total1);
        }
        _mm256_storeu_ps(output + row + 0, total0);
        _mm256_storeu_ps(output + row + 8, total1);
    }
    return 0;
}

int cq3_aff3_debug_dot_x16_v17(
        const void * weights_pointer,
        const int8_t * activation,
        int32_t activation_sum,
        int32_t output[16]) {
    if (weights_pointer == nullptr || activation == nullptr || output == nullptr) {
        return 1;
    }
    const auto * weight = static_cast<const Aff3X16Group *>(weights_pointer);
    __m256i dot0 = _mm256_setzero_si256();
    __m256i dot1 = _mm256_setzero_si256();
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        __m256i values0;
        __m256i values1;
        __m256i values2;
        __m256i values3;
        load_activation_words(activation, chunk, &values0, &values1, &values2, &values3);
        const uint8_t * packed = weight->qs + chunk * 64;
        dot0 = _mm256_add_epi32(dot0, chunk_dot(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed)),
            values0, values1, values2, values3));
        dot1 = _mm256_add_epi32(dot1, chunk_dot(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + 32)),
            values0, values1, values2, values3));
    }
    const __m256i sum = _mm256_set1_epi32(activation_sum);
    dot0 = _mm256_sub_epi32(dot0, sum);
    dot1 = _mm256_sub_epi32(dot1, sum);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + 0), dot0);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + 8), dot1);
    return 0;
}

int cq3_aff3_debug_decode_scales_x16_v17(
        const uint8_t combined_indices[16],
        const uint16_t alpha_anchors[16],
        const uint16_t mu_anchors[16],
        const uint16_t alpha_codebook[16],
        const uint16_t mu_codebook[16],
        float output_alpha[16],
        float output_mu[16]) {
    if (combined_indices == nullptr || alpha_anchors == nullptr || mu_anchors == nullptr ||
            alpha_codebook == nullptr || mu_codebook == nullptr || output_alpha == nullptr ||
            output_mu == nullptr) {
        return 1;
    }
    float alpha_ratios[16];
    float mu_ratios[16];
    decode_ratio_tables(alpha_codebook, mu_codebook, alpha_ratios, mu_ratios);
    __m256 alpha_ratio0;
    __m256 alpha_ratio1;
    __m256 mu_ratio0;
    __m256 mu_ratio1;
    split_scale_indices(combined_indices,
        &alpha_ratio0, &alpha_ratio1, &mu_ratio0, &mu_ratio1,
        alpha_ratios, mu_ratios);
    _mm256_storeu_ps(output_alpha + 0, _mm256_mul_ps(alpha_ratio0,
        _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(alpha_anchors + 0)))));
    _mm256_storeu_ps(output_alpha + 8, _mm256_mul_ps(alpha_ratio1,
        _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(alpha_anchors + 8)))));
    _mm256_storeu_ps(output_mu + 0, _mm256_mul_ps(mu_ratio0,
        _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(mu_anchors + 0)))));
    _mm256_storeu_ps(output_mu + 8, _mm256_mul_ps(mu_ratio1,
        _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(mu_anchors + 8)))));
    return 0;
}

}  // extern "C"
