// Register-codebook AFF3_X16 research kernel for Qwen/Qwen3.8-27B on AVX2.
//
// This translation unit reuses the frozen v17 layout and exact dot machinery,
// but its v18 entry point replaces every per-group FP32 gather with FP16 byte
// shuffles.  The lookup tables remain in registers, add zero persistent or
// additional runtime compute-image bytes, and never materialize weights or a
// decoded per-group scale matrix.

#include "qwen38_affine_avx2_x16_fused_v17.cpp"

namespace {

struct Fp16ByteLookupV18 {
    __m128i low;
    __m128i high;
};

inline Fp16ByteLookupV18 make_fp16_lookup_v18(const uint16_t * codebook) {
    alignas(16) uint8_t low[16];
    alignas(16) uint8_t high[16];
    for (int index = 0; index < 16; ++index) {
        low[index] = static_cast<uint8_t>(codebook[index]);
        high[index] = static_cast<uint8_t>(codebook[index] >> 8U);
    }
    return {
        _mm_load_si128(reinterpret_cast<const __m128i *>(low)),
        _mm_load_si128(reinterpret_cast<const __m128i *>(high)),
    };
}

inline void lookup_fp16x16_v18(
        const Fp16ByteLookupV18 & table,
        __m128i indices,
        __m256 * output0,
        __m256 * output1) {
    const __m128i low = _mm_shuffle_epi8(table.low, indices);
    const __m128i high = _mm_shuffle_epi8(table.high, indices);
    const __m128i half0 = _mm_unpacklo_epi8(low, high);
    const __m128i half1 = _mm_unpackhi_epi8(low, high);
    *output0 = _mm256_cvtph_ps(half0);
    *output1 = _mm256_cvtph_ps(half1);
}

inline void split_scale_indices_register_v18(
        const uint8_t * combined,
        const Fp16ByteLookupV18 & alpha_table,
        const Fp16ByteLookupV18 & mu_table,
        __m256 * alpha_ratio0,
        __m256 * alpha_ratio1,
        __m256 * mu_ratio0,
        __m256 * mu_ratio1) {
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(combined));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i alpha = _mm_and_si128(packed, nibble_mask);
    const __m128i mu = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
    lookup_fp16x16_v18(alpha_table, alpha, alpha_ratio0, alpha_ratio1);
    lookup_fp16x16_v18(mu_table, mu, mu_ratio0, mu_ratio1);
}

}  // namespace

extern "C" {

int cq3_aff3_gemv_x16_register_v18(
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
    const Fp16ByteLookupV18 alpha_table = make_fp16_lookup_v18(alpha_codebook);
    const Fp16ByteLookupV18 mu_table = make_fp16_lookup_v18(mu_codebook);
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
            split_scale_indices_register_v18(weight.scale_indices,
                alpha_table, mu_table,
                &alpha_ratio0, &alpha_ratio1, &mu_ratio0, &mu_ratio1);
            const __m256 alpha0 = _mm256_mul_ps(alpha_anchor0, alpha_ratio0);
            const __m256 alpha1 = _mm256_mul_ps(alpha_anchor1, alpha_ratio1);
            const __m256 mu0 = _mm256_mul_ps(mu_anchor0, mu_ratio0);
            const __m256 mu1 = _mm256_mul_ps(mu_anchor1, mu_ratio1);
            const __m256 sumf = _mm256_set1_ps(
                static_cast<float>(activation_sums[group]));
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

int cq3_aff3_debug_lookup_v18(
        const uint16_t codebook[16],
        const uint8_t indices[16],
        float output[16]) {
    if (codebook == nullptr || indices == nullptr || output == nullptr) {
        return 1;
    }
    const Fp16ByteLookupV18 table = make_fp16_lookup_v18(codebook);
    __m256 output0;
    __m256 output1;
    lookup_fp16x16_v18(table,
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(indices)),
        &output0, &output1);
    _mm256_storeu_ps(output + 0, output0);
    _mm256_storeu_ps(output + 8, output1);
    return 0;
}

}  // extern "C"
