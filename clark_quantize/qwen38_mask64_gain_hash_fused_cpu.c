#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_MGHF_X86 1
#include <immintrin.h>
#else
#define CQ3_MGHF_X86 0
#endif

enum {
    CQ3_MGHF_OK = 0,
    CQ3_MGHF_ERROR_ARGUMENT = 1,
    CQ3_MGHF_ERROR_SHAPE = 2,
    CQ3_MGHF_ERROR_CODE = 3,
    CQ3_MGHF_ERROR_NONFINITE = 4,
    CQ3_MGHF_ERROR_ISA = 5,
};

extern int cq3_hsm64_gemv_q8_row_scale4_scalar(
    const uint8_t *, uint64_t, const uint8_t *, const uint16_t *,
    const uint16_t *, const int8_t *, const float *, const int32_t *,
    uint64_t, uint64_t, float *
);
extern int cq3_hsm64_gemv_q8_row_scale4_scalar_trusted(
    const uint8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
    const int8_t *, const float *, const int32_t *, uint64_t, uint64_t, float *
);
extern int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2(
    const uint8_t *, uint64_t, const uint8_t *, const uint16_t *,
    const uint16_t *, const int8_t *, const float *, const int32_t *,
    uint64_t, uint64_t, float *
);
extern int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
    const int8_t *, const float *, const int32_t *, uint64_t, uint64_t, float *
);

static float cq3_mghf_half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            exponent = 127U - 15U + 1U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                exponent -= 1U;
            }
            mantissa &= 0x03ffU;
            bits = sign | (exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) |
            (mantissa << 13U);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static int8_t cq3_mghf_decode_int4(uint8_t code) {
    return (int8_t)(code < 8U ? code : (int32_t)code - 16);
}

static int cq3_mghf_validate_correction(
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint64_t input_dim,
    uint32_t sketch_dim,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output
) {
    if (buckets == NULL || signs == NULL || gain_codes == NULL ||
        gain_codebook == NULL || packed_codes == NULL || row_scales == NULL ||
        sketch_scratch == NULL || output == NULL) {
        return CQ3_MGHF_ERROR_ARGUMENT;
    }
    if (input_dim == 0U || input_dim % 128U != 0U || sketch_dim == 0U ||
        sketch_dim > UINT16_MAX || rows == 0U) {
        return CQ3_MGHF_ERROR_SHAPE;
    }
    const uint64_t row_bytes = ((uint64_t)sketch_dim + 1U) / 2U;
    if (rows > UINT64_MAX / row_bytes) {
        return CQ3_MGHF_ERROR_SHAPE;
    }
    for (uint32_t code = 0U; code < 8U; ++code) {
        if (!isfinite(gain_codebook[code]) || gain_codebook[code] < 0.0f) {
            return CQ3_MGHF_ERROR_NONFINITE;
        }
    }
    for (uint64_t index = 0U; index < input_dim; ++index) {
        if (buckets[index] >= sketch_dim) {
            return CQ3_MGHF_ERROR_SHAPE;
        }
        if (signs[index] != -1 && signs[index] != 1) {
            return CQ3_MGHF_ERROR_CODE;
        }
        if (gain_codes[index] >= 8U) {
            return CQ3_MGHF_ERROR_CODE;
        }
    }
    for (uint64_t row = 0U; row < rows; ++row) {
        if (!isfinite(cq3_mghf_half_to_float(row_scales[row]))) {
            return CQ3_MGHF_ERROR_NONFINITE;
        }
        const uint8_t *codes = packed_codes + row * row_bytes;
        for (uint32_t column = 0U; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t nibble = (column & 1U) == 0U
                ? byte & 0x0fU
                : byte >> 4U;
            if (nibble == 8U) {
                return CQ3_MGHF_ERROR_CODE;
            }
        }
        if ((sketch_dim & 1U) != 0U && (codes[row_bytes - 1U] & 0xf0U) != 0U) {
            return CQ3_MGHF_ERROR_CODE;
        }
    }
    return CQ3_MGHF_OK;
}

int cq3_mghf_projection_q8_scalar_trusted(
    const int8_t *activation,
    const float *activation_scales,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint64_t input_dim,
    uint32_t sketch_dim,
    float *sketch
) {
    memset(sketch, 0, (size_t)sketch_dim * sizeof(float));
    const uint64_t groups = input_dim / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        const float scale = activation_scales[group];
        const uint64_t start = group * 128U;
        for (uint32_t lane = 0U; lane < 128U; ++lane) {
            const uint64_t index = start + lane;
            sketch[buckets[index]] += (float)activation[index] *
                (float)signs[index] * gain_codebook[gain_codes[index]] * scale;
        }
    }
    return CQ3_MGHF_OK;
}

static int cq3_mghf_int4_add_scalar(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    const uint64_t row_bytes = ((uint64_t)sketch_dim + 1U) / 2U;
    for (uint64_t row = 0U; row < rows; ++row) {
        const uint8_t *codes = packed_codes + row * row_bytes;
        float total = 0.0f;
        for (uint32_t column = 0U; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t nibble = (column & 1U) == 0U
                ? byte & 0x0fU
                : byte >> 4U;
            total += (float)cq3_mghf_decode_int4(nibble) * sketch[column];
        }
        output[row] += total * cq3_mghf_half_to_float(row_scales[row]);
    }
    return CQ3_MGHF_OK;
}

#if CQ3_MGHF_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static float cq3_mghf_sum8(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
static int cq3_mghf_int4_add_avx2(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    const uint64_t row_bytes = ((uint64_t)sketch_dim + 1U) / 2U;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_bit = _mm_set1_epi8(8);
    for (uint64_t row = 0U; row < rows; ++row) {
        const uint8_t *codes = packed_codes + row * row_bytes;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        uint32_t column = 0U;
        for (; column + 16U <= sketch_dim; column += 16U) {
            const __m128i bytes = _mm_loadl_epi64(
                (const __m128i *)(codes + (column >> 1U))
            );
            const __m128i low = _mm_and_si128(bytes, nibble_mask);
            const __m128i high = _mm_and_si128(
                _mm_srli_epi16(bytes, 4), nibble_mask
            );
            __m128i lanes = _mm_unpacklo_epi8(low, high);
            lanes = _mm_sub_epi8(_mm_xor_si128(lanes, sign_bit), sign_bit);
            const __m256 code0 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(lanes)
            );
            const __m128i lanes_high = _mm_srli_si128(lanes, 8);
            const __m256 code1 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(lanes_high)
            );
            sum0 = _mm256_fmadd_ps(
                code0, _mm256_loadu_ps(sketch + column), sum0
            );
            sum1 = _mm256_fmadd_ps(
                code1, _mm256_loadu_ps(sketch + column + 8U), sum1
            );
        }
        float total = cq3_mghf_sum8(sum0) + cq3_mghf_sum8(sum1);
        for (; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t nibble = (column & 1U) == 0U
                ? byte & 0x0fU
                : byte >> 4U;
            total += (float)cq3_mghf_decode_int4(nibble) * sketch[column];
        }
        output[row] += total * cq3_mghf_half_to_float(row_scales[row]);
    }
    return CQ3_MGHF_OK;
}
#endif

int cq3_mghf_gemv_q8_row_scale4_scalar_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *output
) {
    int status = cq3_hsm64_gemv_q8_row_scale4_scalar_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
    cq3_mghf_projection_q8_scalar_trusted(
        activation, activation_scales, buckets, signs, gain_codes,
        gain_codebook, columns, sketch_dim, sketch_scratch
    );
    return cq3_mghf_int4_add_scalar(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, output
    );
}

int cq3_mghf_gemv_q8_row_scale4_scalar(
    const uint8_t *weights,
    uint64_t weight_bytes,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *output
) {
    int status = cq3_mghf_validate_correction(
        buckets, signs, gain_codes, gain_codebook, columns, sketch_dim,
        correction_codes, correction_row_scales, rows, sketch_scratch, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
    status = cq3_hsm64_gemv_q8_row_scale4_scalar(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
    cq3_mghf_projection_q8_scalar_trusted(
        activation, activation_scales, buckets, signs, gain_codes,
        gain_codebook, columns, sketch_dim, sketch_scratch
    );
    return cq3_mghf_int4_add_scalar(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, output
    );
}

int cq3_mghf_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *output
) {
#if CQ3_MGHF_X86 && (defined(__GNUC__) || defined(__clang__))
    int status = cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
    cq3_mghf_projection_q8_scalar_trusted(
        activation, activation_scales, buckets, signs, gain_codes,
        gain_codebook, columns, sketch_dim, sketch_scratch
    );
    return cq3_mghf_int4_add_avx2(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, output
    );
#else
    (void)weights;
    (void)packed_scale_indices;
    (void)row_anchors;
    (void)ratio_codebook;
    (void)activation;
    (void)activation_scales;
    (void)activation_sums;
    (void)rows;
    (void)columns;
    (void)buckets;
    (void)signs;
    (void)gain_codes;
    (void)gain_codebook;
    (void)sketch_dim;
    (void)correction_codes;
    (void)correction_row_scales;
    (void)sketch_scratch;
    (void)output;
    return CQ3_MGHF_ERROR_ISA;
#endif
}

int cq3_mghf_gemv_q8_row_scale4_avx2_bmi2(
    const uint8_t *weights,
    uint64_t weight_bytes,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *gain_codes,
    const float *gain_codebook,
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *output
) {
    int status = cq3_mghf_validate_correction(
        buckets, signs, gain_codes, gain_codebook, columns, sketch_dim,
        correction_codes, correction_row_scales, rows, sketch_scratch, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
    status = cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_MGHF_OK) {
        return status;
    }
#if CQ3_MGHF_X86 && (defined(__GNUC__) || defined(__clang__))
    cq3_mghf_projection_q8_scalar_trusted(
        activation, activation_scales, buckets, signs, gain_codes,
        gain_codebook, columns, sketch_dim, sketch_scratch
    );
    return cq3_mghf_int4_add_avx2(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, output
    );
#else
    return CQ3_MGHF_ERROR_ISA;
#endif
}
