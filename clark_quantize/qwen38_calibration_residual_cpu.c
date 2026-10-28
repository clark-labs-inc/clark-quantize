#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_CALIBRATION_X86 1
#include <immintrin.h>
#else
#define CQ3_CALIBRATION_X86 0
#endif

enum {
    CQ3_CALIBRATION_OK = 0,
    CQ3_CALIBRATION_ERROR_ARGUMENT = 1,
    CQ3_CALIBRATION_ERROR_SHAPE = 2,
    CQ3_CALIBRATION_ERROR_NONFINITE = 3,
    CQ3_CALIBRATION_ERROR_ISA = 4,
};

enum {
    CQ3_CALIBRATION_FEATURE_AVX2 = 1U,
    CQ3_CALIBRATION_FEATURE_FMA = 2U,
    CQ3_CALIBRATION_FEATURE_F16C = 4U,
};

uint32_t cq3_calibration_residual_cpu_features(void) {
    uint32_t features = 0U;
#if CQ3_CALIBRATION_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    features |= __builtin_cpu_supports("avx2") ? CQ3_CALIBRATION_FEATURE_AVX2 : 0U;
    features |= __builtin_cpu_supports("fma") ? CQ3_CALIBRATION_FEATURE_FMA : 0U;
    features |= __builtin_cpu_supports("f16c") ? CQ3_CALIBRATION_FEATURE_F16C : 0U;
#endif
    return features;
}

static float cq3_calibration_half_to_float(uint16_t value) {
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
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static int cq3_calibration_validate(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    if (input == NULL || left_codes == NULL || left_scales == NULL ||
        right_codes == NULL || right_scales == NULL || bias == NULL ||
        rank_scratch == NULL || output == NULL) {
        return CQ3_CALIBRATION_ERROR_ARGUMENT;
    }
    if (input_dim == 0U || output_dim == 0U || rank != 16U ||
        output_dim % 8U != 0U ||
        input_dim > UINT32_MAX / rank || output_dim > UINT32_MAX / rank ||
        add_to_output > 1U) {
        return CQ3_CALIBRATION_ERROR_SHAPE;
    }
    for (uint32_t index = 0U; index < input_dim; ++index) {
        if (!isfinite(input[index])) {
            return CQ3_CALIBRATION_ERROR_NONFINITE;
        }
    }
    for (uint32_t index = 0U; index < rank; ++index) {
        if (!isfinite(cq3_calibration_half_to_float(left_scales[index])) ||
            !isfinite(cq3_calibration_half_to_float(right_scales[index]))) {
            return CQ3_CALIBRATION_ERROR_NONFINITE;
        }
    }
    for (uint32_t index = 0U; index < output_dim; ++index) {
        if (!isfinite(cq3_calibration_half_to_float(bias[index])) ||
            (add_to_output != 0U && !isfinite(output[index]))) {
            return CQ3_CALIBRATION_ERROR_NONFINITE;
        }
    }
    return CQ3_CALIBRATION_OK;
}

int cq3_calibration_residual_f32_scalar_trusted(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    for (uint32_t component = 0U; component < rank; ++component) {
        float total = 0.0f;
        for (uint32_t index = 0U; index < input_dim; ++index) {
            total += input[index] * (float)left_codes[(size_t)index * rank + component];
        }
        rank_scratch[component] = total *
            cq3_calibration_half_to_float(left_scales[component]) *
            cq3_calibration_half_to_float(right_scales[component]);
    }
    for (uint32_t column = 0U; column < output_dim; ++column) {
        float correction = cq3_calibration_half_to_float(bias[column]);
        for (uint32_t component = 0U; component < rank; ++component) {
            correction += rank_scratch[component] *
                (float)right_codes[(size_t)component * output_dim + column];
        }
        output[column] = add_to_output != 0U ? output[column] + correction : correction;
    }
    return CQ3_CALIBRATION_OK;
}

int cq3_calibration_residual_f32_scalar(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    const int status = cq3_calibration_validate(
        input, left_codes, left_scales, right_codes, right_scales, bias,
        input_dim, output_dim, rank, rank_scratch, output, add_to_output
    );
    if (status != CQ3_CALIBRATION_OK) {
        return status;
    }
    return cq3_calibration_residual_f32_scalar_trusted(
        input, left_codes, left_scales, right_codes, right_scales, bias,
        input_dim, output_dim, rank, rank_scratch, output, add_to_output
    );
}

#if CQ3_CALIBRATION_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma,f16c")))
int cq3_calibration_residual_f32_avx2_trusted(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    (void)rank;
    __m256 projected_low = _mm256_setzero_ps();
    __m256 projected_high = _mm256_setzero_ps();
    for (uint32_t index = 0U; index < input_dim; ++index) {
        const __m128i packed = _mm_loadu_si128(
            (const __m128i *)(left_codes + (size_t)index * 16U)
        );
        const __m256 values = _mm256_set1_ps(input[index]);
        const __m256 low = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
        const __m128i packed_high = _mm_srli_si128(packed, 8);
        const __m256 high = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed_high));
        projected_low = _mm256_fmadd_ps(values, low, projected_low);
        projected_high = _mm256_fmadd_ps(values, high, projected_high);
    }
    const __m256 left_low = _mm256_cvtph_ps(
        _mm_loadu_si128((const __m128i *)left_scales)
    );
    const __m256 left_high = _mm256_cvtph_ps(
        _mm_loadu_si128((const __m128i *)(left_scales + 8U))
    );
    const __m256 right_low = _mm256_cvtph_ps(
        _mm_loadu_si128((const __m128i *)right_scales)
    );
    const __m256 right_high = _mm256_cvtph_ps(
        _mm_loadu_si128((const __m128i *)(right_scales + 8U))
    );
    projected_low = _mm256_mul_ps(_mm256_mul_ps(projected_low, left_low), right_low);
    projected_high = _mm256_mul_ps(
        _mm256_mul_ps(projected_high, left_high), right_high
    );
    _mm256_storeu_ps(rank_scratch, projected_low);
    _mm256_storeu_ps(rank_scratch + 8U, projected_high);

    for (uint32_t column = 0U; column < output_dim; column += 8U) {
        __m256 correction = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)(bias + column))
        );
        for (uint32_t component = 0U; component < 16U; ++component) {
            const __m128i packed = _mm_loadl_epi64(
                (const __m128i *)(right_codes +
                    (size_t)component * output_dim + column)
            );
            const __m256 codes = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
            correction = _mm256_fmadd_ps(
                _mm256_set1_ps(rank_scratch[component]), codes, correction
            );
        }
        if (add_to_output != 0U) {
            correction = _mm256_add_ps(
                _mm256_loadu_ps(output + column), correction
            );
        }
        _mm256_storeu_ps(output + column, correction);
    }
    return CQ3_CALIBRATION_OK;
}
#else
int cq3_calibration_residual_f32_avx2_trusted(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    (void)input;
    (void)left_codes;
    (void)left_scales;
    (void)right_codes;
    (void)right_scales;
    (void)bias;
    (void)input_dim;
    (void)output_dim;
    (void)rank;
    (void)rank_scratch;
    (void)output;
    (void)add_to_output;
    return CQ3_CALIBRATION_ERROR_ISA;
}
#endif

int cq3_calibration_residual_f32_avx2(
    const float *input,
    const int8_t *left_codes,
    const uint16_t *left_scales,
    const int8_t *right_codes,
    const uint16_t *right_scales,
    const uint16_t *bias,
    uint32_t input_dim,
    uint32_t output_dim,
    uint32_t rank,
    float *rank_scratch,
    float *output,
    uint32_t add_to_output
) {
    const int status = cq3_calibration_validate(
        input, left_codes, left_scales, right_codes, right_scales, bias,
        input_dim, output_dim, rank, rank_scratch, output, add_to_output
    );
    if (status != CQ3_CALIBRATION_OK) {
        return status;
    }
    const uint32_t required = CQ3_CALIBRATION_FEATURE_AVX2 |
        CQ3_CALIBRATION_FEATURE_FMA | CQ3_CALIBRATION_FEATURE_F16C;
    if ((cq3_calibration_residual_cpu_features() & required) != required) {
        return CQ3_CALIBRATION_ERROR_ISA;
    }
    return cq3_calibration_residual_f32_avx2_trusted(
        input, left_codes, left_scales, right_codes, right_scales, bias,
        input_dim, output_dim, rank, rank_scratch, output, add_to_output
    );
}
