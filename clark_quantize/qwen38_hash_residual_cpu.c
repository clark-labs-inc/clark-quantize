#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_HASH_X86 1
#include <immintrin.h>
#else
#define CQ3_HASH_X86 0
#endif

enum {
    CQ3_HASH_OK = 0,
    CQ3_HASH_ERROR_ARGUMENT = 1,
    CQ3_HASH_ERROR_SHAPE = 2,
    CQ3_HASH_ERROR_CODE = 3,
    CQ3_HASH_ERROR_NONFINITE = 4,
    CQ3_HASH_ERROR_ISA = 5,
};

enum {
    CQ3_HASH_FEATURE_AVX2 = 1U,
    CQ3_HASH_FEATURE_FMA = 2U,
};

uint32_t cq3_hash_cpu_features(void) {
    uint32_t features = 0U;
#if CQ3_HASH_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    features |= __builtin_cpu_supports("avx2") ? CQ3_HASH_FEATURE_AVX2 : 0U;
    features |= __builtin_cpu_supports("fma") ? CQ3_HASH_FEATURE_FMA : 0U;
#endif
    return features;
}

static float cq3_hash_half_to_float(uint16_t value) {
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

static int8_t cq3_hash_decode_int4(uint8_t code) {
    return (int8_t)(code < 8U ? code : (int32_t)code - 16);
}

int cq3_hash_projection_f32_scalar_trusted(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    float *sketch
) {
    memset(sketch, 0, (size_t)sketch_dim * sizeof(float));
    for (uint32_t repetition = 0U; repetition < hashes_per_input; ++repetition) {
        const uint64_t offset = (uint64_t)repetition * input_dim;
        for (uint64_t index = 0U; index < input_dim; ++index) {
            sketch[buckets[offset + index]] +=
                input[index] * (float)signs[offset + index] * normalization;
        }
    }
    return CQ3_HASH_OK;
}

int cq3_hash_projection_f32_scalar(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    float *sketch
) {
    if (input == NULL || buckets == NULL || signs == NULL || sketch == NULL) {
        return CQ3_HASH_ERROR_ARGUMENT;
    }
    if (input_dim == 0U || sketch_dim == 0U || sketch_dim > UINT16_MAX ||
        hashes_per_input == 0U || hashes_per_input > 32U ||
        input_dim > UINT64_MAX / hashes_per_input) {
        return CQ3_HASH_ERROR_SHAPE;
    }
    if (!isfinite(normalization) || normalization <= 0.0f) {
        return CQ3_HASH_ERROR_NONFINITE;
    }
    const uint64_t mapping_count = input_dim * hashes_per_input;
    for (uint64_t index = 0U; index < input_dim; ++index) {
        if (!isfinite(input[index])) {
            return CQ3_HASH_ERROR_NONFINITE;
        }
    }
    for (uint64_t index = 0U; index < mapping_count; ++index) {
        if (buckets[index] >= sketch_dim) {
            return CQ3_HASH_ERROR_SHAPE;
        }
        if (signs[index] != -1 && signs[index] != 1) {
            return CQ3_HASH_ERROR_CODE;
        }
    }
    return cq3_hash_projection_f32_scalar_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, sketch
    );
}

static int cq3_hash_validate_int4_gemv(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    if (packed_codes == NULL || row_scales == NULL || sketch == NULL || output == NULL) {
        return CQ3_HASH_ERROR_ARGUMENT;
    }
    if (rows == 0U || sketch_dim == 0U ||
        rows > UINT64_MAX / (((uint64_t)sketch_dim + 1U) / 2U)) {
        return CQ3_HASH_ERROR_SHAPE;
    }
    for (uint32_t column = 0U; column < sketch_dim; ++column) {
        if (!isfinite(sketch[column])) {
            return CQ3_HASH_ERROR_NONFINITE;
        }
    }
    const uint64_t row_bytes = ((uint64_t)sketch_dim + 1U) / 2U;
    for (uint64_t row = 0U; row < rows; ++row) {
        if (!isfinite(cq3_hash_half_to_float(row_scales[row]))) {
            return CQ3_HASH_ERROR_NONFINITE;
        }
        const uint8_t *codes = packed_codes + row * row_bytes;
        for (uint32_t column = 0U; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t code = (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
            if (code == 8U) {
                return CQ3_HASH_ERROR_CODE;
            }
        }
        if ((sketch_dim & 1U) != 0U && (codes[row_bytes - 1U] & 0xf0U) != 0U) {
            return CQ3_HASH_ERROR_CODE;
        }
    }
    return CQ3_HASH_OK;
}

int cq3_hash_int4_gemv_f32_scalar_trusted(
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
            const uint8_t nibble =
                (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
            total += (float)cq3_hash_decode_int4(nibble) * sketch[column];
        }
        output[row] = total * cq3_hash_half_to_float(row_scales[row]);
    }
    return CQ3_HASH_OK;
}

int cq3_hash_int4_gemv_f32_scalar(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    const int status = cq3_hash_validate_int4_gemv(
        packed_codes, row_scales, sketch, rows, sketch_dim, output
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    return cq3_hash_int4_gemv_f32_scalar_trusted(
        packed_codes, row_scales, sketch, rows, sketch_dim, output
    );
}

int cq3_hash_residual_f32_scalar_trusted(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    cq3_hash_projection_f32_scalar_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, sketch_scratch
    );
    const uint64_t row_bytes = ((uint64_t)sketch_dim + 1U) / 2U;
    for (uint64_t row = 0U; row < rows; ++row) {
        const uint8_t *codes = packed_codes + row * row_bytes;
        float total = 0.0f;
        for (uint32_t column = 0U; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t nibble =
                (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
            total += (float)cq3_hash_decode_int4(nibble) * sketch_scratch[column];
        }
        const float correction = total * cq3_hash_half_to_float(row_scales[row]);
        output[row] = add_to_output != 0U ? output[row] + correction : correction;
    }
    return CQ3_HASH_OK;
}

int cq3_hash_residual_f32_scalar(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    if (add_to_output > 1U) {
        return CQ3_HASH_ERROR_ARGUMENT;
    }
    int status = cq3_hash_projection_f32_scalar(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, sketch_scratch
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    status = cq3_hash_validate_int4_gemv(
        packed_codes, row_scales, sketch_scratch, rows, sketch_dim, output
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    if (add_to_output != 0U) {
        for (uint64_t row = 0U; row < rows; ++row) {
            if (!isfinite(output[row])) {
                return CQ3_HASH_ERROR_NONFINITE;
            }
        }
    }
    return cq3_hash_residual_f32_scalar_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, packed_codes, row_scales, rows, sketch_scratch,
        output, add_to_output
    );
}

#if CQ3_HASH_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static float cq3_hash_sum8(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
static int cq3_hash_int4_gemv_f32_avx2_impl(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output,
    uint32_t add_to_output
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
            const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
            __m128i lanes = _mm_unpacklo_epi8(low, high);
            lanes = _mm_sub_epi8(_mm_xor_si128(lanes, sign_bit), sign_bit);
            const __m256 code0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lanes));
            const __m128i lanes_high = _mm_srli_si128(lanes, 8);
            const __m256 code1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lanes_high));
            sum0 = _mm256_fmadd_ps(code0, _mm256_loadu_ps(sketch + column), sum0);
            sum1 = _mm256_fmadd_ps(code1, _mm256_loadu_ps(sketch + column + 8U), sum1);
        }
        float total = cq3_hash_sum8(sum0) + cq3_hash_sum8(sum1);
        for (; column < sketch_dim; ++column) {
            const uint8_t byte = codes[column >> 1U];
            const uint8_t nibble =
                (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
            total += (float)cq3_hash_decode_int4(nibble) * sketch[column];
        }
        const float correction = total * cq3_hash_half_to_float(row_scales[row]);
        output[row] = add_to_output != 0U ? output[row] + correction : correction;
    }
    return CQ3_HASH_OK;
}

__attribute__((target("avx2,fma")))
int cq3_hash_int4_gemv_f32_avx2_trusted(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    return cq3_hash_int4_gemv_f32_avx2_impl(
        packed_codes, row_scales, sketch, rows, sketch_dim, output, 0U
    );
}

__attribute__((target("avx2,fma")))
int cq3_hash_int4_gemv_f32_avx2(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    const int status = cq3_hash_validate_int4_gemv(
        packed_codes, row_scales, sketch, rows, sketch_dim, output
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    return cq3_hash_int4_gemv_f32_avx2_trusted(
        packed_codes, row_scales, sketch, rows, sketch_dim, output
    );
}

__attribute__((target("avx2,fma")))
int cq3_hash_residual_f32_avx2_trusted(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    cq3_hash_projection_f32_scalar_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, sketch_scratch
    );
    return cq3_hash_int4_gemv_f32_avx2_impl(
        packed_codes, row_scales, sketch_scratch, rows, sketch_dim, output,
        add_to_output
    );
}

__attribute__((target("avx2,fma")))
int cq3_hash_residual_f32_avx2(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    if (add_to_output > 1U) {
        return CQ3_HASH_ERROR_ARGUMENT;
    }
    int status = cq3_hash_projection_f32_scalar(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, sketch_scratch
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    if (add_to_output != 0U) {
        for (uint64_t row = 0U; row < rows; ++row) {
            if (!isfinite(output[row])) {
                return CQ3_HASH_ERROR_NONFINITE;
            }
        }
    }
    status = cq3_hash_validate_int4_gemv(
        packed_codes, row_scales, sketch_scratch, rows, sketch_dim, output
    );
    if (status != CQ3_HASH_OK) {
        return status;
    }
    return cq3_hash_residual_f32_avx2_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, packed_codes, row_scales, rows, sketch_scratch,
        output, add_to_output
    );
}
#else
int cq3_hash_int4_gemv_f32_avx2_trusted(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    (void)packed_codes;
    (void)row_scales;
    (void)sketch;
    (void)rows;
    (void)sketch_dim;
    (void)output;
    return CQ3_HASH_ERROR_ISA;
}

int cq3_hash_int4_gemv_f32_avx2(
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    const float *sketch,
    uint64_t rows,
    uint32_t sketch_dim,
    float *output
) {
    return cq3_hash_int4_gemv_f32_avx2_trusted(
        packed_codes, row_scales, sketch, rows, sketch_dim, output
    );
}


int cq3_hash_residual_f32_avx2_trusted(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    (void)input;
    (void)buckets;
    (void)signs;
    (void)input_dim;
    (void)sketch_dim;
    (void)hashes_per_input;
    (void)normalization;
    (void)packed_codes;
    (void)row_scales;
    (void)rows;
    (void)sketch_scratch;
    (void)output;
    (void)add_to_output;
    return CQ3_HASH_ERROR_ISA;
}

int cq3_hash_residual_f32_avx2(
    const float *input,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t input_dim,
    uint32_t sketch_dim,
    uint32_t hashes_per_input,
    float normalization,
    const uint8_t *packed_codes,
    const uint16_t *row_scales,
    uint64_t rows,
    float *sketch_scratch,
    float *output,
    uint32_t add_to_output
) {
    return cq3_hash_residual_f32_avx2_trusted(
        input, buckets, signs, input_dim, sketch_dim, hashes_per_input,
        normalization, packed_codes, row_scales, rows, sketch_scratch,
        output, add_to_output
    );
}
#endif
