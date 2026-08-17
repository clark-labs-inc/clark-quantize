#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CQ3_X86 1
#include <immintrin.h>
#else
#define CQ3_X86 0
#endif

#define CQ3_MAX_GROUPS_PER_ROW 256U
#define CQ3_RADIX5_BYTES_PER_GROUP 26U
#define CQ3_RADIX5_SEGMENTS_PADDED 28U
#define CQ3_RADIX5_ACTIVATION_STRIDE (CQ3_RADIX5_SEGMENTS_PADDED * 8U)

#if defined(__GNUC__) || defined(__clang__)
#define CQ3_RESTRICT __restrict__
#else
#define CQ3_RESTRICT
#endif

enum {
    CQ3_OK = 0,
    CQ3_ERROR_ARGUMENT = 1,
    CQ3_ERROR_SHAPE = 2,
    CQ3_ERROR_CODE = 3,
    CQ3_ERROR_NONFINITE = 4,
    CQ3_ERROR_ISA = 5,
};

enum {
    CQ3_FEATURE_AVX2 = 1,
    CQ3_FEATURE_F16C = 2,
    CQ3_FEATURE_FMA = 4,
};

uint32_t cq3_cpu_features(void) {
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    uint32_t features = 0U;
    features |= __builtin_cpu_supports("avx2") ? CQ3_FEATURE_AVX2 : 0U;
    features |= __builtin_cpu_supports("f16c") ? CQ3_FEATURE_F16C : 0U;
    features |= __builtin_cpu_supports("fma") ? CQ3_FEATURE_FMA : 0U;
    return features;
#else
    return 0U;
#endif
}

static float cq3_half_to_float(uint16_t value) {
    const uint32_t sign = ((uint32_t)value & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)value >> 10U) & 0x1fU;
    uint32_t mantissa = (uint32_t)value & 0x03ffU;
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
        exponent += 127U - 15U;
        bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

int cq3_pack_aligned2(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if ((count != 0U && (codes == NULL || packed == NULL))) {
        return CQ3_ERROR_ARGUMENT;
    }
    const uint64_t packed_count = (count + 3U) / 4U;
    for (uint64_t byte_index = 0U; byte_index < packed_count; ++byte_index) {
        uint8_t value = 0U;
        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const uint64_t index = byte_index * 4U + lane;
            const int code = index < count ? (int)codes[index] : 0;
            if (code < -1 || code > 1) {
                return CQ3_ERROR_CODE;
            }
            value |= (uint8_t)(code + 1) << (lane * 2U);
        }
        packed[byte_index] = value;
    }
    return CQ3_OK;
}

int cq3_unpack_aligned2(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if ((count != 0U && (packed == NULL || codes == NULL))) {
        return CQ3_ERROR_ARGUMENT;
    }
    for (uint64_t index = 0U; index < count; ++index) {
        const uint8_t digit = (packed[index / 4U] >> ((index % 4U) * 2U)) & 3U;
        if (digit == 3U) {
            return CQ3_ERROR_CODE;
        }
        codes[index] = (int8_t)digit - 1;
    }
    const uint32_t tail = (uint32_t)(count % 4U);
    if (tail != 0U) {
        const uint8_t last = packed[count / 4U];
        for (uint32_t lane = tail; lane < 4U; ++lane) {
            if (((last >> (lane * 2U)) & 3U) != 1U) {
                return CQ3_ERROR_CODE;
            }
        }
    }
    return CQ3_OK;
}

int cq3_pack_group128_2bit(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if (codes == NULL || packed == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t lane = 0U; lane < 32U; ++lane) {
            uint8_t value = 0U;
            for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
                const int code = (int)codes[group * 128U + quarter * 32U + lane];
                if (code < -1 || code > 1) {
                    return CQ3_ERROR_CODE;
                }
                value |= (uint8_t)(code + 1) << (quarter * 2U);
            }
            packed[group * 32U + lane] = value;
        }
    }
    return CQ3_OK;
}

int cq3_unpack_group128_2bit(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if (packed == NULL || codes == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t lane = 0U; lane < 32U; ++lane) {
            const uint8_t value = packed[group * 32U + lane];
            for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
                const uint8_t digit = (value >> (quarter * 2U)) & 3U;
                if (digit == 3U) {
                    return CQ3_ERROR_CODE;
                }
                codes[group * 128U + quarter * 32U + lane] = (int8_t)digit - 1;
            }
        }
    }
    return CQ3_OK;
}

int cq3_pack_group128_radix5(
    const int8_t *codes,
    uint64_t count,
    uint8_t *packed
) {
    if (codes == NULL || packed == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
            uint32_t value = 0U;
            uint32_t power = 1U;
            for (uint32_t lane = 0U; lane < 5U; ++lane) {
                const uint32_t position = segment * 5U + lane;
                int code = -1;
                if (position < 128U) {
                    code = (int)codes[group * 128U + position];
                    if (code < -1 || code > 1) {
                        return CQ3_ERROR_CODE;
                    }
                }
                value += (uint32_t)(code + 1) * power;
                power *= 3U;
            }
            packed[group * CQ3_RADIX5_BYTES_PER_GROUP + segment] = (uint8_t)value;
        }
    }
    return CQ3_OK;
}

int cq3_unpack_group128_radix5(
    const uint8_t *packed,
    uint64_t count,
    int8_t *codes
) {
    if (packed == NULL || codes == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (count == 0U || count % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = count / 128U;
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
            uint32_t value = packed[group * CQ3_RADIX5_BYTES_PER_GROUP + segment];
            if (value > 242U) {
                return CQ3_ERROR_CODE;
            }
            for (uint32_t lane = 0U; lane < 5U; ++lane) {
                const uint32_t position = segment * 5U + lane;
                const uint32_t digit = value % 3U;
                value /= 3U;
                if (position < 128U) {
                    codes[group * 128U + position] = (int8_t)digit - 1;
                } else if (digit != 0U) {
                    return CQ3_ERROR_CODE;
                }
            }
        }
    }
    return CQ3_OK;
}

int cq3_quantize_f32_q8(
    const float *input,
    uint64_t columns,
    uint32_t group_size,
    int8_t *quantized,
    float *scales,
    int32_t *sums
) {
    if (input == NULL || quantized == NULL || scales == NULL || sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (group_size == 0U || columns == 0U || columns % group_size != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = columns / group_size;
    for (uint64_t group = 0U; group < groups; ++group) {
        const uint64_t offset = group * group_size;
        float maximum = 0.0f;
        for (uint32_t lane = 0U; lane < group_size; ++lane) {
            const float value = input[offset + lane];
            if (!isfinite(value)) {
                return CQ3_ERROR_NONFINITE;
            }
            const float absolute = fabsf(value);
            maximum = absolute > maximum ? absolute : maximum;
        }
        const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        int32_t sum = 0;
        for (uint32_t lane = 0U; lane < group_size; ++lane) {
            long rounded = lrintf(input[offset + lane] / scale);
            rounded = rounded < -127L ? -127L : rounded;
            rounded = rounded > 127L ? 127L : rounded;
            quantized[offset + lane] = (int8_t)rounded;
            sum += (int32_t)rounded;
        }
        scales[group] = scale;
        sums[group] = sum;
    }
    return CQ3_OK;
}

int cq3_expand_q8_radix5(
    const int8_t *activation,
    uint64_t columns,
    int8_t *expanded
) {
    if (activation == NULL || expanded == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (columns == 0U || columns % 128U != 0U) {
        return CQ3_ERROR_SHAPE;
    }
    const uint64_t groups = columns / 128U;
    memset(expanded, 0, groups * CQ3_RADIX5_ACTIVATION_STRIDE);
    for (uint64_t group = 0U; group < groups; ++group) {
        for (uint32_t position = 0U; position < 128U; ++position) {
            const uint32_t segment = position / 5U;
            const uint32_t lane = position % 5U;
            expanded[group * CQ3_RADIX5_ACTIVATION_STRIDE + segment * 8U + lane] =
                activation[group * 128U + position];
        }
    }
    return CQ3_OK;
}

static int cq3_validate_gemv(
    const uint8_t *weights,
    const uint16_t *weight_scales,
    const int8_t *activation,
    const float *activation_scales,
    float *output,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size
) {
    if (weights == NULL || weight_scales == NULL || activation == NULL ||
        activation_scales == NULL || output == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || group_size != 128U ||
        columns % group_size != 0U || columns % 4U != 0U ||
        columns / group_size > CQ3_MAX_GROUPS_PER_ROW) {
        return CQ3_ERROR_SHAPE;
    }
    return CQ3_OK;
}

int cq3_gemv_q8_group128_2bit_scalar(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = columns / 4U;
    for (uint64_t row = 0U; row < rows; ++row) {
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint64_t code_offset = row * row_bytes + group * (group_size / 4U);
            const uint64_t activation_offset = group * group_size;
            int32_t dot = 0;
            for (uint32_t lane = 0U; lane < group_size; ++lane) {
                const uint8_t digit =
                    (weights[code_offset + lane % 32U] >> ((lane / 32U) * 2U)) & 3U;
                if (digit == 3U) {
                    return CQ3_ERROR_CODE;
                }
                dot += ((int32_t)digit - 1) * (int32_t)activation[activation_offset + lane];
            }
            const float weight_scale = cq3_half_to_float(weight_scales[row * groups + group]);
            total += (float)dot * weight_scale * activation_scales[group];
        }
        output[row] = total;
    }
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_2bit_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = columns / 4U;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint64_t row = 0U; row < rows; ++row) {
        float combined_scales[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m128i half_scales = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m256 float_scales = _mm256_mul_ps(
                _mm256_cvtph_ps(half_scales),
                _mm256_loadu_ps(activation_scales + scale_group)
            );
            _mm256_storeu_ps(combined_scales + scale_group, float_scales);
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
        }
        __m256 total = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const __m256i packed = _mm256_loadu_si256(
                (const __m256i *)(weights + row * row_bytes + group * 32U)
            );
            const int8_t *x = activation + group * 128U;
            const __m256i digits0 = _mm256_and_si256(packed, mask);
            const __m256i digits1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
            const __m256i digits2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            const __m256i digits3 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
            const __m256i values0 = _mm256_loadu_si256((const __m256i *)(x + 0U));
            const __m256i values1 = _mm256_loadu_si256((const __m256i *)(x + 32U));
            const __m256i values2 = _mm256_loadu_si256((const __m256i *)(x + 64U));
            const __m256i values3 = _mm256_loadu_si256((const __m256i *)(x + 96U));
            const __m256i products01 = _mm256_add_epi16(
                _mm256_maddubs_epi16(digits0, values0),
                _mm256_maddubs_epi16(digits1, values1)
            );
            const __m256i products23 = _mm256_add_epi16(
                _mm256_maddubs_epi16(digits2, values2),
                _mm256_maddubs_epi16(digits3, values3)
            );
            const __m256i accumulator = _mm256_madd_epi16(
                _mm256_add_epi16(products01, products23),
                ones
            );
            const __m256i centered = _mm256_sub_epi32(
                accumulator,
                _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, activation_sums[group])
            );
            total = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(centered),
                _mm256_set1_ps(combined_scales[group]),
                total
            );
        }
        __m128 horizontal = _mm_add_ps(
            _mm256_castps256_ps128(total),
            _mm256_extractf128_ps(total, 1)
        );
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        output[row] = _mm_cvtss_f32(horizontal);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_2bit_avx2(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_2bit_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_gemv_q8_group128_radix5_scalar(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    for (uint64_t row = 0U; row < rows; ++row) {
        float total = 0.0f;
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes = weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const int8_t *values = activation + group * CQ3_RADIX5_ACTIVATION_STRIDE;
            int32_t dot = 0;
            for (uint32_t segment = 0U; segment < CQ3_RADIX5_BYTES_PER_GROUP; ++segment) {
                uint32_t packed = codes[segment];
                if (packed > 242U) {
                    return CQ3_ERROR_CODE;
                }
                for (uint32_t lane = 0U; lane < 5U; ++lane) {
                    const int32_t digit = (int32_t)(packed % 3U);
                    packed /= 3U;
                    dot += (digit - 1) * (int32_t)values[segment * 8U + lane];
                }
            }
            const float weight_scale = cq3_half_to_float(weight_scales[row * groups + group]);
            total += (float)dot * weight_scale * activation_scales[group];
        }
        output[row] = total;
    }
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c,fma")))
static int cq3_gemv_q8_group128_radix5_avx2_impl(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    uint64_t digit_lookup[256];
    for (uint32_t value = 0U; value < 256U; ++value) {
        uint32_t remaining = value;
        uint64_t digits = 0U;
        for (uint32_t lane = 0U; lane < 5U; ++lane) {
            digits |= (uint64_t)(remaining % 3U) << (lane * 8U);
            remaining /= 3U;
        }
        digit_lookup[value] = digits;
    }

    const uint64_t groups = columns / group_size;
    const uint64_t row_bytes = groups * CQ3_RADIX5_BYTES_PER_GROUP;
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint64_t row = 0U; row < rows; ++row) {
        float combined_scales[CQ3_MAX_GROUPS_PER_ROW];
        uint64_t scale_group = 0U;
        for (; scale_group + 8U <= groups; scale_group += 8U) {
            const __m128i half_scales = _mm_loadu_si128(
                (const __m128i *)(weight_scales + row * groups + scale_group)
            );
            const __m256 float_scales = _mm256_mul_ps(
                _mm256_cvtph_ps(half_scales),
                _mm256_loadu_ps(activation_scales + scale_group)
            );
            _mm256_storeu_ps(combined_scales + scale_group, float_scales);
        }
        for (; scale_group < groups; ++scale_group) {
            combined_scales[scale_group] =
                _cvtsh_ss(weight_scales[row * groups + scale_group])
                * activation_scales[scale_group];
        }

        __m256 total = _mm256_setzero_ps();
        for (uint64_t group = 0U; group < groups; ++group) {
            const uint8_t *codes = weights + row * row_bytes + group * CQ3_RADIX5_BYTES_PER_GROUP;
            const int8_t *values = activation + group * CQ3_RADIX5_ACTIVATION_STRIDE;
            __m256i accumulator = _mm256_setzero_si256();
            for (uint32_t segment = 0U; segment < 24U; segment += 4U) {
                uint32_t indices;
                memcpy(&indices, codes + segment, sizeof(indices));
                const __m256i digits = _mm256_i32gather_epi64(
                    (const long long int *)digit_lookup,
                    _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)indices)),
                    8
                );
                const __m256i activations = _mm256_loadu_si256(
                    (const __m256i *)(values + segment * 8U)
                );
                accumulator = _mm256_add_epi32(
                    accumulator,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(digits, activations),
                        ones
                    )
                );
            }
            const __m128i tail_indices = _mm_set_epi32(0, 0, codes[25], codes[24]);
            const __m256i tail_digits = _mm256_i32gather_epi64(
                (const long long int *)digit_lookup,
                tail_indices,
                8
            );
            const __m256i tail_activations = _mm256_loadu_si256(
                (const __m256i *)(values + 24U * 8U)
            );
            accumulator = _mm256_add_epi32(
                accumulator,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(tail_digits, tail_activations),
                    ones
                )
            );
            const __m256i centered = _mm256_sub_epi32(
                accumulator,
                _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, activation_sums[group])
            );
            total = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(centered),
                _mm256_set1_ps(combined_scales[group]),
                total
            );
        }
        __m128 horizontal = _mm_add_ps(
            _mm256_castps256_ps128(total),
            _mm256_extractf128_ps(total, 1)
        );
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        horizontal = _mm_hadd_ps(horizontal, horizontal);
        output[row] = _mm_cvtss_f32(horizontal);
    }
    return CQ3_OK;
}
#endif

int cq3_gemv_q8_group128_radix5_avx2(
    const uint8_t *CQ3_RESTRICT weights,
    const uint16_t *CQ3_RESTRICT weight_scales,
    const int8_t *CQ3_RESTRICT activation,
    const float *CQ3_RESTRICT activation_scales,
    const int32_t *CQ3_RESTRICT activation_sums,
    uint64_t rows,
    uint64_t columns,
    uint32_t group_size,
    float *CQ3_RESTRICT output
) {
    if (activation_sums == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    const int validation = cq3_validate_gemv(
        weights, weight_scales, activation, activation_scales,
        output, rows, columns, group_size
    );
    if (validation != CQ3_OK) {
        return validation;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    const uint32_t required = CQ3_FEATURE_AVX2 | CQ3_FEATURE_F16C | CQ3_FEATURE_FMA;
    if ((cq3_cpu_features() & required) != required) {
        return CQ3_ERROR_ISA;
    }
    return cq3_gemv_q8_group128_radix5_avx2_impl(
        weights, weight_scales, activation, activation_scales,
        activation_sums, rows, columns, group_size, output
    );
#else
    return CQ3_ERROR_ISA;
#endif
}

int cq3_sum_u8_scalar(const uint8_t *input, uint64_t count, uint64_t *sum) {
    if (input == NULL || sum == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
    uint64_t total = 0U;
    for (uint64_t index = 0U; index < count; ++index) {
        total += input[index];
    }
    *sum = total;
    return CQ3_OK;
}

#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static uint64_t cq3_sum_u8_avx2_impl(const uint8_t *input, uint64_t count) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i accumulator = _mm256_setzero_si256();
    uint64_t index = 0U;
    for (; index + 32U <= count; index += 32U) {
        const __m256i values = _mm256_loadu_si256((const __m256i *)(input + index));
        accumulator = _mm256_add_epi64(accumulator, _mm256_sad_epu8(values, zero));
    }
    uint64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, accumulator);
    uint64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; index < count; ++index) {
        total += input[index];
    }
    return total;
}
#endif

int cq3_sum_u8_avx2(const uint8_t *input, uint64_t count, uint64_t *sum) {
    if (input == NULL || sum == NULL) {
        return CQ3_ERROR_ARGUMENT;
    }
#if CQ3_X86 && (defined(__GNUC__) || defined(__clang__))
    if ((cq3_cpu_features() & CQ3_FEATURE_AVX2) == 0U) {
        return CQ3_ERROR_ISA;
    }
    *sum = cq3_sum_u8_avx2_impl(input, count);
    return CQ3_OK;
#else
    (void)count;
    return CQ3_ERROR_ISA;
#endif
}
