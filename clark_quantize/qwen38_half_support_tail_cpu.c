#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    CQ3_HST_OK = 0,
    CQ3_HST_ERROR_ARGUMENT = 1,
    CQ3_HST_ERROR_SHAPE = 2,
    CQ3_HST_ERROR_CODE = 3,
    CQ3_HST_ERROR_NONFINITE = 4,
    CQ3_HST_ERROR_ISA = 5,
};

#define CQ3_HST_GROUP 128U
#define CQ3_HST_VISION_COLUMNS 4304U
#define CQ3_HST_VISION_FULL_COLUMNS 4224U

extern uint32_t cq3_hsm64_cpu_features(void);
extern int cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
    const int8_t *, const float *, const int32_t *, uint64_t, uint64_t, float *
);

static float cq3_hst_half_to_float(uint16_t value) {
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

static uint64_t cq3_hst_group_bytes(uint32_t width) {
    return ((uint64_t)width + width / 2U + 7U) / 8U;
}

static uint32_t cq3_hst_groups(uint64_t columns) {
    return (uint32_t)((columns + CQ3_HST_GROUP - 1U) / CQ3_HST_GROUP);
}

static uint64_t cq3_hst_row_bytes(uint64_t columns) {
    const uint64_t full = columns / CQ3_HST_GROUP;
    const uint32_t tail = (uint32_t)(columns % CQ3_HST_GROUP);
    return full * cq3_hst_group_bytes(CQ3_HST_GROUP) +
        (tail == 0U ? 0U : cq3_hst_group_bytes(tail));
}

static uint32_t cq3_hst_get_bit(const uint8_t *source, uint32_t bit) {
    return (source[bit >> 3U] >> (bit & 7U)) & 1U;
}

static void cq3_hst_set_bit(uint8_t *target, uint32_t bit) {
    target[bit >> 3U] |= (uint8_t)(1U << (bit & 7U));
}

int cq3_hst_matrix_size(
    uint64_t rows,
    uint64_t columns,
    uint64_t *payload_bytes
) {
    if (payload_bytes == NULL) {
        return CQ3_HST_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns < CQ3_HST_GROUP ||
        columns > CQ3_HST_GROUP * 256U || (columns & 1U) != 0U ||
        rows > UINT64_MAX / columns) {
        return CQ3_HST_ERROR_SHAPE;
    }
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    if (row_bytes == 0U || rows > UINT64_MAX / row_bytes) {
        return CQ3_HST_ERROR_SHAPE;
    }
    *payload_bytes = rows * row_bytes;
    return CQ3_HST_OK;
}

int cq3_hst_pack_matrix(
    const int8_t *codes,
    uint64_t rows,
    uint64_t columns,
    uint8_t *payload,
    uint64_t payload_capacity_bytes
) {
    if (codes == NULL || payload == NULL) {
        return CQ3_HST_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    int status = cq3_hst_matrix_size(rows, columns, &required);
    if (status != CQ3_HST_OK) {
        return status;
    }
    if (payload_capacity_bytes < required) {
        return CQ3_HST_ERROR_SHAPE;
    }
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    for (uint64_t row = 0U; row < rows; ++row) {
        uint64_t column = 0U;
        uint64_t byte_offset = 0U;
        while (column < columns) {
            const uint32_t width = (uint32_t)(
                columns - column < CQ3_HST_GROUP
                    ? columns - column
                    : CQ3_HST_GROUP
            );
            const uint64_t group_bytes = cq3_hst_group_bytes(width);
            uint8_t *target = payload + row * row_bytes + byte_offset;
            memset(target, 0, (size_t)group_bytes);
            uint32_t observed = 0U;
            for (uint32_t position = 0U; position < width; ++position) {
                const int code = codes[row * columns + column + position];
                if (code < -1 || code > 1) {
                    return CQ3_HST_ERROR_CODE;
                }
                if (code == 0) {
                    continue;
                }
                if (observed >= width / 2U) {
                    return CQ3_HST_ERROR_CODE;
                }
                cq3_hst_set_bit(target, position);
                if (code > 0) {
                    cq3_hst_set_bit(target, width + observed);
                }
                observed += 1U;
            }
            if (observed != width / 2U) {
                return CQ3_HST_ERROR_CODE;
            }
            column += width;
            byte_offset += group_bytes;
        }
    }
    return CQ3_HST_OK;
}

static int cq3_hst_validate_payload(
    const uint8_t *payload,
    uint64_t rows,
    uint64_t columns
) {
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    for (uint64_t row = 0U; row < rows; ++row) {
        uint64_t column = 0U;
        uint64_t byte_offset = 0U;
        while (column < columns) {
            const uint32_t width = (uint32_t)(
                columns - column < CQ3_HST_GROUP
                    ? columns - column
                    : CQ3_HST_GROUP
            );
            const uint8_t *source = payload + row * row_bytes + byte_offset;
            uint32_t support = 0U;
            for (uint32_t position = 0U; position < width; ++position) {
                support += cq3_hst_get_bit(source, position);
            }
            if (support != width / 2U) {
                return CQ3_HST_ERROR_CODE;
            }
            const uint32_t used_bits = width + width / 2U;
            const uint32_t allocated_bits = (uint32_t)cq3_hst_group_bytes(width) * 8U;
            for (uint32_t bit = used_bits; bit < allocated_bits; ++bit) {
                if (cq3_hst_get_bit(source, bit) != 0U) {
                    return CQ3_HST_ERROR_CODE;
                }
            }
            column += width;
            byte_offset += cq3_hst_group_bytes(width);
        }
    }
    return CQ3_HST_OK;
}

int cq3_hst_unpack_matrix(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t rows,
    uint64_t columns,
    int8_t *codes
) {
    if (payload == NULL || codes == NULL) {
        return CQ3_HST_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    int status = cq3_hst_matrix_size(rows, columns, &required);
    if (status != CQ3_HST_OK) {
        return status;
    }
    if (payload_bytes != required) {
        return CQ3_HST_ERROR_SHAPE;
    }
    status = cq3_hst_validate_payload(payload, rows, columns);
    if (status != CQ3_HST_OK) {
        return status;
    }
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    memset(codes, 0, (size_t)(rows * columns));
    for (uint64_t row = 0U; row < rows; ++row) {
        uint64_t column = 0U;
        uint64_t byte_offset = 0U;
        while (column < columns) {
            const uint32_t width = (uint32_t)(
                columns - column < CQ3_HST_GROUP
                    ? columns - column
                    : CQ3_HST_GROUP
            );
            const uint8_t *source = payload + row * row_bytes + byte_offset;
            uint32_t sign_index = 0U;
            for (uint32_t position = 0U; position < width; ++position) {
                if (cq3_hst_get_bit(source, position) != 0U) {
                    codes[row * columns + column + position] =
                        cq3_hst_get_bit(source, width + sign_index) != 0U
                            ? 1
                            : -1;
                    sign_index += 1U;
                }
            }
            column += width;
            byte_offset += cq3_hst_group_bytes(width);
        }
    }
    return CQ3_HST_OK;
}

static int cq3_hst_validate_gemv(
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
    float *output
) {
    if (weights == NULL || packed_scale_indices == NULL ||
        row_anchors == NULL || ratio_codebook == NULL || activation == NULL ||
        activation_scales == NULL || activation_sums == NULL || output == NULL) {
        return CQ3_HST_ERROR_ARGUMENT;
    }
    uint64_t required = 0U;
    int status = cq3_hst_matrix_size(rows, columns, &required);
    if (status != CQ3_HST_OK || weight_bytes != required) {
        return CQ3_HST_ERROR_SHAPE;
    }
    status = cq3_hst_validate_payload(weights, rows, columns);
    if (status != CQ3_HST_OK) {
        return status;
    }
    for (uint32_t index = 0U; index < 16U; ++index) {
        if (!isfinite(cq3_hst_half_to_float(ratio_codebook[index]))) {
            return CQ3_HST_ERROR_NONFINITE;
        }
    }
    for (uint64_t row = 0U; row < rows; ++row) {
        if (!isfinite(cq3_hst_half_to_float(row_anchors[row]))) {
            return CQ3_HST_ERROR_NONFINITE;
        }
    }
    const uint32_t groups = cq3_hst_groups(columns);
    for (uint32_t group = 0U; group < groups; ++group) {
        if (!isfinite(activation_scales[group])) {
            return CQ3_HST_ERROR_NONFINITE;
        }
    }
    return CQ3_HST_OK;
}

int cq3_hst_gemv_q8_row_scale4_scalar_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    (void)activation_sums;
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    const uint32_t groups = cq3_hst_groups(columns);
    for (uint64_t row = 0U; row < rows; ++row) {
        float total = 0.0f;
        uint64_t column = 0U;
        uint64_t byte_offset = 0U;
        for (uint32_t group = 0U; group < groups; ++group) {
            const uint32_t width = (uint32_t)(
                columns - column < CQ3_HST_GROUP
                    ? columns - column
                    : CQ3_HST_GROUP
            );
            const uint8_t *source = weights + row * row_bytes + byte_offset;
            int32_t dot = 0;
            uint32_t sign_index = 0U;
            for (uint32_t position = 0U; position < width; ++position) {
                if (cq3_hst_get_bit(source, position) != 0U) {
                    const int32_t value = (int32_t)activation[column + position];
                    dot += cq3_hst_get_bit(source, width + sign_index) != 0U
                        ? value
                        : -value;
                    sign_index += 1U;
                }
            }
            const uint64_t scale_index = row * groups + group;
            const uint8_t ratio_index = (packed_scale_indices[scale_index / 2U] >>
                ((scale_index & 1U) * 4U)) & 0x0fU;
            total += (float)dot * cq3_hst_half_to_float(row_anchors[row]) *
                cq3_hst_half_to_float(ratio_codebook[ratio_index]) *
                activation_scales[group];
            column += width;
            byte_offset += cq3_hst_group_bytes(width);
        }
        output[row] = total;
    }
    return CQ3_HST_OK;
}

int cq3_hst_gemv_q8_row_scale4_scalar(
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
    float *output
) {
    int status = cq3_hst_validate_gemv(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HST_OK) {
        return status;
    }
    return cq3_hst_gemv_q8_row_scale4_scalar_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
}

int cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *weights,
    const uint8_t *packed_scale_indices,
    const uint16_t *row_anchors,
    const uint16_t *ratio_codebook,
    const int8_t *activation,
    const float *activation_scales,
    const int32_t *activation_sums,
    uint64_t rows,
    uint64_t columns,
    float *output
) {
    if (columns != CQ3_HST_VISION_COLUMNS) {
        return CQ3_HST_ERROR_SHAPE;
    }
    const uint64_t row_bytes = cq3_hst_row_bytes(columns);
    const uint32_t groups = cq3_hst_groups(columns);
    const uint32_t scale_row_bytes = (groups + 1U) / 2U;
    const uint32_t full_groups = CQ3_HST_VISION_FULL_COLUMNS / CQ3_HST_GROUP;
    const int8_t *tail_activation = activation + CQ3_HST_VISION_FULL_COLUMNS;
    for (uint64_t row = 0U; row < rows; ++row) {
        int status = cq3_hsm64_gemv_q8_row_scale4_avx2_bmi2_trusted(
            weights + row * row_bytes,
            packed_scale_indices + row * scale_row_bytes,
            row_anchors + row,
            ratio_codebook,
            activation,
            activation_scales,
            activation_sums,
            1U,
            CQ3_HST_VISION_FULL_COLUMNS,
            output + row
        );
        if (status != CQ3_HST_OK) {
            return status;
        }
        const uint8_t *tail = weights + row * row_bytes +
            full_groups * cq3_hst_group_bytes(CQ3_HST_GROUP);
        int32_t dot = 0;
        uint32_t sign_index = 0U;
        for (uint32_t position = 0U; position < 80U; ++position) {
            if (cq3_hst_get_bit(tail, position) != 0U) {
                const int32_t value = (int32_t)tail_activation[position];
                dot += cq3_hst_get_bit(tail, 80U + sign_index) != 0U
                    ? value
                    : -value;
                sign_index += 1U;
            }
        }
        const uint8_t ratio_index =
            (packed_scale_indices[row * scale_row_bytes + 16U] >> 4U) & 0x0fU;
        output[row] += (float)dot * cq3_hst_half_to_float(row_anchors[row]) *
            cq3_hst_half_to_float(ratio_codebook[ratio_index]) *
            activation_scales[full_groups];
    }
    return CQ3_HST_OK;
}

int cq3_hst_gemv_q8_row_scale4_avx2_bmi2(
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
    float *output
) {
    int status = cq3_hst_validate_gemv(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HST_OK) {
        return status;
    }
    const uint32_t required = 1U | 2U | 4U | 8U;
    if ((cq3_hsm64_cpu_features() & required) != required) {
        return CQ3_HST_ERROR_ISA;
    }
    return cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
}
