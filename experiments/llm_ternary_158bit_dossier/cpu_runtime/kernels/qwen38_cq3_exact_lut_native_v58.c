#include "qwen38_cq3_exact_lut_cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 32,
    COLUMNS = 256,
    GROUPS = COLUMNS / CQ3_EXACT_LUT_GROUP,
};

static int8_t code_at(int row, int column) {
    return (int8_t)(((row * 17 + column * 29 + 7) % 3) - 1);
}

static int8_t activation_at(int group, int position) {
    return (int8_t)(((group * 43 + position * 37 + 19) % 256) - 128);
}

static void set_source_code(
        cq3_exact_lut_source_block *block,
        int group,
        int position,
        int8_t code) {
    const int lane = position & 31;
    const int quarter = position >> 5;
    const uint8_t digit = (uint8_t)(code + 1);
    block->qs[group * 32 + lane] |= (uint8_t)(digit << (2 * quarter));
}

static int float_bits_equal(const float *left, const float *right, int count) {
    return memcmp(left, right, (size_t)count * sizeof(float)) == 0;
}

int main(void) {
    cq3_exact_lut_source_block source[ROWS];
    cq3_exact_lut_activation activations[GROUPS];
    float reference[ROWS] = {0.0f};
    float scalar[ROWS] = {0.0f};
    float avx2[ROWS] = {0.0f};
    memset(source, 0, sizeof(source));
    memset(activations, 0, sizeof(activations));

    for (int row = 0; row < ROWS; ++row) {
        source[row].d[0] = UINT16_C(0x3800);  // binary16 0.5
        source[row].d[1] = UINT16_C(0x3c00);  // binary16 1.0
        for (int group = 0; group < GROUPS; ++group) {
            for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
                const int column = group * CQ3_EXACT_LUT_GROUP + position;
                set_source_code(
                    source + row, group, position, code_at(row, column)
                );
            }
        }
    }
    for (int group = 0; group < GROUPS; ++group) {
        activations[group].d = group == 0 ? 0.25f : 0.5f;
        int sum = 0;
        for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
            const int8_t value = activation_at(group, position);
            activations[group].qs[position] = value;
            sum += value;
        }
        activations[group].sum = (int16_t)sum;
    }
    for (int row = 0; row < ROWS; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            int dot = 0;
            for (int position = 0; position < CQ3_EXACT_LUT_GROUP; ++position) {
                const int column = group * CQ3_EXACT_LUT_GROUP + position;
                dot += code_at(row, column) * activation_at(group, position);
            }
            const float weight_scale = group == 0 ? 0.5f : 1.0f;
            reference[row] +=
                (float)dot * weight_scale * activations[group].d;
        }
    }

    const size_t compute_bytes = cq3_exact_lut_compute_image_bytes(ROWS, COLUMNS);
    const size_t temporary_bytes = cq3_exact_lut_temporary_bytes(COLUMNS);
    if (compute_bytes != sizeof(source) || temporary_bytes != 2U * 1024U) {
        fprintf(stderr, "v58 byte contract changed\n");
        return 2;
    }
    cq3_exact_lut_tile32 *compute_image =
        (cq3_exact_lut_tile32 *)malloc(compute_bytes);
    cq3_exact_lut_table *tables =
        (cq3_exact_lut_table *)malloc(temporary_bytes);
    if (compute_image == NULL || tables == NULL) {
        fprintf(stderr, "v58 allocation failed\n");
        free(compute_image);
        free(tables);
        return 2;
    }
    if (cq3_exact_lut_repack(source, compute_image, ROWS, COLUMNS) != 0 ||
        cq3_exact_lut_build_tables(activations, tables, COLUMNS) != 0 ||
        cq3_exact_lut_gemv_scalar_trusted(
            compute_image, activations, tables, scalar, ROWS, COLUMNS
        ) != 0) {
        fprintf(stderr, "v58 scalar path failed\n");
        free(compute_image);
        free(tables);
        return 2;
    }
    if (!float_bits_equal(reference, scalar, ROWS)) {
        fprintf(stderr, "v58 scalar output changed direct algebra\n");
        free(compute_image);
        free(tables);
        return 2;
    }

    const int avx_status = cq3_exact_lut_gemv_avx2_trusted(
        compute_image, activations, tables, avx2, ROWS, COLUMNS
    );
    if (avx_status != CQ3_EXACT_LUT_OK &&
        avx_status != CQ3_EXACT_LUT_ERROR_ISA) {
        fprintf(stderr, "v58 AVX2 path failed unexpectedly\n");
        free(compute_image);
        free(tables);
        return 2;
    }
    if (avx_status == CQ3_EXACT_LUT_OK &&
        !float_bits_equal(scalar, avx2, ROWS)) {
        float maximum = 0.0f;
        for (int row = 0; row < ROWS; ++row) {
            const float delta = fabsf(scalar[row] - avx2[row]);
            maximum = delta > maximum ? delta : maximum;
        }
        fprintf(stderr, "v58 AVX2 output changed scalar bits: %.9g\n", maximum);
        free(compute_image);
        free(tables);
        return 2;
    }
    printf(
        "{\"schema\":\"qwen38_cq3_exact_lut_native_v58\","
        "\"rows\":%d,\"columns\":%d,\"compute_bytes\":%zu,"
        "\"temporary_bytes\":%zu,\"scalar_direct_bit_exact\":true,"
        "\"avx2_executed\":%s,\"avx2_scalar_bit_exact\":%s,"
        "\"complete\":true,\"terminal\":true}\n",
        ROWS,
        COLUMNS,
        compute_bytes,
        temporary_bytes,
        avx_status == CQ3_EXACT_LUT_OK ? "true" : "false",
        avx_status == CQ3_EXACT_LUT_OK ? "true" : "null"
    );
    free(compute_image);
    free(tables);
    return 0;
}
