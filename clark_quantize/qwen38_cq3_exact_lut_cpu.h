#ifndef CLARK_QUANTIZE_QWEN38_CQ3_EXACT_LUT_CPU_H
#define CLARK_QUANTIZE_QWEN38_CQ3_EXACT_LUT_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CQ3_EXACT_LUT_OK = 0,
    CQ3_EXACT_LUT_ERROR_ARGUMENT = 1,
    CQ3_EXACT_LUT_ERROR_SHAPE = 2,
    CQ3_EXACT_LUT_ERROR_CODE = 3,
    CQ3_EXACT_LUT_ERROR_ISA = 4,
};

enum {
    CQ3_EXACT_LUT_GROUP = 128,
    CQ3_EXACT_LUT_BLOCK = 256,
    CQ3_EXACT_LUT_TILE_ROWS = 32,
    CQ3_EXACT_LUT_QUARTETS = 32,
    CQ3_EXACT_LUT_SUBSETS = 16,
};

// Byte-compatible mathematical view of one upstream CQ3_0 block.
typedef struct {
    uint8_t qs[64];
    uint16_t d[2];
} cq3_exact_lut_source_block;

// Storage-neutral 32-row compute tile for one 256-column CQ3_0 block.
typedef struct {
    uint16_t d[2][CQ3_EXACT_LUT_TILE_ROWS];
    uint8_t masks[2][CQ3_EXACT_LUT_QUARTETS][CQ3_EXACT_LUT_TILE_ROWS];
} cq3_exact_lut_tile32;

// Exact view of the already-quantized CQ8 activation algebra.
typedef struct {
    float d;
    int8_t qs[CQ3_EXACT_LUT_GROUP];
    int16_t sum;
} cq3_exact_lut_activation;

// Low and high bytes of sixteen signed-INT16 subset sums.
typedef struct {
    uint8_t low[CQ3_EXACT_LUT_SUBSETS];
    uint8_t high[CQ3_EXACT_LUT_SUBSETS];
} cq3_exact_lut_table;

size_t cq3_exact_lut_compute_image_bytes(int rows, int columns);
size_t cq3_exact_lut_temporary_bytes(int columns);

int cq3_exact_lut_repack(
        const cq3_exact_lut_source_block *source,
        cq3_exact_lut_tile32 *destination,
        int rows,
        int columns);

int cq3_exact_lut_build_tables(
        const cq3_exact_lut_activation *activations,
        cq3_exact_lut_table *tables,
        int columns);

int cq3_exact_lut_gemv_scalar_trusted(
        const cq3_exact_lut_tile32 *weights,
        const cq3_exact_lut_activation *activations,
        const cq3_exact_lut_table *tables,
        float *output,
        int rows,
        int columns);

int cq3_exact_lut_gemv_avx2_trusted(
        const cq3_exact_lut_tile32 *weights,
        const cq3_exact_lut_activation *activations,
        const cq3_exact_lut_table *tables,
        float *output,
        int rows,
        int columns);

#ifdef __cplusplus
}
#endif

#endif
