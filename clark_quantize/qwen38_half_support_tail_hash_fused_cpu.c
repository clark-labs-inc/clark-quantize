#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    CQ3_HTF_OK = 0,
    CQ3_HTF_ERROR_ARGUMENT = 1,
    CQ3_HTF_ERROR_SHAPE = 2,
    CQ3_HTF_ERROR_CODE = 3,
};

extern int cq3_hst_gemv_q8_row_scale4_scalar(
    const uint8_t *, uint64_t, const uint8_t *, const uint16_t *,
    const uint16_t *, const int8_t *, const float *, const int32_t *,
    uint64_t, uint64_t, float *
);
extern int cq3_hst_gemv_q8_row_scale4_scalar_trusted(
    const uint8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
    const int8_t *, const float *, const int32_t *, uint64_t, uint64_t, float *
);
extern int cq3_hst_gemv_q8_row_scale4_avx2_bmi2(
    const uint8_t *, uint64_t, const uint8_t *, const uint16_t *,
    const uint16_t *, const int8_t *, const float *, const int32_t *,
    uint64_t, uint64_t, float *
);
extern int cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted(
    const uint8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
    const int8_t *, const float *, const int32_t *, uint64_t, uint64_t, float *
);
extern int cq3_hash_int4_gemv_f32_scalar(
    const uint8_t *, const uint16_t *, const float *,
    uint64_t, uint32_t, float *
);
extern int cq3_hash_int4_gemv_f32_scalar_trusted(
    const uint8_t *, const uint16_t *, const float *,
    uint64_t, uint32_t, float *
);
extern int cq3_hash_int4_gemv_f32_avx2(
    const uint8_t *, const uint16_t *, const float *,
    uint64_t, uint32_t, float *
);
extern int cq3_hash_int4_gemv_f32_avx2_trusted(
    const uint8_t *, const uint16_t *, const float *,
    uint64_t, uint32_t, float *
);

static int cq3_htf_validate_route(
    const int8_t *activation,
    const float *activation_scales,
    const uint16_t *buckets,
    const int8_t *signs,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    uint64_t rows,
    uint64_t columns,
    uint32_t sketch_dim,
    float *sketch,
    float *correction,
    float *output
) {
    if (activation == NULL || activation_scales == NULL || buckets == NULL ||
        signs == NULL || correction_codes == NULL ||
        correction_row_scales == NULL || sketch == NULL ||
        correction == NULL || output == NULL) {
        return CQ3_HTF_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || sketch_dim == 0U ||
        sketch_dim > UINT16_MAX) {
        return CQ3_HTF_ERROR_SHAPE;
    }
    for (uint64_t index = 0U; index < columns; ++index) {
        if (buckets[index] >= sketch_dim) {
            return CQ3_HTF_ERROR_SHAPE;
        }
        if (signs[index] != -1 && signs[index] != 1) {
            return CQ3_HTF_ERROR_CODE;
        }
    }
    return CQ3_HTF_OK;
}

static void cq3_htf_projection_q8(
    const int8_t *activation,
    const float *activation_scales,
    const uint16_t *buckets,
    const int8_t *signs,
    uint64_t columns,
    uint32_t sketch_dim,
    float *sketch
) {
    memset(sketch, 0, (size_t)sketch_dim * sizeof(float));
    const uint32_t groups = (uint32_t)((columns + 127U) / 128U);
    for (uint32_t group = 0U; group < groups; ++group) {
        const uint64_t start = (uint64_t)group * 128U;
        const uint64_t stop = columns - start < 128U ? columns : start + 128U;
        const float scale = activation_scales[group];
        for (uint64_t index = start; index < stop; ++index) {
            sketch[buckets[index]] += (float)activation[index] *
                (float)signs[index] * scale;
        }
    }
}

static void cq3_htf_add(
    const float *correction,
    uint64_t rows,
    float *output
) {
    for (uint64_t row = 0U; row < rows; ++row) {
        output[row] += correction[row];
    }
}

int cq3_htf_gemv_q8_row_scale4_scalar_trusted(
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
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *correction_scratch,
    float *output
) {
    int status = cq3_hst_gemv_q8_row_scale4_scalar_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_projection_q8(
        activation, activation_scales, buckets, signs,
        columns, sketch_dim, sketch_scratch
    );
    status = cq3_hash_int4_gemv_f32_scalar_trusted(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, correction_scratch
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_add(correction_scratch, rows, output);
    return CQ3_HTF_OK;
}

int cq3_htf_gemv_q8_row_scale4_scalar(
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
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *correction_scratch,
    float *output
) {
    int status = cq3_htf_validate_route(
        activation, activation_scales, buckets, signs,
        correction_codes, correction_row_scales, rows, columns, sketch_dim,
        sketch_scratch, correction_scratch, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_projection_q8(
        activation, activation_scales, buckets, signs,
        columns, sketch_dim, sketch_scratch
    );
    status = cq3_hash_int4_gemv_f32_scalar(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, correction_scratch
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    status = cq3_hst_gemv_q8_row_scale4_scalar(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_add(correction_scratch, rows, output);
    return CQ3_HTF_OK;
}

int cq3_htf_gemv_q8_row_scale4_avx2_bmi2_trusted(
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
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *correction_scratch,
    float *output
) {
    int status = cq3_hst_gemv_q8_row_scale4_avx2_bmi2_trusted(
        weights, packed_scale_indices, row_anchors, ratio_codebook,
        activation, activation_scales, activation_sums, rows, columns, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_projection_q8(
        activation, activation_scales, buckets, signs,
        columns, sketch_dim, sketch_scratch
    );
    status = cq3_hash_int4_gemv_f32_avx2_trusted(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, correction_scratch
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_add(correction_scratch, rows, output);
    return CQ3_HTF_OK;
}

int cq3_htf_gemv_q8_row_scale4_avx2_bmi2(
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
    uint32_t sketch_dim,
    const uint8_t *correction_codes,
    const uint16_t *correction_row_scales,
    float *sketch_scratch,
    float *correction_scratch,
    float *output
) {
    int status = cq3_htf_validate_route(
        activation, activation_scales, buckets, signs,
        correction_codes, correction_row_scales, rows, columns, sketch_dim,
        sketch_scratch, correction_scratch, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_projection_q8(
        activation, activation_scales, buckets, signs,
        columns, sketch_dim, sketch_scratch
    );
    status = cq3_hash_int4_gemv_f32_avx2(
        correction_codes, correction_row_scales, sketch_scratch,
        rows, sketch_dim, correction_scratch
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    status = cq3_hst_gemv_q8_row_scale4_avx2_bmi2(
        weights, weight_bytes, packed_scale_indices, row_anchors,
        ratio_codebook, activation, activation_scales, activation_sums,
        rows, columns, output
    );
    if (status != CQ3_HTF_OK) {
        return status;
    }
    cq3_htf_add(correction_scratch, rows, output);
    return CQ3_HTF_OK;
}
