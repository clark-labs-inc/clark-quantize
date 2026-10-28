#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

enum {
    CQ3_HFS_OK = 0,
    CQ3_HFS_ERROR_ARGUMENT = 1,
    CQ3_HFS_ERROR_SHAPE = 2,
    CQ3_HFS_ERROR_CODE = 3,
    CQ3_HFS_ERROR_CAPACITY = 4,
};

#define CQ3_HFS_GROUP 128U

static unsigned __int128 cq3_hfs_binomial_table[CQ3_HFS_GROUP + 1U][65U];
static atomic_int cq3_hfs_binomial_state = ATOMIC_VAR_INIT(0);

static void cq3_hfs_initialize_binomial_table(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(
            &cq3_hfs_binomial_state,
            &expected,
            1,
            memory_order_acq_rel,
            memory_order_acquire)) {
        cq3_hfs_binomial_table[0U][0U] = 1U;
        for (uint32_t n = 1U; n <= CQ3_HFS_GROUP; ++n) {
            cq3_hfs_binomial_table[n][0U] = 1U;
            const uint32_t maximum = n < 64U ? n : 64U;
            for (uint32_t k = 1U; k <= maximum; ++k) {
                const unsigned __int128 left =
                    cq3_hfs_binomial_table[n - 1U][k - 1U];
                const unsigned __int128 right =
                    k < n ? cq3_hfs_binomial_table[n - 1U][k] : 0U;
                cq3_hfs_binomial_table[n][k] = left + right;
            }
        }
        atomic_store_explicit(
            &cq3_hfs_binomial_state, 2, memory_order_release
        );
        return;
    }
    while (atomic_load_explicit(
            &cq3_hfs_binomial_state, memory_order_acquire) != 2) {
    }
}

static int cq3_hfs_binomial(
    uint32_t n,
    uint32_t k,
    unsigned __int128 *output
) {
    if (output == NULL || k > n || n > CQ3_HFS_GROUP) {
        return CQ3_HFS_ERROR_ARGUMENT;
    }
    if (k > n - k) {
        k = n - k;
    }
    cq3_hfs_initialize_binomial_table();
    *output = cq3_hfs_binomial_table[n][k];
    return CQ3_HFS_OK;
}

static uint32_t cq3_hfs_bit_length_u128(unsigned __int128 value) {
    uint32_t bits = 0U;
    while (value != 0U) {
        value >>= 1U;
        bits += 1U;
    }
    return bits;
}

int cq3_hfs_group_width(uint32_t nonzero_per_group, uint32_t *width_bits) {
    if (width_bits == NULL || nonzero_per_group == 0U ||
        nonzero_per_group > 64U) {
        return CQ3_HFS_ERROR_ARGUMENT;
    }
    unsigned __int128 combinations = 0U;
    const int status = cq3_hfs_binomial(
        CQ3_HFS_GROUP, nonzero_per_group, &combinations
    );
    if (status != CQ3_HFS_OK || combinations == 0U) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    *width_bits = cq3_hfs_bit_length_u128(combinations - 1U) + nonzero_per_group;
    return CQ3_HFS_OK;
}

int cq3_hfs_matrix_size(
    uint64_t rows,
    uint64_t columns,
    uint32_t nonzero_per_group,
    uint64_t *payload_bits,
    uint64_t *payload_bytes
) {
    if (payload_bits == NULL || payload_bytes == NULL) {
        return CQ3_HFS_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || columns % CQ3_HFS_GROUP != 0U) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    uint32_t width = 0U;
    const int status = cq3_hfs_group_width(nonzero_per_group, &width);
    if (status != CQ3_HFS_OK) {
        return status;
    }
    const uint64_t groups_per_row = columns / CQ3_HFS_GROUP;
    if (rows > UINT64_MAX / columns ||
        rows * columns > (uint64_t)SIZE_MAX ||
        rows > UINT64_MAX / groups_per_row ||
        rows * groups_per_row > UINT64_MAX / width) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    const uint64_t bits = rows * groups_per_row * width;
    const uint64_t bytes = bits / 8U + (bits % 8U != 0U ? 1U : 0U);
    if (bytes > (uint64_t)SIZE_MAX) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    *payload_bits = bits;
    *payload_bytes = bytes;
    return CQ3_HFS_OK;
}

static int cq3_hfs_rank_group(
    const int8_t codes[CQ3_HFS_GROUP],
    uint32_t support_size,
    uint64_t limbs[3]
) {
    uint8_t positions[64];
    uint32_t observed = 0U;
    uint64_t signs = 0U;
    for (uint32_t position = 0U; position < CQ3_HFS_GROUP; ++position) {
        const int code = (int)codes[position];
        if (code < -1 || code > 1) {
            return CQ3_HFS_ERROR_CODE;
        }
        if (code != 0) {
            if (observed >= support_size || observed >= 64U) {
                return CQ3_HFS_ERROR_CODE;
            }
            positions[observed] = (uint8_t)position;
            if (code > 0) {
                signs |= UINT64_C(1) << observed;
            }
            observed += 1U;
        }
    }
    if (observed != support_size) {
        return CQ3_HFS_ERROR_CODE;
    }
    unsigned __int128 rank = 0U;
    int32_t previous = -1;
    uint32_t remaining = support_size;
    for (uint32_t index = 0U; index < support_size; ++index) {
        const uint32_t position = positions[index];
        for (uint32_t candidate = (uint32_t)(previous + 1);
             candidate < position; ++candidate) {
            unsigned __int128 count = 0U;
            const int status = cq3_hfs_binomial(
                CQ3_HFS_GROUP - candidate - 1U, remaining - 1U, &count
            );
            if (status != CQ3_HFS_OK) {
                return status;
            }
            rank += count;
        }
        previous = (int32_t)position;
        remaining -= 1U;
    }
    const uint64_t rank_low = (uint64_t)rank;
    const uint64_t rank_high = (uint64_t)(rank >> 64U);
    if (support_size == 64U) {
        limbs[0] = signs;
        limbs[1] = rank_low;
        limbs[2] = rank_high;
    } else if (support_size == 0U) {
        limbs[0] = rank_low;
        limbs[1] = rank_high;
        limbs[2] = 0U;
    } else {
        limbs[0] = signs | (rank_low << support_size);
        limbs[1] = (rank_low >> (64U - support_size)) |
            (rank_high << support_size);
        limbs[2] = rank_high >> (64U - support_size);
    }
    return CQ3_HFS_OK;
}

static int cq3_hfs_unrank_group(
    const uint64_t limbs[3],
    uint32_t support_size,
    int8_t codes[CQ3_HFS_GROUP]
) {
    uint64_t signs;
    unsigned __int128 rank;
    if (support_size == 64U) {
        signs = limbs[0];
        rank = (unsigned __int128)limbs[1] |
            ((unsigned __int128)limbs[2] << 64U);
    } else if (support_size == 0U) {
        signs = 0U;
        rank = (unsigned __int128)limbs[0] |
            ((unsigned __int128)limbs[1] << 64U);
    } else {
        signs = limbs[0] & ((UINT64_C(1) << support_size) - 1U);
        const uint64_t rank_low = (limbs[0] >> support_size) |
            (limbs[1] << (64U - support_size));
        const uint64_t rank_high = (limbs[1] >> support_size) |
            (limbs[2] << (64U - support_size));
        rank = (unsigned __int128)rank_low |
            ((unsigned __int128)rank_high << 64U);
    }
    unsigned __int128 combinations = 0U;
    int status = cq3_hfs_binomial(
        CQ3_HFS_GROUP, support_size, &combinations
    );
    if (status != CQ3_HFS_OK || rank >= combinations) {
        return CQ3_HFS_ERROR_CODE;
    }
    memset(codes, 0, CQ3_HFS_GROUP * sizeof(int8_t));
    uint32_t start = 0U;
    for (uint32_t remaining = support_size; remaining > 0U; --remaining) {
        uint32_t selected = CQ3_HFS_GROUP;
        for (uint32_t candidate = start;
             candidate <= CQ3_HFS_GROUP - remaining; ++candidate) {
            unsigned __int128 count = 0U;
            status = cq3_hfs_binomial(
                CQ3_HFS_GROUP - candidate - 1U, remaining - 1U, &count
            );
            if (status != CQ3_HFS_OK) {
                return status;
            }
            if (rank < count) {
                selected = candidate;
                break;
            }
            rank -= count;
        }
        if (selected == CQ3_HFS_GROUP) {
            return CQ3_HFS_ERROR_CODE;
        }
        const uint32_t sign_index = support_size - remaining;
        codes[selected] = ((signs >> sign_index) & 1U) != 0U ? 1 : -1;
        start = selected + 1U;
    }
    return rank == 0U ? CQ3_HFS_OK : CQ3_HFS_ERROR_CODE;
}

static void cq3_hfs_write_limbs(
    const uint64_t limbs[3],
    uint32_t width,
    uint8_t *payload,
    uint64_t *cursor
) {
    for (uint32_t bit = 0U; bit < width; ++bit) {
        if (((limbs[bit >> 6U] >> (bit & 63U)) & 1U) != 0U) {
            payload[*cursor >> 3U] |= (uint8_t)(1U << (*cursor & 7U));
        }
        *cursor += 1U;
    }
}

static void cq3_hfs_read_limbs(
    const uint8_t *payload,
    uint32_t width,
    uint64_t *cursor,
    uint64_t limbs[3]
) {
    limbs[0] = 0U;
    limbs[1] = 0U;
    limbs[2] = 0U;
    for (uint32_t bit = 0U; bit < width; ++bit) {
        if (((payload[*cursor >> 3U] >> (*cursor & 7U)) & 1U) != 0U) {
            limbs[bit >> 6U] |= UINT64_C(1) << (bit & 63U);
        }
        *cursor += 1U;
    }
}

int cq3_hfs_pack_matrix(
    const int8_t *codes,
    uint64_t rows,
    uint64_t columns,
    uint32_t nonzero_per_group,
    uint8_t *payload,
    uint64_t payload_capacity_bytes,
    uint64_t *payload_bits
) {
    if (codes == NULL || payload == NULL || payload_bits == NULL) {
        return CQ3_HFS_ERROR_ARGUMENT;
    }
    uint64_t required_bits = 0U;
    uint64_t required_bytes = 0U;
    int status = cq3_hfs_matrix_size(
        rows, columns, nonzero_per_group, &required_bits, &required_bytes
    );
    if (status != CQ3_HFS_OK) {
        return status;
    }
    if (payload_capacity_bytes < required_bytes) {
        return CQ3_HFS_ERROR_CAPACITY;
    }
    memset(payload, 0, (size_t)required_bytes);
    uint32_t width = 0U;
    status = cq3_hfs_group_width(nonzero_per_group, &width);
    if (status != CQ3_HFS_OK) {
        return status;
    }
    const uint64_t groups = rows * (columns / CQ3_HFS_GROUP);
    uint64_t cursor = 0U;
    for (uint64_t group = 0U; group < groups; ++group) {
        uint64_t limbs[3];
        status = cq3_hfs_rank_group(
            codes + group * CQ3_HFS_GROUP, nonzero_per_group, limbs
        );
        if (status != CQ3_HFS_OK) {
            return status;
        }
        cq3_hfs_write_limbs(limbs, width, payload, &cursor);
    }
    if (cursor != required_bits) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    *payload_bits = required_bits;
    return CQ3_HFS_OK;
}

int cq3_hfs_unpack_matrix(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t payload_bits,
    uint64_t rows,
    uint64_t columns,
    uint32_t nonzero_per_group,
    int8_t *codes
) {
    if (payload == NULL || codes == NULL) {
        return CQ3_HFS_ERROR_ARGUMENT;
    }
    uint64_t required_bits = 0U;
    uint64_t required_bytes = 0U;
    int status = cq3_hfs_matrix_size(
        rows, columns, nonzero_per_group, &required_bits, &required_bytes
    );
    if (status != CQ3_HFS_OK) {
        return status;
    }
    if (payload_bits != required_bits || payload_bytes != required_bytes) {
        return CQ3_HFS_ERROR_SHAPE;
    }
    if (payload_bits % 8U != 0U &&
        (payload[payload_bytes - 1U] >> (payload_bits % 8U)) != 0U) {
        return CQ3_HFS_ERROR_CODE;
    }
    uint32_t width = 0U;
    status = cq3_hfs_group_width(nonzero_per_group, &width);
    if (status != CQ3_HFS_OK) {
        return status;
    }
    const uint64_t groups = rows * (columns / CQ3_HFS_GROUP);
    uint64_t cursor = 0U;
    for (uint64_t group = 0U; group < groups; ++group) {
        uint64_t limbs[3];
        cq3_hfs_read_limbs(payload, width, &cursor, limbs);
        status = cq3_hfs_unrank_group(
            limbs, nonzero_per_group, codes + group * CQ3_HFS_GROUP
        );
        if (status != CQ3_HFS_OK) {
            return status;
        }
    }
    return cursor == required_bits ? CQ3_HFS_OK : CQ3_HFS_ERROR_SHAPE;
}
