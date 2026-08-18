#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

enum {
    CQ3_GHLT_OK = 0,
    CQ3_GHLT_ERROR_ARGUMENT = 1,
    CQ3_GHLT_ERROR_SHAPE = 2,
    CQ3_GHLT_ERROR_CODE = 3,
    CQ3_GHLT_ERROR_CAPACITY = 4,
};

#define CQ3_GHLT_GROUP 128U
#define CQ3_GHLT_SUPPORT 64U
#define CQ3_GHLT_INPUT_BITS 189U
#define CQ3_GHLT_OUTPUT_BYTES 24U

static unsigned __int128 cq3_ghlt_binomial_table[129U][65U];
static atomic_int cq3_ghlt_binomial_state = ATOMIC_VAR_INIT(0);

static void cq3_ghlt_initialize_binomial_table(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(
            &cq3_ghlt_binomial_state,
            &expected,
            1,
            memory_order_acq_rel,
            memory_order_acquire)) {
        cq3_ghlt_binomial_table[0U][0U] = 1U;
        for (uint32_t n = 1U; n <= CQ3_GHLT_GROUP; ++n) {
            cq3_ghlt_binomial_table[n][0U] = 1U;
            const uint32_t maximum = n < CQ3_GHLT_SUPPORT
                ? n : CQ3_GHLT_SUPPORT;
            for (uint32_t k = 1U; k <= maximum; ++k) {
                const unsigned __int128 left =
                    cq3_ghlt_binomial_table[n - 1U][k - 1U];
                const unsigned __int128 right =
                    k < n ? cq3_ghlt_binomial_table[n - 1U][k] : 0U;
                cq3_ghlt_binomial_table[n][k] = left + right;
            }
        }
        atomic_store_explicit(
            &cq3_ghlt_binomial_state, 2, memory_order_release
        );
        return;
    }
    while (atomic_load_explicit(
            &cq3_ghlt_binomial_state, memory_order_acquire) != 2) {
    }
}

static uint64_t cq3_ghlt_read_u64_window(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t bit_cursor
) {
    const uint64_t byte_cursor = bit_cursor >> 3U;
    const uint32_t shift = (uint32_t)(bit_cursor & 7U);
    uint64_t low = 0U;
    for (uint32_t index = 0U; index < 8U; ++index) {
        if (byte_cursor + index < payload_bytes) {
            low |= (uint64_t)payload[byte_cursor + index] << (8U * index);
        }
    }
    if (shift == 0U) {
        return low;
    }
    const uint8_t high = byte_cursor + 8U < payload_bytes
        ? payload[byte_cursor + 8U] : 0U;
    return (low >> shift) | ((uint64_t)high << (64U - shift));
}

static void cq3_ghlt_read_group(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t *cursor,
    uint64_t limbs[3]
) {
    limbs[0] = cq3_ghlt_read_u64_window(payload, payload_bytes, *cursor);
    limbs[1] = cq3_ghlt_read_u64_window(
        payload, payload_bytes, *cursor + 64U
    );
    limbs[2] = cq3_ghlt_read_u64_window(
        payload, payload_bytes, *cursor + 128U
    ) & ((UINT64_C(1) << 61U) - 1U);
    *cursor += CQ3_GHLT_INPUT_BITS;
}

static void cq3_ghlt_store_le64(uint8_t *target, uint64_t value) {
    for (uint32_t index = 0U; index < 8U; ++index) {
        target[index] = (uint8_t)(value >> (8U * index));
    }
}

static int cq3_ghlt_unrank_mask64(
    const uint64_t limbs[3],
    uint64_t *support_low,
    uint64_t *support_high
) {
    unsigned __int128 rank = (unsigned __int128)limbs[1] |
        ((unsigned __int128)limbs[2] << 64U);
    const unsigned __int128 combinations =
        cq3_ghlt_binomial_table[CQ3_GHLT_GROUP][CQ3_GHLT_SUPPORT];
    if (rank >= combinations) {
        return CQ3_GHLT_ERROR_CODE;
    }
    uint64_t low = 0U;
    uint64_t high = 0U;
    uint32_t start = 0U;
    for (uint32_t remaining = CQ3_GHLT_SUPPORT; remaining > 0U; --remaining) {
        uint32_t selected = CQ3_GHLT_GROUP;
        for (uint32_t candidate = start;
             candidate <= CQ3_GHLT_GROUP - remaining; ++candidate) {
            const uint32_t n = CQ3_GHLT_GROUP - candidate - 1U;
            uint32_t k = remaining - 1U;
            if (k > n - k) {
                k = n - k;
            }
            const unsigned __int128 count = cq3_ghlt_binomial_table[n][k];
            if (rank < count) {
                selected = candidate;
                break;
            }
            rank -= count;
        }
        if (selected == CQ3_GHLT_GROUP) {
            return CQ3_GHLT_ERROR_CODE;
        }
        if (selected < 64U) {
            low |= UINT64_C(1) << selected;
        } else {
            high |= UINT64_C(1) << (selected - 64U);
        }
        start = selected + 1U;
    }
    if (rank != 0U) {
        return CQ3_GHLT_ERROR_CODE;
    }
    *support_low = low;
    *support_high = high;
    return CQ3_GHLT_OK;
}

int cq3_ghlt_mask64_size(
    uint64_t rows,
    uint64_t columns,
    uint64_t *output_bytes
) {
    if (output_bytes == NULL) {
        return CQ3_GHLT_ERROR_ARGUMENT;
    }
    if (rows == 0U || columns == 0U || columns % CQ3_GHLT_GROUP != 0U) {
        return CQ3_GHLT_ERROR_SHAPE;
    }
    const uint64_t groups_per_row = columns / CQ3_GHLT_GROUP;
    if (rows > UINT64_MAX / groups_per_row ||
        rows * groups_per_row > UINT64_MAX / CQ3_GHLT_OUTPUT_BYTES) {
        return CQ3_GHLT_ERROR_SHAPE;
    }
    *output_bytes = rows * groups_per_row * CQ3_GHLT_OUTPUT_BYTES;
    return CQ3_GHLT_OK;
}

int cq3_ghlt_transcode_mask64(
    const uint8_t *payload,
    uint64_t payload_bytes,
    uint64_t payload_bits,
    uint64_t rows,
    uint64_t columns,
    uint8_t *output,
    uint64_t output_capacity_bytes
) {
    if (payload == NULL || output == NULL) {
        return CQ3_GHLT_ERROR_ARGUMENT;
    }
    uint64_t required_output = 0U;
    const int size_status = cq3_ghlt_mask64_size(
        rows, columns, &required_output
    );
    if (size_status != CQ3_GHLT_OK) {
        return size_status;
    }
    const uint64_t groups = rows * (columns / CQ3_GHLT_GROUP);
    if (groups > UINT64_MAX / CQ3_GHLT_INPUT_BITS) {
        return CQ3_GHLT_ERROR_SHAPE;
    }
    const uint64_t required_bits = groups * CQ3_GHLT_INPUT_BITS;
    const uint64_t required_bytes = required_bits / 8U +
        (required_bits % 8U != 0U ? 1U : 0U);
    if (payload_bits != required_bits || payload_bytes != required_bytes) {
        return CQ3_GHLT_ERROR_SHAPE;
    }
    if (output_capacity_bytes < required_output) {
        return CQ3_GHLT_ERROR_CAPACITY;
    }
    if (required_bits % 8U != 0U &&
        (payload[required_bytes - 1U] >> (required_bits % 8U)) != 0U) {
        return CQ3_GHLT_ERROR_CODE;
    }
    cq3_ghlt_initialize_binomial_table();
    uint64_t cursor = 0U;
    for (uint64_t group = 0U; group < groups; ++group) {
        uint64_t limbs[3];
        cq3_ghlt_read_group(payload, payload_bytes, &cursor, limbs);
        uint64_t support_low = 0U;
        uint64_t support_high = 0U;
        const int status = cq3_ghlt_unrank_mask64(
            limbs, &support_low, &support_high
        );
        if (status != CQ3_GHLT_OK) {
            return status;
        }
        uint8_t *target = output + group * CQ3_GHLT_OUTPUT_BYTES;
        cq3_ghlt_store_le64(target, support_low);
        cq3_ghlt_store_le64(target + 8U, support_high);
        cq3_ghlt_store_le64(target + 16U, limbs[0]);
    }
    return cursor == required_bits ? CQ3_GHLT_OK : CQ3_GHLT_ERROR_SHAPE;
}
