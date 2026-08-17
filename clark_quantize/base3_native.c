#include <stdint.h>

#define CQ3_STATE_BITS 32U
#define CQ3_MASK UINT64_C(0xffffffff)
#define CQ3_HALF UINT64_C(0x80000000)
#define CQ3_QUARTER UINT64_C(0x40000000)
#define CQ3_THREE_QUARTER UINT64_C(0xc0000000)

static inline uint32_t cq3_read_bit(
    const uint8_t *payload,
    uint64_t bit_length,
    uint64_t *position
) {
    if (*position >= bit_length) {
        return 0U;
    }
    const uint64_t offset = *position;
    *position = offset + 1U;
    return (uint32_t)((payload[offset >> 3U] >> (7U - (offset & 7U))) & 1U);
}

typedef struct {
    uint8_t *payload;
    uint64_t capacity_bits;
    uint64_t position;
} cq3_bit_writer;

static inline int cq3_write_bit(cq3_bit_writer *writer, uint32_t bit) {
    if (writer->position >= writer->capacity_bits) {
        return -4;
    }
    const uint64_t offset = writer->position++;
    if (bit != 0U) {
        writer->payload[offset >> 3U] |=
            (uint8_t)(1U << (7U - (offset & 7U)));
    }
    return 0;
}

static int cq3_emit_static_bit(
    cq3_bit_writer *writer,
    uint32_t bit,
    uint64_t *pending
) {
    int status = cq3_write_bit(writer, bit);
    if (status != 0) {
        return status;
    }
    const uint32_t inverse = 1U - bit;
    while (*pending != 0U) {
        status = cq3_write_bit(writer, inverse);
        if (status != 0) {
            return status;
        }
        *pending -= 1U;
    }
    return 0;
}

int cq3_decode_uniform(
    const uint8_t *payload,
    uint64_t bit_length,
    uint64_t count,
    int8_t *output
) {
    uint64_t position = 0U;
    uint64_t low = 0U;
    uint64_t high = CQ3_MASK;
    uint64_t value = 0U;
    for (uint32_t index = 0U; index < CQ3_STATE_BITS; ++index) {
        value = ((value << 1U) & CQ3_MASK) | cq3_read_bit(payload, bit_length, &position);
    }
    for (uint64_t index = 0U; index < count; ++index) {
        const uint64_t width = high - low + 1U;
        const uint64_t scaled = ((value - low + 1U) * 3U - 1U) / width;
        const uint64_t symbol = scaled > 2U ? 2U : scaled;
        output[index] = (int8_t)symbol - 1;
        const uint64_t old_low = low;
        low = old_low + width * symbol / 3U;
        high = old_low + width * (symbol + 1U) / 3U - 1U;
        for (;;) {
            if (high < CQ3_HALF) {
                /* no offset adjustment */
            } else if (low >= CQ3_HALF) {
                low -= CQ3_HALF;
                high -= CQ3_HALF;
                value -= CQ3_HALF;
            } else if (low >= CQ3_QUARTER && high < CQ3_THREE_QUARTER) {
                low -= CQ3_QUARTER;
                high -= CQ3_QUARTER;
                value -= CQ3_QUARTER;
            } else {
                break;
            }
            low = (low << 1U) & CQ3_MASK;
            high = ((high << 1U) & CQ3_MASK) | 1U;
            value = ((value << 1U) & CQ3_MASK) | cq3_read_bit(payload, bit_length, &position);
        }
    }
    return 0;
}

int cq3_decode_static(
    const uint8_t *payload,
    uint64_t bit_length,
    uint64_t count,
    uint32_t frequency_minus,
    uint32_t frequency_zero,
    uint32_t frequency_plus,
    int8_t *output
) {
    const uint64_t cumulative_zero = (uint64_t)frequency_minus;
    const uint64_t cumulative_plus = cumulative_zero + (uint64_t)frequency_zero;
    const uint64_t total = cumulative_plus + (uint64_t)frequency_plus;
    if (total == 0U || total > 32768U) {
        return -1;
    }
    if ((bit_length != 0U && payload == 0) || (count != 0U && output == 0)) {
        return -2;
    }

    uint64_t position = 0U;
    uint64_t low = 0U;
    uint64_t high = CQ3_MASK;
    uint64_t value = 0U;
    for (uint32_t index = 0U; index < CQ3_STATE_BITS; ++index) {
        value = ((value << 1U) & CQ3_MASK) |
            cq3_read_bit(payload, bit_length, &position);
    }
    for (uint64_t index = 0U; index < count; ++index) {
        const uint64_t width = high - low + 1U;
        const uint64_t scaled = ((value - low + 1U) * total - 1U) / width;
        uint64_t lower;
        uint64_t upper;
        uint64_t symbol;
        if (scaled < cumulative_zero) {
            lower = 0U;
            upper = cumulative_zero;
            symbol = 0U;
        } else if (scaled < cumulative_plus) {
            lower = cumulative_zero;
            upper = cumulative_plus;
            symbol = 1U;
        } else {
            lower = cumulative_plus;
            upper = total;
            symbol = 2U;
        }
        if (lower == upper) {
            return -3;
        }
        output[index] = (int8_t)symbol - 1;
        const uint64_t old_low = low;
        low = old_low + width * lower / total;
        high = old_low + width * upper / total - 1U;
        for (;;) {
            if (high < CQ3_HALF) {
                /* no offset adjustment */
            } else if (low >= CQ3_HALF) {
                low -= CQ3_HALF;
                high -= CQ3_HALF;
                value -= CQ3_HALF;
            } else if (low >= CQ3_QUARTER && high < CQ3_THREE_QUARTER) {
                low -= CQ3_QUARTER;
                high -= CQ3_QUARTER;
                value -= CQ3_QUARTER;
            } else {
                break;
            }
            low = (low << 1U) & CQ3_MASK;
            high = ((high << 1U) & CQ3_MASK) | 1U;
            value = ((value << 1U) & CQ3_MASK) |
                cq3_read_bit(payload, bit_length, &position);
        }
    }
    return 0;
}

int cq3_encode_static(
    const int8_t *codes,
    uint64_t count,
    uint32_t frequency_minus,
    uint32_t frequency_zero,
    uint32_t frequency_plus,
    uint8_t *payload,
    uint64_t payload_capacity_bytes,
    uint64_t *bit_length
) {
    const uint64_t cumulative[4] = {
        0U,
        (uint64_t)frequency_minus,
        (uint64_t)frequency_minus + (uint64_t)frequency_zero,
        (uint64_t)frequency_minus + (uint64_t)frequency_zero +
            (uint64_t)frequency_plus,
    };
    const uint64_t total = cumulative[3];
    if (total == 0U || total > 32768U || bit_length == 0) {
        return -1;
    }
    if ((count != 0U && codes == 0) || payload == 0 ||
        payload_capacity_bytes > UINT64_MAX / 8U) {
        return -2;
    }
    cq3_bit_writer writer = {
        payload,
        payload_capacity_bytes * 8U,
        0U,
    };
    uint64_t low = 0U;
    uint64_t high = CQ3_MASK;
    uint64_t pending = 0U;
    for (uint64_t index = 0U; index < count; ++index) {
        const int32_t signed_symbol = (int32_t)codes[index] + 1;
        if (signed_symbol < 0 || signed_symbol > 2) {
            return -3;
        }
        const uint64_t symbol = (uint64_t)signed_symbol;
        if (cumulative[symbol] == cumulative[symbol + 1U]) {
            return -3;
        }
        const uint64_t width = high - low + 1U;
        const uint64_t old_low = low;
        low = old_low + width * cumulative[symbol] / total;
        high = old_low + width * cumulative[symbol + 1U] / total - 1U;
        for (;;) {
            int status;
            if (high < CQ3_HALF) {
                status = cq3_emit_static_bit(&writer, 0U, &pending);
            } else if (low >= CQ3_HALF) {
                status = cq3_emit_static_bit(&writer, 1U, &pending);
                low -= CQ3_HALF;
                high -= CQ3_HALF;
            } else if (low >= CQ3_QUARTER && high < CQ3_THREE_QUARTER) {
                pending += 1U;
                low -= CQ3_QUARTER;
                high -= CQ3_QUARTER;
                status = 0;
            } else {
                break;
            }
            if (status != 0) {
                return status;
            }
            low = (low << 1U) & CQ3_MASK;
            high = ((high << 1U) & CQ3_MASK) | 1U;
        }
    }
    pending += 1U;
    const int status = cq3_emit_static_bit(
        &writer,
        low < CQ3_QUARTER ? 0U : 1U,
        &pending
    );
    if (status != 0) {
        return status;
    }
    *bit_length = writer.position;
    return 0;
}
