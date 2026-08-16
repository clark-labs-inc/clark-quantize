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
