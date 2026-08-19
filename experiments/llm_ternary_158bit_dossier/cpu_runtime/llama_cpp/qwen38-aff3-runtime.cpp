#include "qwen38-aff3-runtime.h"

#include "qwen38-aff3-kernels/qwen38_aff3_rotated_avx2_v47.cpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#if !defined(__AVX2__) || !defined(__F16C__) || !defined(__FMA__) || !defined(__SSSE3__)
#error "Qwen3.8 AFF3 runtime requires AVX2, F16C, FMA, and SSSE3"
#endif

namespace {

constexpr size_t kHeaderBytes = 256;
constexpr size_t kPayloadOffset = 4096;
constexpr size_t kRootOffset = 168;
constexpr size_t kRootBytes = 32;
constexpr int kImageGroup = 128;
constexpr int kRowsPerTile = 16;
constexpr size_t kX16GroupBytes = 528;
constexpr int kQ8Max = 127;
constexpr int kSectionCount = 7;
constexpr int kGraphTasks = 4;
constexpr std::array<char, 8> kMagic = {'A', 'F', 'F', '3', 'X', '1', '6', '1'};

enum Section : int {
    kWeights = 0,
    kAlphaAnchors = 1,
    kMuAnchors = 2,
    kAlphaCodebook = 3,
    kMuCodebook = 4,
    kAngles = 5,
    kPairings = 6,
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error("Qwen3.8 AFF3: " + message);
}

uint16_t read_u16(const uint8_t *pointer) {
    return static_cast<uint16_t>(pointer[0]) |
        static_cast<uint16_t>(pointer[1]) << 8U;
}

uint32_t read_u32(const uint8_t *pointer) {
    return static_cast<uint32_t>(pointer[0]) |
        static_cast<uint32_t>(pointer[1]) << 8U |
        static_cast<uint32_t>(pointer[2]) << 16U |
        static_cast<uint32_t>(pointer[3]) << 24U;
}

uint64_t read_u64(const uint8_t *pointer) {
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<uint64_t>(pointer[index]) << (8U * index);
    }
    return value;
}

bool finite_half(uint16_t value) {
    return (value & UINT16_C(0x7c00)) != UINT16_C(0x7c00);
}

uint32_t rotate_right(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
}

class Sha256 {
public:
    Sha256() : state_({
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19),
    }) {}

    void update(const void *content, size_t bytes) {
        const auto *source = static_cast<const uint8_t *>(content);
        total_bytes_ += bytes;
        while (bytes != 0) {
            const size_t take = std::min(bytes, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, source, take);
            buffered_ += take;
            source += take;
            bytes -= take;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bits = static_cast<uint64_t>(total_bytes_) * 8U;
        block_[buffered_++] = UINT8_C(0x80);
        if (buffered_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                      block_.end(), UINT8_C(0));
            transform(block_.data());
            buffered_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                  block_.begin() + 56, UINT8_C(0));
        for (int index = 0; index < 8; ++index) {
            block_[56 + index] = static_cast<uint8_t>(bits >> (56U - 8U * index));
        }
        transform(block_.data());
        std::array<uint8_t, 32> output {};
        for (size_t index = 0; index < state_.size(); ++index) {
            for (int byte = 0; byte < 4; ++byte) {
                output[index * 4 + byte] = static_cast<uint8_t>(
                    state_[index] >> (24U - 8U * byte));
            }
        }
        return output;
    }

private:
    void transform(const uint8_t *block) {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        uint32_t schedule[64];
        for (int index = 0; index < 16; ++index) {
            schedule[index] =
                static_cast<uint32_t>(block[index * 4]) << 24U |
                static_cast<uint32_t>(block[index * 4 + 1]) << 16U |
                static_cast<uint32_t>(block[index * 4 + 2]) << 8U |
                static_cast<uint32_t>(block[index * 4 + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const uint32_t left = schedule[index - 15];
            const uint32_t right = schedule[index - 2];
            const uint32_t sigma0 = rotate_right(left, 7) ^
                rotate_right(left, 18) ^ (left >> 3U);
            const uint32_t sigma1 = rotate_right(right, 17) ^
                rotate_right(right, 19) ^ (right >> 10U);
            schedule[index] = schedule[index - 16] + sigma0 +
                schedule[index - 7] + sigma1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (int index = 0; index < 64; ++index) {
            const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t first = h + sum1 + choice + constants[index] +
                schedule[index];
            const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t second = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ {};
    std::array<uint8_t, 64> block_ {};
    size_t buffered_ = 0;
    size_t total_bytes_ = 0;
};

class MappedFile {
public:
    explicit MappedFile(const std::string &path) {
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            fail("could not open image " + path + ": " + std::strerror(errno));
        }
        struct stat status {};
        if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            close(descriptor);
            fail("could not stat image " + path);
        }
        bytes_ = static_cast<size_t>(status.st_size);
        void *mapped = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        close(descriptor);
        if (mapped == MAP_FAILED) {
            fail("could not mmap image " + path);
        }
        data_ = static_cast<const uint8_t *>(mapped);
    }

    ~MappedFile() {
        if (data_ != nullptr) {
            munmap(const_cast<uint8_t *>(data_), bytes_);
        }
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    const uint8_t *data() const { return data_; }
    size_t bytes() const { return bytes_; }

private:
    const uint8_t *data_ = nullptr;
    size_t bytes_ = 0;
};

struct SectionRecord {
    size_t offset = 0;
    size_t bytes = 0;
};

struct ThreadScratch {
    explicit ThreadScratch(int columns)
        : rotated(static_cast<size_t>(columns)),
          quantized(static_cast<size_t>(columns)),
          scales(static_cast<size_t>(columns / kImageGroup)),
          sums(static_cast<size_t>(columns / kImageGroup)) {}

    std::vector<float> rotated;
    std::vector<int8_t> quantized;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

std::vector<int> parse_layers(const char *value, int n_layers) {
    if (value == nullptr || *value == '\0') {
        fail("QWEN38_AFF3_LAYERS is required when QWEN38_AFF3_DIR is set");
    }
    std::vector<int> layers;
    std::vector<uint8_t> seen(static_cast<size_t>(n_layers));
    const std::string text(value);
    size_t cursor = 0;
    while (cursor < text.size()) {
        const size_t comma = text.find(',', cursor);
        const std::string token = text.substr(
            cursor, comma == std::string::npos ? std::string::npos : comma - cursor);
        if (token.empty()) {
            fail("empty AFF3 layer token");
        }
        const size_t dash = token.find('-');
        auto parse_one = [&](const std::string &part) -> int {
            if (part.empty() || !std::all_of(part.begin(), part.end(),
                    [](unsigned char c) { return c >= '0' && c <= '9'; })) {
                fail("invalid AFF3 layer token " + token);
            }
            const long parsed = std::strtol(part.c_str(), nullptr, 10);
            if (parsed < 0 || parsed >= n_layers) {
                fail("AFF3 layer is outside the model");
            }
            return static_cast<int>(parsed);
        };
        const int first = parse_one(token.substr(0, dash));
        const int last = dash == std::string::npos
            ? first
            : parse_one(token.substr(dash + 1));
        if (dash != std::string::npos && token.find('-', dash + 1) != std::string::npos) {
            fail("AFF3 layer range contains multiple dashes");
        }
        if (last < first) {
            fail("AFF3 layer range is descending");
        }
        for (int layer = first; layer <= last; ++layer) {
            if (seen[layer] != 0) {
                fail("AFF3 layer declaration contains a duplicate");
            }
            seen[layer] = 1;
            layers.push_back(layer);
        }
        if (comma == std::string::npos) {
            break;
        }
        cursor = comma + 1;
    }
    if (layers.empty()) {
        fail("AFF3 layer set is empty");
    }
    std::sort(layers.begin(), layers.end());
    return layers;
}

}  // namespace

struct qwen38_aff3_image {
    qwen38_aff3_image(
            const std::string &path,
            int expected_layer,
            int expected_rows,
            int expected_columns)
        : mapping(path) {
        parse_and_authenticate(expected_layer, expected_rows, expected_columns);
    }

    const uint8_t *section_pointer(Section section) const {
        return mapping.data() + sections[section].offset;
    }

    const void *weights() const { return section_pointer(kWeights); }
    const uint16_t *alpha_anchors() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kAlphaAnchors));
    }
    const uint16_t *mu_anchors() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kMuAnchors));
    }
    const uint16_t *alpha_codebook() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kAlphaCodebook));
    }
    const uint16_t *mu_codebook() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kMuCodebook));
    }
    const uint16_t *angles() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kAngles));
    }
    const uint16_t *pairings() const {
        return reinterpret_cast<const uint16_t *>(section_pointer(kPairings));
    }
    int groups() const { return columns / kImageGroup; }

    void require_zero(size_t begin, size_t end, const char *label) const {
        if (begin > end || end > mapping.bytes()) {
            fail(std::string("invalid zero range: ") + label);
        }
        for (size_t index = begin; index < end; ++index) {
            if (mapping.data()[index] != 0) {
                fail(std::string("nonzero image padding: ") + label);
            }
        }
    }

    void verify_half_section(Section section) const {
        const SectionRecord &record = sections[section];
        for (size_t offset = 0; offset < record.bytes; offset += 2) {
            if (!finite_half(read_u16(section_pointer(section) + offset))) {
                fail("FP16 section contains a non-finite value");
            }
        }
    }

    void verify_weight_codes() const {
        const uint8_t *weight_bytes = section_pointer(kWeights);
        const size_t tile_groups = static_cast<size_t>(rows / kRowsPerTile) * groups();
        for (size_t group = 0; group < tile_groups; ++group) {
            const uint8_t *codes = weight_bytes + group * kX16GroupBytes + kRowsPerTile;
            for (size_t index = 0; index < 512; ++index) {
                const uint8_t value = codes[index];
                if ((value & 3U) == 3U || ((value >> 2U) & 3U) == 3U ||
                    ((value >> 4U) & 3U) == 3U || ((value >> 6U) & 3U) == 3U) {
                    fail("image contains reserved ternary code 3");
                }
            }
        }
    }

    void verify_pairings() const {
        const uint16_t *stored = pairings();
        std::vector<uint8_t> seen(static_cast<size_t>(columns));
        for (int stage = 0; stage < stages; ++stage) {
            std::fill(seen.begin(), seen.end(), UINT8_C(0));
            for (int index = 0; index < columns; ++index) {
                const uint16_t value = stored[stage * columns + index];
                if (value >= columns || seen[value] != 0) {
                    fail("stored pairing is not a perfect matching");
                }
                seen[value] = 1;
            }
        }
    }

    void parse_and_authenticate(
            int expected_layer,
            int expected_rows,
            int expected_columns) {
        if (mapping.bytes() < kPayloadOffset ||
            std::memcmp(mapping.data(), kMagic.data(), kMagic.size()) != 0 ||
            read_u16(mapping.data() + 8) != 1 ||
            read_u16(mapping.data() + 10) != 0 ||
            read_u32(mapping.data() + 12) != kHeaderBytes) {
            fail("runtime header identity changed");
        }
        layer = static_cast<int>(read_u32(mapping.data() + 16));
        rows = static_cast<int>(read_u32(mapping.data() + 20));
        columns = static_cast<int>(read_u32(mapping.data() + 24));
        const int group_size = static_cast<int>(read_u32(mapping.data() + 28));
        stages = static_cast<int>(read_u32(mapping.data() + 32));
        pairs = static_cast<int>(read_u32(mapping.data() + 36));
        const uint64_t payload_offset = read_u64(mapping.data() + 40);
        const uint64_t payload_bytes = read_u64(mapping.data() + 48);
        if (layer != expected_layer || rows != expected_rows ||
            columns != expected_columns || rows <= 0 ||
            rows % kRowsPerTile != 0 || columns <= 0 ||
            columns % kImageGroup != 0 || group_size != kImageGroup ||
            stages <= 0 || pairs * 2 != columns ||
            payload_offset != kPayloadOffset ||
            payload_bytes != mapping.bytes() - kPayloadOffset) {
            fail("runtime layer or dimensions changed");
        }
        size_t cursor = 56;
        for (SectionRecord &record : sections) {
            const uint64_t offset = read_u64(mapping.data() + cursor);
            const uint64_t bytes = read_u64(mapping.data() + cursor + 8);
            cursor += 16;
            if (offset > std::numeric_limits<size_t>::max() ||
                bytes > std::numeric_limits<size_t>::max()) {
                fail("section exceeds host address space");
            }
            record = {static_cast<size_t>(offset), static_cast<size_t>(bytes)};
        }
        if (cursor != kRootOffset) {
            fail("header parser changed");
        }
        require_zero(kHeaderBytes, kPayloadOffset, "header-to-payload");
        size_t previous = kPayloadOffset;
        for (const SectionRecord &record : sections) {
            if (record.offset % 64 != 0 || record.offset < previous ||
                record.bytes == 0 || record.offset > mapping.bytes() ||
                record.bytes > mapping.bytes() - record.offset) {
                fail("invalid section bounds");
            }
            require_zero(previous, record.offset, "section alignment");
            previous = record.offset + record.bytes;
        }
        if (previous != mapping.bytes()) {
            fail("runtime image has trailing bytes");
        }
        const std::array<size_t, kSectionCount> expected = {
            static_cast<size_t>(rows / kRowsPerTile) * groups() * kX16GroupBytes,
            static_cast<size_t>(rows) * 2,
            static_cast<size_t>(rows) * 2,
            32,
            32,
            static_cast<size_t>(stages) * pairs * 2,
            static_cast<size_t>(stages) * pairs * 4,
        };
        for (int index = 0; index < kSectionCount; ++index) {
            if (sections[index].bytes != expected[index]) {
                fail("runtime section size changed");
            }
        }
        for (size_t index = kRootOffset + kRootBytes; index < kHeaderBytes; ++index) {
            if (mapping.data()[index] != 0) {
                fail("reserved header bytes are nonzero");
            }
        }
        std::array<uint8_t, kHeaderBytes> zero_header {};
        std::memcpy(zero_header.data(), mapping.data(), kHeaderBytes);
        std::fill(zero_header.begin() + kRootOffset,
                  zero_header.begin() + kRootOffset + kRootBytes, UINT8_C(0));
        Sha256 digest;
        digest.update(zero_header.data(), zero_header.size());
        digest.update(mapping.data() + kPayloadOffset,
                      mapping.bytes() - kPayloadOffset);
        const std::array<uint8_t, kRootBytes> observed = digest.finish();
        if (std::memcmp(observed.data(), mapping.data() + kRootOffset,
                        kRootBytes) != 0) {
            fail("runtime root SHA-256 mismatch");
        }
        verify_weight_codes();
        verify_half_section(kAlphaAnchors);
        verify_half_section(kMuAnchors);
        verify_half_section(kAlphaCodebook);
        verify_half_section(kMuCodebook);
        verify_half_section(kAngles);
        verify_pairings();
        const size_t coefficient_count = static_cast<size_t>(stages) * pairs;
        cosines.resize(coefficient_count);
        sines.resize(coefficient_count);
        for (size_t index = 0; index < coefficient_count; ++index) {
            const float theta = half_to_float(angles()[index]);
            cosines[index] = std::cos(theta);
            sines[index] = std::sin(theta);
        }
    }

    MappedFile mapping;
    std::array<SectionRecord, kSectionCount> sections {};
    int layer = -1;
    int rows = 0;
    int columns = 0;
    int stages = 0;
    int pairs = 0;
    std::vector<float> cosines;
    std::vector<float> sines;
};

struct qwen38_aff3_node_state {
    explicit qwen38_aff3_node_state(qwen38_aff3_image_ptr image_value)
        : image(std::move(image_value)) {
        for (int task = 0; task < kGraphTasks; ++task) {
            scratch.emplace_back(image->columns);
        }
    }

    qwen38_aff3_image_ptr image;
    std::vector<ThreadScratch> scratch;
#ifdef QWEN38_AFF3_DEVELOPMENT_TESTING
    std::atomic<uint32_t> observed_task_mask {0};
#endif
};

namespace {

int rotate_quantize_precomputed(
        const qwen38_aff3_image &image,
        const float *input,
        ThreadScratch &scratch) {
    std::memcpy(scratch.rotated.data(), input,
                static_cast<size_t>(image.columns) * sizeof(float));
    const uint16_t *pairings = image.pairings();
    for (int stage = 0; stage < image.stages; ++stage) {
        const int stage_pair = stage * image.pairs;
        const int stage_index = stage * image.columns;
        for (int pair = 0; pair < image.pairs; ++pair) {
            const uint16_t first = pairings[stage_index + pair * 2];
            const uint16_t second = pairings[stage_index + pair * 2 + 1];
            const float cosine = image.cosines[stage_pair + pair];
            const float sine = image.sines[stage_pair + pair];
            const float left = scratch.rotated[first];
            const float right = scratch.rotated[second];
            scratch.rotated[first] = cosine * left - sine * right;
            scratch.rotated[second] = sine * left + cosine * right;
        }
    }
    for (int group = 0; group < image.groups(); ++group) {
        const int start = group * kImageGroup;
        float maximum = 0.0f;
        for (int offset = 0; offset < kImageGroup; ++offset) {
            const float value = scratch.rotated[start + offset];
            if (!std::isfinite(value)) {
                return 4;
            }
            maximum = std::max(maximum, std::abs(value));
        }
        if (maximum == 0.0f) {
            std::memset(scratch.quantized.data() + start, 0, kImageGroup);
            scratch.scales[group] = 0.0f;
            scratch.sums[group] = 0;
            continue;
        }
        const float scale = maximum / static_cast<float>(kQ8Max);
        const float inverse = 1.0f / scale;
        int32_t sum = 0;
        for (int offset = 0; offset < kImageGroup; ++offset) {
            const long rounded = std::lrintf(
                scratch.rotated[start + offset] * inverse);
            const int value = std::max(
                -kQ8Max, std::min(kQ8Max, static_cast<int>(rounded)));
            scratch.quantized[start + offset] = static_cast<int8_t>(value);
            sum += value;
        }
        scratch.scales[group] = scale;
        scratch.sums[group] = sum;
    }
    return 0;
}

int gemv_rows(
        const qwen38_aff3_image &image,
        const ThreadScratch &scratch,
        int row_begin,
        int row_count,
        float *output) {
    if (row_begin < 0 || row_count <= 0 || row_begin % kRowsPerTile != 0 ||
        row_count % kRowsPerTile != 0 || row_begin + row_count > image.rows) {
        return 20;
    }
    const size_t weight_offset = static_cast<size_t>(row_begin / kRowsPerTile) *
        image.groups() * kX16GroupBytes;
    return cq3_aff3_gemv_x16_register_v18(
        static_cast<const uint8_t *>(image.weights()) + weight_offset,
        image.alpha_anchors() + row_begin,
        image.mu_anchors() + row_begin,
        image.alpha_codebook(),
        image.mu_codebook(),
        scratch.quantized.data(),
        scratch.scales.data(),
        scratch.sums.data(),
        row_count,
        image.columns,
        output);
}

[[noreturn]] void callback_abort(const char *message) {
    std::fprintf(stderr, "Qwen3.8 AFF3 callback failure: %s\n", message);
    std::abort();
}

void graph_callback(
        ggml_tensor *dst,
        const ggml_tensor *shape_anchor,
        const ggml_tensor *input,
        int ith,
        int nth,
        void *opaque) {
    auto *state = static_cast<qwen38_aff3_node_state *>(opaque);
    if (state == nullptr || state->image == nullptr || ith < 0 ||
        ith >= nth || ith >= static_cast<int>(state->scratch.size())) {
        callback_abort("invalid task identity");
    }
    const qwen38_aff3_image &image = *state->image;
#ifdef QWEN38_AFF3_DEVELOPMENT_TESTING
    state->observed_task_mask.fetch_or(
        UINT32_C(1) << static_cast<uint32_t>(ith),
        std::memory_order_relaxed);
#endif
    if (dst->type != GGML_TYPE_F32 || shape_anchor->type != GGML_TYPE_F32 ||
        input->type != GGML_TYPE_F32 || dst->ne[0] != image.rows ||
        shape_anchor->ne[0] != image.rows || input->ne[0] != image.columns ||
        dst->ne[1] != input->ne[1] || shape_anchor->ne[1] != input->ne[1] ||
        dst->ne[2] != 1 || input->ne[2] != 1 || shape_anchor->ne[2] != 1) {
        callback_abort("tensor contract changed");
    }
    const int tokens = static_cast<int>(input->ne[1]);
    const int tiles = image.rows / kRowsPerTile;
    const int64_t total_jobs = static_cast<int64_t>(tokens) * tiles;
    const int64_t job_begin = total_jobs * ith / nth;
    const int64_t job_end = total_jobs * (ith + 1) / nth;
    if (job_end <= job_begin) {
        return;
    }
    ThreadScratch &scratch = state->scratch[ith];
    const int first_token = static_cast<int>(job_begin / tiles);
    const int last_token = static_cast<int>((job_end - 1) / tiles);
    for (int token = first_token; token <= last_token; ++token) {
        const int64_t token_job_begin = static_cast<int64_t>(token) * tiles;
        const int tile_begin = static_cast<int>(
            std::max(job_begin, token_job_begin) - token_job_begin);
        const int tile_end = static_cast<int>(
            std::min(job_end, token_job_begin + tiles) - token_job_begin);
        if (tile_end <= tile_begin) {
            continue;
        }
        const auto *input_row = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(input->data) + token * input->nb[1]);
        auto *output_row = reinterpret_cast<float *>(
            static_cast<uint8_t *>(dst->data) + token * dst->nb[1]);
        int status = rotate_quantize_precomputed(image, input_row, scratch);
        if (status == 0) {
            const int row_begin = tile_begin * kRowsPerTile;
            const int row_count = (tile_end - tile_begin) * kRowsPerTile;
            status = gemv_rows(
                image, scratch, row_begin, row_count, output_row + row_begin);
        }
        if (status != 0) {
            callback_abort("native projection rejected its inputs");
        }
    }
}

}  // namespace

bool qwen38_aff3_environment_requested() {
    const char *directory = std::getenv("QWEN38_AFF3_DIR");
    const char *layers = std::getenv("QWEN38_AFF3_LAYERS");
    return (directory != nullptr && *directory != '\0') ||
        (layers != nullptr && *layers != '\0');
}

std::vector<qwen38_aff3_image_ptr> qwen38_aff3_load_from_environment(
        int n_layers,
        int expected_rows,
        int expected_columns) {
    const char *directory = std::getenv("QWEN38_AFF3_DIR");
    const char *layer_text = std::getenv("QWEN38_AFF3_LAYERS");
    const bool has_directory = directory != nullptr && *directory != '\0';
    const bool has_layers = layer_text != nullptr && *layer_text != '\0';
    if (!has_directory && !has_layers) {
        return std::vector<qwen38_aff3_image_ptr>(static_cast<size_t>(n_layers));
    }
    if (!has_directory || !has_layers || n_layers <= 0 ||
        expected_rows <= 0 || expected_columns <= 0) {
        fail("both QWEN38_AFF3_DIR and QWEN38_AFF3_LAYERS are required");
    }
    std::vector<qwen38_aff3_image_ptr> images(static_cast<size_t>(n_layers));
    for (int layer : parse_layers(layer_text, n_layers)) {
        char name[64];
        const int written = std::snprintf(
            name, sizeof(name), "layer-%02d.aff3x16", layer);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
            fail("AFF3 layer filename overflow");
        }
        std::string path(directory);
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        path += name;
        images[layer] = std::make_shared<const qwen38_aff3_image>(
            path, layer, expected_rows, expected_columns);
    }
    return images;
}

bool qwen38_aff3_any_enabled(
        const std::vector<qwen38_aff3_image_ptr> &images) {
    return std::any_of(images.begin(), images.end(),
        [](const qwen38_aff3_image_ptr &image) { return image != nullptr; });
}

ggml_tensor * qwen38_aff3_build_projection(
        ggml_context *ctx,
        ggml_tensor *input,
        const qwen38_aff3_image_ptr &image,
        std::vector<qwen38_aff3_node_state_ptr> &node_states) {
    if (ctx == nullptr || input == nullptr || image == nullptr ||
        input->ne[0] != image->columns || input->type != GGML_TYPE_F32) {
        fail("graph projection input contract changed");
    }
    const int64_t elements = ggml_nelements(input);
    if (elements <= 0 || elements % image->columns != 0) {
        fail("graph projection token shape changed");
    }
    const int64_t tokens = elements / image->columns;
    ggml_tensor *flat_input = ggml_cont_2d(ctx, input, image->columns, tokens);
    ggml_tensor *shape_anchor = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, image->rows, tokens);
    auto state = std::make_shared<qwen38_aff3_node_state>(image);
    ggml_tensor *output = ggml_map_custom2(
        ctx,
        shape_anchor,
        flat_input,
        graph_callback,
        kGraphTasks,
        state.get());
    ggml_set_name(shape_anchor, "qwen38_aff3_shape_anchor");
    ggml_set_name(flat_input, "qwen38_aff3_input");
    ggml_set_name(output, "qwen38_aff3_output");
    node_states.push_back(std::move(state));
    return output;
}
