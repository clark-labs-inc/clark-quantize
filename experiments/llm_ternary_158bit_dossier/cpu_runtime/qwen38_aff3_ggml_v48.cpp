// Authenticated mmap AFF3 image -> fused four-task GGML CPU node for
// Qwen/Qwen3.8-27B.  This development harness intentionally exposes only a
// correctness mode; performance requires a separately frozen protocol.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "kernels/qwen38_aff3_rotated_avx2_v47.cpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

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
    throw std::runtime_error(message);
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
    Sha256() {
        state_ = {
            UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
            UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
            UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
            UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19),
        };
    }

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

std::string hex(const uint8_t *content, size_t bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output(bytes * 2, '0');
    for (size_t index = 0; index < bytes; ++index) {
        output[index * 2] = digits[content[index] >> 4U];
        output[index * 2 + 1] = digits[content[index] & 15U];
    }
    return output;
}

class MappedFile {
public:
    explicit MappedFile(const std::string &path) {
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            fail("could not open AFF3 image: " + path);
        }
        struct stat status {};
        if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            close(descriptor);
            fail("could not stat AFF3 image");
        }
        bytes_ = static_cast<size_t>(status.st_size);
        void *mapped = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        close(descriptor);
        if (mapped == MAP_FAILED) {
            fail("could not mmap AFF3 image");
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

class Aff3Image {
public:
    explicit Aff3Image(const std::string &path) : mapping_(path) {
        parse_and_authenticate();
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
    size_t file_bytes() const { return mapping_.bytes(); }
    int layer() const { return layer_; }
    int rows() const { return rows_; }
    int columns() const { return columns_; }
    int groups() const { return columns_ / kImageGroup; }
    int stages() const { return stages_; }
    int pairs() const { return pairs_; }
    const std::string &root_sha256() const { return root_sha256_; }
    const std::vector<float> &cosines() const { return cosines_; }
    const std::vector<float> &sines() const { return sines_; }

private:
    const uint8_t *section_pointer(Section section) const {
        return mapping_.data() + sections_[section].offset;
    }

    void require_zero(size_t begin, size_t end, const char *label) const {
        if (begin > end || end > mapping_.bytes()) {
            fail(std::string("invalid zero range: ") + label);
        }
        for (size_t index = begin; index < end; ++index) {
            if (mapping_.data()[index] != 0) {
                fail(std::string("nonzero AFF3 padding: ") + label);
            }
        }
    }

    void verify_half_section(Section section) const {
        const SectionRecord &record = sections_[section];
        for (size_t offset = 0; offset < record.bytes; offset += 2) {
            if (!finite_half(read_u16(section_pointer(section) + offset))) {
                fail("AFF3 FP16 section contains a non-finite value");
            }
        }
    }

    void verify_weight_codes() const {
        const uint8_t *weights_bytes = section_pointer(kWeights);
        const size_t tile_groups = static_cast<size_t>(rows_ / kRowsPerTile) * groups();
        for (size_t group = 0; group < tile_groups; ++group) {
            const uint8_t *codes = weights_bytes + group * kX16GroupBytes + kRowsPerTile;
            for (size_t index = 0; index < 512; ++index) {
                const uint8_t value = codes[index];
                if ((value & 3U) == 3U || ((value >> 2U) & 3U) == 3U ||
                    ((value >> 4U) & 3U) == 3U || ((value >> 6U) & 3U) == 3U) {
                    fail("AFF3 image contains reserved ternary code 3");
                }
            }
        }
    }

    void verify_pairings() const {
        const uint16_t *stored = pairings();
        std::vector<uint8_t> seen(static_cast<size_t>(columns_));
        for (int stage = 0; stage < stages_; ++stage) {
            std::fill(seen.begin(), seen.end(), UINT8_C(0));
            for (int index = 0; index < columns_; ++index) {
                const uint16_t value = stored[stage * columns_ + index];
                if (value >= columns_ || seen[value] != 0) {
                    fail("AFF3 pairing is not a perfect matching");
                }
                seen[value] = 1;
            }
        }
    }

    void parse_and_authenticate() {
        if (mapping_.bytes() < kPayloadOffset ||
            std::memcmp(mapping_.data(), kMagic.data(), kMagic.size()) != 0 ||
            read_u16(mapping_.data() + 8) != 1 ||
            read_u16(mapping_.data() + 10) != 0 ||
            read_u32(mapping_.data() + 12) != kHeaderBytes) {
            fail("AFF3 runtime header identity changed");
        }
        layer_ = static_cast<int>(read_u32(mapping_.data() + 16));
        rows_ = static_cast<int>(read_u32(mapping_.data() + 20));
        columns_ = static_cast<int>(read_u32(mapping_.data() + 24));
        const int group_size = static_cast<int>(read_u32(mapping_.data() + 28));
        stages_ = static_cast<int>(read_u32(mapping_.data() + 32));
        pairs_ = static_cast<int>(read_u32(mapping_.data() + 36));
        const uint64_t payload_offset = read_u64(mapping_.data() + 40);
        const uint64_t payload_bytes = read_u64(mapping_.data() + 48);
        if (layer_ < 0 || layer_ >= 64 || rows_ <= 0 ||
            rows_ % kRowsPerTile != 0 || columns_ <= 0 ||
            columns_ % kImageGroup != 0 || group_size != kImageGroup ||
            stages_ <= 0 || pairs_ * 2 != columns_ ||
            payload_offset != kPayloadOffset ||
            payload_bytes != mapping_.bytes() - kPayloadOffset) {
            fail("AFF3 runtime dimensions changed");
        }
        size_t cursor = 56;
        for (SectionRecord &record : sections_) {
            const uint64_t offset = read_u64(mapping_.data() + cursor);
            const uint64_t bytes = read_u64(mapping_.data() + cursor + 8);
            cursor += 16;
            if (offset > std::numeric_limits<size_t>::max() ||
                bytes > std::numeric_limits<size_t>::max()) {
                fail("AFF3 section size exceeds host address space");
            }
            record = {static_cast<size_t>(offset), static_cast<size_t>(bytes)};
        }
        if (cursor != kRootOffset) {
            fail("AFF3 header parser changed");
        }
        require_zero(kHeaderBytes, kPayloadOffset, "header-to-payload");
        size_t previous = kPayloadOffset;
        for (const SectionRecord &record : sections_) {
            if (record.offset % 64 != 0 || record.offset < previous ||
                record.bytes == 0 || record.offset > mapping_.bytes() ||
                record.bytes > mapping_.bytes() - record.offset) {
                fail("invalid AFF3 section bounds");
            }
            require_zero(previous, record.offset, "section alignment");
            previous = record.offset + record.bytes;
        }
        if (previous != mapping_.bytes()) {
            fail("AFF3 runtime image has trailing bytes");
        }
        const std::array<size_t, kSectionCount> expected = {
            static_cast<size_t>(rows_ / kRowsPerTile) * groups() * kX16GroupBytes,
            static_cast<size_t>(rows_) * 2,
            static_cast<size_t>(rows_) * 2,
            32,
            32,
            static_cast<size_t>(stages_) * pairs_ * 2,
            static_cast<size_t>(stages_) * pairs_ * 4,
        };
        for (int index = 0; index < kSectionCount; ++index) {
            if (sections_[index].bytes != expected[index]) {
                fail("AFF3 runtime section size changed");
            }
        }
        for (size_t index = kRootOffset + kRootBytes; index < kHeaderBytes; ++index) {
            if (mapping_.data()[index] != 0) {
                fail("AFF3 reserved header bytes are nonzero");
            }
        }
        std::array<uint8_t, kHeaderBytes> zero_header {};
        std::memcpy(zero_header.data(), mapping_.data(), kHeaderBytes);
        std::fill(zero_header.begin() + kRootOffset,
                  zero_header.begin() + kRootOffset + kRootBytes, UINT8_C(0));
        Sha256 digest;
        digest.update(zero_header.data(), zero_header.size());
        digest.update(mapping_.data() + kPayloadOffset,
                      mapping_.bytes() - kPayloadOffset);
        const std::array<uint8_t, kRootBytes> observed = digest.finish();
        if (std::memcmp(observed.data(), mapping_.data() + kRootOffset,
                        kRootBytes) != 0) {
            fail("AFF3 runtime root SHA-256 mismatch");
        }
        root_sha256_ = hex(observed.data(), observed.size());
        verify_weight_codes();
        verify_half_section(kAlphaAnchors);
        verify_half_section(kMuAnchors);
        verify_half_section(kAlphaCodebook);
        verify_half_section(kMuCodebook);
        verify_half_section(kAngles);
        verify_pairings();

        const size_t coefficients = static_cast<size_t>(stages_) * pairs_;
        cosines_.resize(coefficients);
        sines_.resize(coefficients);
        for (size_t index = 0; index < coefficients; ++index) {
            const float theta = half_to_float(angles()[index]);
            cosines_[index] = std::cos(theta);
            sines_[index] = std::sin(theta);
        }
    }

    MappedFile mapping_;
    std::array<SectionRecord, kSectionCount> sections_ {};
    int layer_ = -1;
    int rows_ = 0;
    int columns_ = 0;
    int stages_ = 0;
    int pairs_ = 0;
    std::string root_sha256_;
    std::vector<float> cosines_;
    std::vector<float> sines_;
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

int rotate_quantize_precomputed(
        const Aff3Image &image,
        const float *input,
        ThreadScratch &scratch) {
    std::memcpy(scratch.rotated.data(), input,
                static_cast<size_t>(image.columns()) * sizeof(float));
    const uint16_t *pairings = image.pairings();
    for (int stage = 0; stage < image.stages(); ++stage) {
        const int stage_pair = stage * image.pairs();
        const int stage_index = stage * image.columns();
        for (int pair = 0; pair < image.pairs(); ++pair) {
            const uint16_t first = pairings[stage_index + pair * 2];
            const uint16_t second = pairings[stage_index + pair * 2 + 1];
            const float cosine = image.cosines()[stage_pair + pair];
            const float sine = image.sines()[stage_pair + pair];
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
        const Aff3Image &image,
        const ThreadScratch &scratch,
        int row_begin,
        int row_count,
        float *output) {
    if (row_begin < 0 || row_count <= 0 || row_begin % kRowsPerTile != 0 ||
        row_count % kRowsPerTile != 0 || row_begin + row_count > image.rows()) {
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
        image.columns(),
        output);
}

struct GraphUserData {
    const Aff3Image *image = nullptr;
    std::vector<ThreadScratch> scratch;
    std::atomic<int> status {0};
    std::atomic<uint32_t> task_mask {0};
};

void graph_callback(
        ggml_tensor *dst,
        const ggml_tensor *shape_anchor,
        const ggml_tensor *input,
        int ith,
        int nth,
        void *opaque) {
    auto *userdata = static_cast<GraphUserData *>(opaque);
    if (userdata == nullptr || userdata->image == nullptr || ith < 0 ||
        ith >= nth || ith >= static_cast<int>(userdata->scratch.size())) {
        if (userdata != nullptr) {
            userdata->status.store(30, std::memory_order_relaxed);
        }
        return;
    }
    const Aff3Image &image = *userdata->image;
    if (dst->type != GGML_TYPE_F32 || shape_anchor->type != GGML_TYPE_F32 ||
        input->type != GGML_TYPE_F32 || dst->ne[0] != image.rows() ||
        shape_anchor->ne[0] != image.rows() || input->ne[0] != image.columns() ||
        dst->ne[1] != input->ne[1] || shape_anchor->ne[1] != input->ne[1]) {
        userdata->status.store(31, std::memory_order_relaxed);
        return;
    }
    userdata->task_mask.fetch_or(UINT32_C(1) << ith, std::memory_order_relaxed);
    ThreadScratch &scratch = userdata->scratch[ith];
    const int tokens = static_cast<int>(input->ne[1]);
    if (tokens >= nth) {
        for (int token = ith; token < tokens; token += nth) {
            const auto *input_row = reinterpret_cast<const float *>(
                static_cast<const uint8_t *>(input->data) + token * input->nb[1]);
            auto *output_row = reinterpret_cast<float *>(
                static_cast<uint8_t *>(dst->data) + token * dst->nb[1]);
            int status = rotate_quantize_precomputed(image, input_row, scratch);
            if (status == 0) {
                status = gemv_rows(image, scratch, 0, image.rows(), output_row);
            }
            if (status != 0) {
                userdata->status.store(status, std::memory_order_relaxed);
                return;
            }
        }
        return;
    }
    if (tokens != 1) {
        userdata->status.store(32, std::memory_order_relaxed);
        return;
    }
    const auto *input_row = static_cast<const float *>(input->data);
    int status = rotate_quantize_precomputed(image, input_row, scratch);
    const int tiles = image.rows() / kRowsPerTile;
    const int tile_begin = tiles * ith / nth;
    const int tile_end = tiles * (ith + 1) / nth;
    if (status == 0 && tile_end > tile_begin) {
        const int row_begin = tile_begin * kRowsPerTile;
        const int row_count = (tile_end - tile_begin) * kRowsPerTile;
        status = gemv_rows(
            image, scratch, row_begin, row_count,
            static_cast<float *>(dst->data) + row_begin);
    }
    if (status != 0) {
        userdata->status.store(status, std::memory_order_relaxed);
    }
}

struct GraphRuntime {
    ggml_context *context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *input = nullptr;
    ggml_tensor *anchor = nullptr;
    ggml_tensor *output = nullptr;
    GraphUserData userdata;

    ~GraphRuntime() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
    }

    GraphRuntime() = default;
    GraphRuntime(const GraphRuntime &) = delete;
    GraphRuntime &operator=(const GraphRuntime &) = delete;
};

std::unique_ptr<GraphRuntime> make_graph(
        const Aff3Image &image,
        int tokens,
        int threads) {
    auto runtime = std::make_unique<GraphRuntime>();
    runtime->userdata.image = &image;
    for (int thread = 0; thread < threads; ++thread) {
        runtime->userdata.scratch.emplace_back(image.columns());
    }
    const size_t metadata_bytes = 16U * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(16, false) + 65536U;
    const ggml_init_params params = {
        /*.mem_size =*/ metadata_bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    runtime->context = ggml_init(params);
    if (runtime->context == nullptr) {
        fail("ggml context initialization failed");
    }
    runtime->backend = ggml_backend_cpu_init();
    if (runtime->backend == nullptr) {
        fail("ggml CPU backend initialization failed");
    }
    ggml_backend_cpu_set_n_threads(runtime->backend, threads);
    runtime->anchor = ggml_new_tensor_2d(
        runtime->context, GGML_TYPE_F32, image.rows(), tokens);
    runtime->input = ggml_new_tensor_2d(
        runtime->context, GGML_TYPE_F32, image.columns(), tokens);
    runtime->output = ggml_map_custom2(
        runtime->context,
        runtime->anchor,
        runtime->input,
        graph_callback,
        threads,
        &runtime->userdata);
    ggml_set_name(runtime->anchor, "aff3_shape_anchor");
    ggml_set_name(runtime->input, "aff3_live_activation");
    ggml_set_name(runtime->output, "aff3_rotated_projection");
    runtime->buffer = ggml_backend_alloc_ctx_tensors(
        runtime->context, runtime->backend);
    if (runtime->buffer == nullptr) {
        fail("ggml tensor allocation failed");
    }
    runtime->graph = ggml_new_graph_custom(runtime->context, 16, false);
    if (runtime->graph == nullptr) {
        fail("ggml graph allocation failed");
    }
    ggml_build_forward_expand(runtime->graph, runtime->output);
    if (!ggml_backend_supports_op(runtime->backend, runtime->output)) {
        fail("ggml CPU backend rejected AFF3 custom op");
    }
    return runtime;
}

uint32_t compute_graph(GraphRuntime &runtime) {
    runtime.userdata.status.store(0, std::memory_order_relaxed);
    runtime.userdata.task_mask.store(0, std::memory_order_relaxed);
    const ggml_status status = ggml_backend_graph_compute(
        runtime.backend, runtime.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("ggml graph compute failed: ") +
             ggml_status_to_string(status));
    }
    const int callback_status = runtime.userdata.status.load(
        std::memory_order_relaxed);
    if (callback_status != 0) {
        fail("AFF3 graph callback failed with status " +
             std::to_string(callback_status));
    }
    return runtime.userdata.task_mask.load(std::memory_order_relaxed);
}

void compute_direct(
        const Aff3Image &image,
        const std::vector<float> &input,
        int tokens,
        std::vector<float> &output) {
    output.resize(static_cast<size_t>(tokens) * image.rows());
    ThreadScratch scratch(image.columns());
    for (int token = 0; token < tokens; ++token) {
        const int status = cq3_aff3_rotated_gemv_x16_v47(
            image.weights(),
            image.alpha_anchors(),
            image.mu_anchors(),
            image.alpha_codebook(),
            image.mu_codebook(),
            image.angles(),
            image.pairings(),
            input.data() + static_cast<size_t>(token) * image.columns(),
            image.rows(),
            image.columns(),
            image.stages(),
            scratch.rotated.data(),
            scratch.quantized.data(),
            scratch.scales.data(),
            scratch.sums.data(),
            output.data() + static_cast<size_t>(token) * image.rows());
        if (status != 0) {
            fail("direct v47 kernel failed with status " + std::to_string(status));
        }
    }
}

uint64_t splitmix64(uint64_t &state) {
    uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

float deterministic_float(uint64_t &state) {
    const uint32_t bits = static_cast<uint32_t>(splitmix64(state) >> 40U);
    return static_cast<float>(bits) * (2.0f / 16777215.0f) - 1.0f;
}

struct Options {
    std::string image_path;
    int tokens = 0;
    int threads = 0;
    uint64_t seed = 20260819;
    bool validate_only = false;
};

int parse_positive(const char *value, const char *name) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        fail(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

uint64_t parse_u64(const char *value, const char *name) {
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        fail(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(parsed);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take = [&](const char *name) -> const char * {
            if (index + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (argument == "--image") {
            options.image_path = take("--image");
        } else if (argument == "--tokens") {
            options.tokens = parse_positive(take("--tokens"), "--tokens");
        } else if (argument == "--threads") {
            options.threads = parse_positive(take("--threads"), "--threads");
        } else if (argument == "--seed") {
            options.seed = parse_u64(take("--seed"), "--seed");
        } else if (argument == "--validate-only") {
            options.validate_only = true;
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.image_path.empty() ||
        (options.tokens != 1 && options.tokens != 8) ||
        options.threads != 4 || !options.validate_only) {
        fail("v48 development mode requires --image, --tokens 1|8, "
             "--threads 4, and --validate-only; timing is forbidden");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const Aff3Image image(options.image_path);
        uint64_t random_state = options.seed;
        std::vector<float> input(
            static_cast<size_t>(options.tokens) * image.columns());
        for (float &value : input) {
            value = deterministic_float(random_state);
        }
        std::vector<float> direct;
        compute_direct(image, input, options.tokens, direct);
        auto graph = make_graph(image, options.tokens, options.threads);
        ggml_backend_tensor_set(
            graph->input, input.data(), 0, input.size() * sizeof(float));
        const uint32_t task_mask = compute_graph(*graph);
        std::vector<float> observed(direct.size());
        ggml_backend_tensor_get(
            graph->output, observed.data(), 0,
            observed.size() * sizeof(float));
        double maximum_error = 0.0;
        for (size_t index = 0; index < direct.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(
                static_cast<double>(direct[index]) - observed[index]));
        }
        const bool bit_exact = std::memcmp(
            direct.data(), observed.data(), direct.size() * sizeof(float)) == 0;
        if (!bit_exact || maximum_error != 0.0 ||
            task_mask != ((UINT32_C(1) << options.threads) - 1U)) {
            fail("v48 development correctness gate failed");
        }
        const size_t scratch_bytes =
            static_cast<size_t>(image.columns()) * (sizeof(float) + sizeof(int8_t)) +
            static_cast<size_t>(image.groups()) * (sizeof(float) + sizeof(int32_t));
        const size_t coefficient_bytes = static_cast<size_t>(image.stages()) *
            image.pairs() * 2 * sizeof(float);
        std::cout
            << "{\"schema\":\"qwen38_27b_aff3_ggml_development_case_v48\""
            << ",\"model\":\"Qwen/Qwen3.8-27B\""
            << ",\"layer\":" << image.layer()
            << ",\"shape\":[" << image.rows() << ',' << image.columns() << ']'
            << ",\"stages\":" << image.stages()
            << ",\"tokens\":" << options.tokens
            << ",\"threads\":" << options.threads
            << ",\"observed_task_mask\":" << task_mask
            << ",\"root_sha256\":\"" << image.root_sha256() << "\""
            << ",\"runtime_image_bytes\":" << image.file_bytes()
            << ",\"scratch_bytes_per_task\":" << scratch_bytes
            << ",\"derived_rotation_coefficient_bytes\":" << coefficient_bytes
            << ",\"shape_anchor_bytes\":"
            << static_cast<size_t>(image.rows()) * options.tokens * sizeof(float)
            << ",\"persistent_dense_weight_bytes\":0"
            << ",\"decoded_group_scale_matrix_bytes\":0"
            << ",\"read_only_mmap\":true"
            << ",\"root_authenticated\":true"
            << ",\"semantic_validation_passed\":true"
            << ",\"precomputed_rotation_output_bit_exact_vs_direct_v47\":"
            << (bit_exact ? "true" : "false")
            << ",\"maximum_absolute_error_vs_direct_v47\":"
            << std::setprecision(17) << maximum_error
            << ",\"performance_timing_ran\":false"
            << ",\"passed\":true,\"complete\":true,\"terminal\":true}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "qwen38_aff3_ggml_v48: " << error.what() << '\n';
        return 1;
    }
}
