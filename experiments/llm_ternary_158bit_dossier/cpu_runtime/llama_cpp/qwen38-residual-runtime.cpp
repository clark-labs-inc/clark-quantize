#include "qwen38-residual-runtime.h"

#include "qwen38-residual-kernels/qwen38_calibration_residual_cpu.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#if !defined(__AVX2__) || !defined(__F16C__) || !defined(__FMA__)
#error "Qwen3.8 residual runtime requires AVX2, F16C, and FMA"
#endif

namespace {

constexpr size_t kHeaderBytes = 4096;
constexpr size_t kTableOffset = 4096;
constexpr size_t kTableEntryBytes = 64;
constexpr size_t kPayloadOffset = 8192;
constexpr size_t kPayloadBytes = 174144;
constexpr size_t kRootOffset = 188;
constexpr size_t kRootBytes = 32;
constexpr int kLayerCount = 64;
constexpr int kInputFeatures = 5120;
constexpr int kOutputFeatures = 5120;
constexpr int kRank = 16;
constexpr int kGraphTasks = 4;
constexpr std::array<char, 8> kMagic = {'Q', '3', '8', 'R', '1', '6', 'I', '1'};
constexpr std::array<uint8_t, 20> kModelRevision = {
    0x1d, 0x4b, 0xf0, 0xf2, 0xff, 0x60, 0x12, 0xfd, 0x82, 0x03,
    0x9f, 0x2f, 0xa5, 0x27, 0x39, 0xd0, 0xdd, 0x7c, 0x60, 0xc0,
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error("Qwen3.8 residual: " + message);
}

[[noreturn]] void callback_abort(const char *message) {
    std::fprintf(stderr, "Qwen3.8 residual callback failure: %s\n", message);
    std::abort();
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

uint32_t rotate_right(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
}

class Sha256 {
public:
    Sha256() : state_({
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
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
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
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
            h = g; g = f; f = e; e = d + first;
            d = c; c = b; b = a; a = first + second;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ {};
    std::array<uint8_t, 64> block_ {};
    size_t buffered_ = 0;
    size_t total_bytes_ = 0;
};

std::array<uint8_t, 32> digest(const void *content, size_t bytes) {
    Sha256 sha;
    sha.update(content, bytes);
    return sha.finish();
}

int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

std::array<uint8_t, 32> required_sha_environment(const char *name) {
    const char *text = std::getenv(name);
    if (text == nullptr || std::strlen(text) != 64) {
        fail(std::string(name) + " must be an exact SHA-256");
    }
    std::array<uint8_t, 32> result {};
    for (size_t index = 0; index < result.size(); ++index) {
        const int high = hex_digit(text[index * 2]);
        const int low = hex_digit(text[index * 2 + 1]);
        if (high < 0 || low < 0) {
            fail(std::string(name) + " is not hexadecimal");
        }
        result[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return result;
}

size_t row_offset(const ggml_tensor *tensor, int64_t row) {
    const int64_t i1 = row % tensor->ne[1];
    row /= tensor->ne[1];
    const int64_t i2 = row % tensor->ne[2];
    row /= tensor->ne[2];
    const int64_t i3 = row;
    return static_cast<size_t>(i1) * tensor->nb[1] +
        static_cast<size_t>(i2) * tensor->nb[2] +
        static_cast<size_t>(i3) * tensor->nb[3];
}

}  // namespace

class Qwen38ResidualMappedFile {
public:
    explicit Qwen38ResidualMappedFile(const char *path) {
        const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            fail(std::string("could not open image: ") + std::strerror(errno));
        }
        struct stat status {};
        if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            close(descriptor);
            fail("could not stat image");
        }
        bytes = static_cast<size_t>(status.st_size);
        void *mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
        close(descriptor);
        if (mapped == MAP_FAILED) {
            fail("could not mmap image");
        }
        data = static_cast<const uint8_t *>(mapped);
    }

    ~Qwen38ResidualMappedFile() {
        if (data != nullptr) {
            munmap(const_cast<uint8_t *>(data), bytes);
        }
    }

    Qwen38ResidualMappedFile(const Qwen38ResidualMappedFile &) = delete;
    Qwen38ResidualMappedFile &operator=(const Qwen38ResidualMappedFile &) = delete;

    const uint8_t *data = nullptr;
    size_t bytes = 0;
};

struct Qwen38ResidualLayer {
    const int8_t *left_codes = nullptr;
    const uint16_t *left_scales = nullptr;
    const int8_t *right_codes = nullptr;
    const uint16_t *right_scales = nullptr;
    const uint16_t *bias = nullptr;
    bool active = false;
};

struct qwen38_residual_bundle {
    std::shared_ptr<Qwen38ResidualMappedFile> mapping;
    std::array<Qwen38ResidualLayer, kLayerCount> layers {};
    int active_count = 0;
};

struct qwen38_residual_node_state {
    qwen38_residual_node_state(
            qwen38_residual_bundle_ptr bundle_value,
            int layer_value)
        : bundle(std::move(bundle_value)), layer(layer_value) {}

    qwen38_residual_bundle_ptr bundle;
    int layer = -1;
};

namespace {

qwen38_residual_bundle_ptr load_bundle(
        const char *path,
        const std::array<uint8_t, 32> &training_sha,
        const std::array<uint8_t, 32> &residual_sha,
        const std::array<uint8_t, 32> &preregistration_sha) {
    auto result = std::make_shared<qwen38_residual_bundle>();
    result->mapping = std::make_shared<Qwen38ResidualMappedFile>(path);
    const uint8_t *data = result->mapping->data;
    const size_t bytes = result->mapping->bytes;
    if (bytes < kPayloadOffset ||
        std::memcmp(data, kMagic.data(), kMagic.size()) != 0 ||
        read_u32(data + 8) != 1 || read_u32(data + 12) != kHeaderBytes ||
        read_u32(data + 16) != kLayerCount ||
        read_u32(data + 20) != kInputFeatures ||
        read_u32(data + 24) != kOutputFeatures ||
        read_u32(data + 28) != kRank ||
        read_u32(data + 32) != kPayloadBytes ||
        read_u32(data + 36) != kTableEntryBytes ||
        read_u64(data + 40) != kTableOffset ||
        read_u64(data + 48) != kPayloadOffset ||
        read_u64(data + 56) != bytes ||
        std::memcmp(data + 72, kModelRevision.data(), kModelRevision.size()) != 0 ||
        std::memcmp(data + 92, training_sha.data(), training_sha.size()) != 0 ||
        std::memcmp(data + 124, residual_sha.data(), residual_sha.size()) != 0 ||
        std::memcmp(data + 156, preregistration_sha.data(), preregistration_sha.size()) != 0) {
        fail("image header or parent identity changed");
    }
    for (size_t index = 68; index < 72; ++index) {
        if (data[index] != 0) fail("header reserved bytes changed");
    }
    for (size_t index = 220; index < kHeaderBytes; ++index) {
        if (data[index] != 0) fail("header tail changed");
    }
    Sha256 root;
    root.update(data, kRootOffset);
    const std::array<uint8_t, kRootBytes> zero_root {};
    root.update(zero_root.data(), zero_root.size());
    root.update(data + kRootOffset + kRootBytes,
                bytes - kRootOffset - kRootBytes);
    const auto observed_root = root.finish();
    if (std::memcmp(data + kRootOffset, observed_root.data(), observed_root.size()) != 0) {
        fail("embedded root SHA-256 mismatch");
    }

    const int expected_active = static_cast<int>(read_u32(data + 64));
    size_t expected_offset = kPayloadOffset;
    for (int layer = 0; layer < kLayerCount; ++layer) {
        const uint8_t *entry = data + kTableOffset +
            static_cast<size_t>(layer) * kTableEntryBytes;
        const uint32_t active = read_u32(entry);
        const uint32_t observed_layer = read_u32(entry + 4);
        const uint64_t offset = read_u64(entry + 8);
        const uint64_t payload_bytes = read_u64(entry + 16);
        if (observed_layer != static_cast<uint32_t>(layer) ||
            read_u64(entry + 56) != 0) {
            fail("layer table ordering changed");
        }
        if (active == 0) {
            const std::array<uint8_t, 32> zero_sha {};
            if (offset != 0 || payload_bytes != 0 ||
                std::memcmp(entry + 24, zero_sha.data(), zero_sha.size()) != 0) {
                fail("bypass layer has payload state");
            }
            continue;
        }
        const bool range_valid = offset <= bytes && payload_bytes <= bytes - offset;
        const auto payload_digest =
            range_valid && active == 1
            ? digest(data + offset, payload_bytes)
            : std::array<uint8_t, 32> {};
        if (active != 1 || offset != expected_offset ||
            payload_bytes != kPayloadBytes || !range_valid ||
            std::memcmp(entry + 24, payload_digest.data(), payload_digest.size()) != 0) {
            fail("active layer payload authentication failed");
        }
        Qwen38ResidualLayer &view = result->layers[layer];
        const uint8_t *cursor = data + offset;
        view.left_codes = reinterpret_cast<const int8_t *>(cursor);
        cursor += static_cast<size_t>(kInputFeatures) * kRank;
        view.left_scales = reinterpret_cast<const uint16_t *>(cursor);
        cursor += static_cast<size_t>(kRank) * sizeof(uint16_t);
        view.right_codes = reinterpret_cast<const int8_t *>(cursor);
        cursor += static_cast<size_t>(kRank) * kOutputFeatures;
        view.right_scales = reinterpret_cast<const uint16_t *>(cursor);
        cursor += static_cast<size_t>(kRank) * sizeof(uint16_t);
        view.bias = reinterpret_cast<const uint16_t *>(cursor);
        cursor += static_cast<size_t>(kOutputFeatures) * sizeof(uint16_t);
        if (cursor != data + offset + kPayloadBytes) {
            fail("active payload layout changed");
        }
        view.active = true;
        ++result->active_count;
        expected_offset += kPayloadBytes;
    }
    if (result->active_count != expected_active || expected_offset != bytes ||
        result->active_count <= 0) {
        fail("active layer denominator changed");
    }
    const uint32_t required = CQ3_CALIBRATION_FEATURE_AVX2 |
        CQ3_CALIBRATION_FEATURE_FMA | CQ3_CALIBRATION_FEATURE_F16C;
    if ((cq3_calibration_residual_cpu_features() & required) != required) {
        fail("required AMD AVX2/FMA/F16C features are unavailable");
    }
    return result;
}

void graph_callback(
        ggml_tensor *dst,
        const ggml_tensor *base,
        const ggml_tensor *input,
        int ith,
        int nth,
        void *opaque) {
    auto *state = static_cast<qwen38_residual_node_state *>(opaque);
    if (state == nullptr || state->bundle == nullptr || state->layer < 0 ||
        state->layer >= kLayerCount || ith < 0 || ith >= nth) {
        callback_abort("invalid task identity");
    }
    const Qwen38ResidualLayer &payload = state->bundle->layers[state->layer];
    if (!payload.active || dst->type != GGML_TYPE_F32 ||
        base->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 ||
        dst->ne[0] != kOutputFeatures || base->ne[0] != kOutputFeatures ||
        input->ne[0] != kInputFeatures || dst->nb[0] != sizeof(float) ||
        base->nb[0] != sizeof(float) || input->nb[0] != sizeof(float) ||
        ggml_nrows(dst) != ggml_nrows(base) ||
        ggml_nrows(dst) != ggml_nrows(input)) {
        callback_abort("tensor contract changed");
    }
    const int64_t rows = ggml_nrows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const auto *input_row = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(input->data) + row_offset(input, row));
        const auto *base_row = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(base->data) + row_offset(base, row));
        auto *output_row = reinterpret_cast<float *>(
            static_cast<uint8_t *>(dst->data) + row_offset(dst, row));
        std::memcpy(output_row, base_row, kOutputFeatures * sizeof(float));
        float scratch[kRank];
        const int status = cq3_calibration_residual_f32_avx2_trusted(
            input_row,
            payload.left_codes,
            payload.left_scales,
            payload.right_codes,
            payload.right_scales,
            payload.bias,
            kInputFeatures,
            kOutputFeatures,
            kRank,
            scratch,
            output_row,
            1U);
        if (status != 0) {
            callback_abort("native residual rejected its inputs");
        }
    }
}

}  // namespace

bool qwen38_residual_environment_requested() {
    const char *path = std::getenv("QWEN38_RESIDUAL_IMAGE");
    const char *training = std::getenv("QWEN38_RESIDUAL_TRAINING_SHA256");
    const char *receipt = std::getenv("QWEN38_RESIDUAL_RECEIPT_SHA256");
    const char *prereg = std::getenv("QWEN38_RESIDUAL_PREREG_SHA256");
    return (path != nullptr && *path != '\0') ||
        (training != nullptr && *training != '\0') ||
        (receipt != nullptr && *receipt != '\0') ||
        (prereg != nullptr && *prereg != '\0');
}

qwen38_residual_bundle_ptr qwen38_residual_load_from_environment(
        int n_layers,
        int expected_input_features,
        int expected_output_features) {
    const char *path = std::getenv("QWEN38_RESIDUAL_IMAGE");
    if (path == nullptr || *path == '\0') {
        if (qwen38_residual_environment_requested()) {
            fail("QWEN38_RESIDUAL_IMAGE is required with residual SHA identities");
        }
        return nullptr;
    }
    if (n_layers != kLayerCount || expected_input_features != kInputFeatures ||
        expected_output_features != kOutputFeatures) {
        fail("residual runtime is valid only for Qwen3.8-27B text blocks");
    }
    return load_bundle(
        path,
        required_sha_environment("QWEN38_RESIDUAL_TRAINING_SHA256"),
        required_sha_environment("QWEN38_RESIDUAL_RECEIPT_SHA256"),
        required_sha_environment("QWEN38_RESIDUAL_PREREG_SHA256"));
}

bool qwen38_residual_any_enabled(
        const qwen38_residual_bundle_ptr &bundle) {
    return bundle != nullptr && bundle->active_count > 0;
}

bool qwen38_residual_layer_enabled(
        const qwen38_residual_bundle_ptr &bundle,
        int layer) {
    return bundle != nullptr && layer >= 0 && layer < kLayerCount &&
        bundle->layers[layer].active;
}

ggml_tensor * qwen38_residual_apply(
        ggml_context *ctx,
        ggml_tensor *base_output,
        ggml_tensor *block_input,
        const qwen38_residual_bundle_ptr &bundle,
        int layer,
        std::vector<qwen38_residual_node_state_ptr> &node_states) {
    if (!qwen38_residual_layer_enabled(bundle, layer)) {
        return base_output;
    }
    if (ctx == nullptr || base_output == nullptr || block_input == nullptr ||
        base_output->type != GGML_TYPE_F32 || block_input->type != GGML_TYPE_F32 ||
        base_output->ne[0] != kOutputFeatures ||
        block_input->ne[0] != kInputFeatures ||
        ggml_nrows(base_output) != ggml_nrows(block_input)) {
        fail("post-block graph placement contract changed");
    }
    auto state = std::make_shared<qwen38_residual_node_state>(bundle, layer);
    ggml_tensor *output = ggml_map_custom2(
        ctx,
        base_output,
        block_input,
        graph_callback,
        kGraphTasks,
        state.get());
    ggml_set_name(output, "qwen38_post_block_residual");
    node_states.push_back(std::move(state));
    return output;
}
