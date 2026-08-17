static void qwen38_fill_cq3_bench_row(float * row, int columns, int row_index, uint32_t salt) {
    uint32_t state = salt ^ (uint32_t) row_index*0x9e3779b9u;
    const float scale = 0.5f + 0.03125f*(float) ((row_index + (int) salt) & 15);
    for (int column = 0; column < columns; ++column) {
        state = state*1664525u + 1013904223u;
        row[column] = (float) ((int) ((state >> 24) % 3u) - 1)*scale;
    }
}

static void qwen38_run_cq3_pair_control(
        int rows,
        int columns,
        const block_cq3_0 * gate_weights,
        const block_cq3_0 * up_weights,
        const block_cq8_0 * activation,
        float * gate,
        float * up) {
    const int blocks_per_row = columns/QK_CQ3_0;
    for (int row = 0; row < rows; ++row) {
        ggml_vec_dot_cq3_0_cq8_0(
            columns, gate + row, 0,
            gate_weights + row*blocks_per_row, 0,
            activation, 0, 1);
    }
    for (int row = 0; row < rows; ++row) {
        ggml_vec_dot_cq3_0_cq8_0(
            columns, up + row, 0,
            up_weights + row*blocks_per_row, 0,
            activation, 0, 1);
    }
}

static void qwen38_run_cq3_pair_candidate(
        int rows,
        int columns,
        const block_cq3_0 * gate_weights,
        const block_cq3_0 * up_weights,
        const block_cq8_0 * activation,
        float * gate,
        float * up) {
    const int blocks_per_row = columns/QK_CQ3_0;
    const size_t row_bytes = (size_t) blocks_per_row*sizeof(block_cq3_0);
    for (int row = 0; row < rows; row += 4) {
        ggml_gemv_cq3_0_cq8_0_pair_x86(
            columns, gate + row, up + row,
            gate_weights + row*blocks_per_row, row_bytes,
            up_weights + row*blocks_per_row, row_bytes,
            activation, std::min(4, rows - row));
    }
}

static void qwen38_evict_cq3_bench_cache(std::vector<uint8_t> & cache) {
    static uint8_t epoch = 1;
    for (size_t offset = 0; offset < cache.size(); offset += 64) {
        cache[offset] = (uint8_t) (cache[offset] + epoch);
    }
    ++epoch;
}

static double qwen38_cq3_bench_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size()/2];
}

static void qwen38_print_cq3_bench_values(const std::vector<double> & values) {
    std::printf("[");
    for (size_t index = 0; index < values.size(); ++index) {
        std::printf(index == 0 ? "%.3f" : ",%.3f", values[index]);
    }
    std::printf("]");
}

static void run_qwen38_27b_cq3_x86_pair_benchmark() {
    constexpr int rows = 17408;
    constexpr int columns = 5120;
    constexpr int warmups = 2;
    constexpr int repetitions = 9;
    constexpr size_t cache_eviction_bytes = 256u*1024u*1024u;
    static_assert(columns % QK_CQ3_0 == 0, "Qwen3.8-27B width must fit CQ3 blocks");

    const int blocks_per_row = columns/QK_CQ3_0;
    std::vector<block_cq3_0> gate_weights((size_t) rows*blocks_per_row);
    std::vector<block_cq3_0> up_weights((size_t) rows*blocks_per_row);
    std::vector<block_cq8_0> activation(columns/QK_CQ8_0);
    std::vector<float> source_row(columns);
    std::vector<float> source_activation(columns);
    std::vector<float> control_gate(rows);
    std::vector<float> control_up(rows);
    std::vector<float> candidate_gate(rows);
    std::vector<float> candidate_up(rows);
    std::vector<uint8_t> eviction(cache_eviction_bytes);

    for (int row = 0; row < rows; ++row) {
        qwen38_fill_cq3_bench_row(source_row.data(), columns, row, 0x13579bdu);
        quantize_row_cq3_0_ref(
            source_row.data(), gate_weights.data() + row*blocks_per_row, columns);
        qwen38_fill_cq3_bench_row(source_row.data(), columns, row, 0x2468aceu);
        quantize_row_cq3_0_ref(
            source_row.data(), up_weights.data() + row*blocks_per_row, columns);
    }
    for (int column = 0; column < columns; ++column) {
        const int centered = ((column*37 + 17) % 255) - 127;
        source_activation[column] = (float) centered/53.0f;
    }
    quantize_row_cq8_0_ref(source_activation.data(), activation.data(), columns);

    qwen38_run_cq3_pair_control(
        rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
        control_gate.data(), control_up.data());
    qwen38_run_cq3_pair_candidate(
        rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
        candidate_gate.data(), candidate_up.data());
    assert(std::memcmp(control_gate.data(), candidate_gate.data(), rows*sizeof(float)) == 0);
    assert(std::memcmp(control_up.data(), candidate_up.data(), rows*sizeof(float)) == 0);

    for (int warmup = 0; warmup < warmups; ++warmup) {
        qwen38_run_cq3_pair_control(
            rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
            control_gate.data(), control_up.data());
        qwen38_run_cq3_pair_candidate(
            rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
            candidate_gate.data(), candidate_up.data());
    }

    std::vector<double> control_us;
    std::vector<double> candidate_us;
    control_us.reserve(repetitions);
    candidate_us.reserve(repetitions);
    volatile float sink = 0.0f;
    auto measure_control = [&]() {
        qwen38_evict_cq3_bench_cache(eviction);
        const auto start = std::chrono::steady_clock::now();
        qwen38_run_cq3_pair_control(
            rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
            control_gate.data(), control_up.data());
        const auto stop = std::chrono::steady_clock::now();
        sink += control_gate[0] + control_gate[rows - 1] + control_up[rows/2];
        return std::chrono::duration<double, std::micro>(stop - start).count();
    };
    auto measure_candidate = [&]() {
        qwen38_evict_cq3_bench_cache(eviction);
        const auto start = std::chrono::steady_clock::now();
        qwen38_run_cq3_pair_candidate(
            rows, columns, gate_weights.data(), up_weights.data(), activation.data(),
            candidate_gate.data(), candidate_up.data());
        const auto stop = std::chrono::steady_clock::now();
        sink += candidate_gate[0] + candidate_gate[rows - 1] + candidate_up[rows/2];
        return std::chrono::duration<double, std::micro>(stop - start).count();
    };
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        if ((repetition & 1) == 0) {
            control_us.push_back(measure_control());
            candidate_us.push_back(measure_candidate());
        } else {
            candidate_us.push_back(measure_candidate());
            control_us.push_back(measure_control());
        }
    }

    const double control_median = qwen38_cq3_bench_median(control_us);
    const double candidate_median = qwen38_cq3_bench_median(candidate_us);
    std::printf("{\"schema\":\"qwen38_27b_cq3_x86_pair_benchmark_v1\",");
    std::printf("\"rows\":%d,\"columns\":%d,\"warmups\":%d,\"repetitions\":%d,", rows, columns, warmups, repetitions);
    std::printf("\"cache_eviction_bytes\":%zu,\"exact_control_candidate\":true,", cache_eviction_bytes);
    std::printf("\"control_us\":");
    qwen38_print_cq3_bench_values(control_us);
    std::printf(",\"candidate_us\":");
    qwen38_print_cq3_bench_values(candidate_us);
    std::printf(",\"control_median_us\":%.3f,\"candidate_median_us\":%.3f,", control_median, candidate_median);
    std::printf("\"speedup\":%.8f,\"sink\":%.9g}\n", control_median/candidate_median, (double) sink);
}
