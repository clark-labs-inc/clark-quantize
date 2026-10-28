static constexpr int qwen38_cq3_p256_rows = 256;
static constexpr size_t qwen38_cq3_p256_scale_bytes = 256*sizeof(ggml_half);
static constexpr size_t qwen38_cq3_p256_chunk_bytes = 256*4;
static constexpr size_t qwen38_cq3_p256_panel_bytes =
        qwen38_cq3_p256_scale_bytes + 8*qwen38_cq3_p256_chunk_bytes;

static void test_x86_cq3_activation_stationary_p256() {
    constexpr int rows = 256;
    constexpr int columns = 512;
    const int blocks_per_row = columns/QK_CQ3_0;
    std::vector<float> source_row(columns), source_activation(columns);
    std::vector<block_cq3_0> source((size_t) rows*blocks_per_row);
    std::vector<block_cq3_0x8> repacked_x8((size_t) rows/8*blocks_per_row);
    std::vector<uint8_t> repacked_p256(ggml_cq3_0p256_size(rows, blocks_per_row));
    std::vector<block_cq8_0> activation(columns/QK_CQ8_0);
    std::vector<float> ordinary(rows), output_x8(rows), output_p256(rows);
    std::vector<int32_t> scratch(qwen38_cq3_p256_rows);

    for (int row = 0; row < rows; ++row) {
        qwen38_fill_cq3_bench_row(source_row.data(), columns, row, 0x256c03u);
        quantize_row_cq3_0_ref(source_row.data(), source.data() + row*blocks_per_row, columns);
    }
    for (int column = 0; column < columns; ++column) {
        source_activation[column] = (float) (((column*43 + 29) % 251) - 125)/47.0f;
    }
    quantize_row_cq8_0_ref(source_activation.data(), activation.data(), columns);
    ggml_repack_cq3_0_to_cq3_0x8(source.data(), repacked_x8.data(), rows, blocks_per_row);
    ggml_repack_cq3_0_to_cq3_0p256(source.data(), repacked_p256.data(), rows, blocks_per_row);
    assert(repacked_p256.size() == source.size()*sizeof(block_cq3_0));

    const int panels = rows/qwen38_cq3_p256_rows;
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        for (int group = 0; group < 2; ++group) {
            for (int panel = 0; panel < panels; ++panel) {
                const uint8_t * panel_base = repacked_p256.data() +
                        ((size_t) (ib*2 + group)*panels + panel)*qwen38_cq3_p256_panel_bytes;
                const ggml_half * scales = (const ggml_half *) panel_base;
                const uint8_t * codes = panel_base + qwen38_cq3_p256_scale_bytes;
                for (int row = 0; row < qwen38_cq3_p256_rows; ++row) {
                    const int source_row_index = panel*qwen38_cq3_p256_rows + row;
                    const block_cq3_0 & original = source[(size_t) source_row_index*blocks_per_row + ib];
                    assert(scales[row] == original.d[group]);
                    for (int chunk = 0; chunk < 8; ++chunk) {
                        assert(std::memcmp(
                                codes + (size_t) chunk*qwen38_cq3_p256_chunk_bytes + row*4,
                                original.qs + group*32 + chunk*4, 4) == 0);
                    }
                }
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        ggml_vec_dot_cq3_0_cq8_0(columns, ordinary.data() + row, 0,
                source.data() + row*blocks_per_row, 0, activation.data(), 0, 1);
    }
    ggml_gemv_cq3_0x8_cq8_0(columns, output_x8.data(), repacked_x8.data(), activation.data(), rows);
    ggml_gemv_cq3_0p256_cq8_0(columns, output_p256.data(), repacked_p256.data(),
            activation.data(), rows, scratch.data());
    assert(std::memcmp(output_x8.data(), output_p256.data(), rows*sizeof(float)) == 0);
    for (int row = 0; row < rows; ++row) {
        assert(std::isfinite(output_p256[row]));
        assert(qwen38_cq3_scaled_delta(ordinary[row], output_p256[row]) <= 5e-5f);
    }
}

static void run_qwen38_27b_cq3_activation_stationary_p256_v20() {
    constexpr int rows = 17408;
    constexpr int columns = 5120;
    constexpr int warmup_pairs = 2;
    constexpr int measured_pairs = 21;
    constexpr size_t cache_eviction_bytes = 256u*1024u*1024u;
    const int blocks_per_row = columns/QK_CQ3_0;

    std::vector<block_cq3_0> gate_weights((size_t) rows*blocks_per_row);
    std::vector<block_cq3_0> up_weights((size_t) rows*blocks_per_row);
    std::vector<block_cq3_0x8> gate_x8((size_t) rows/8*blocks_per_row);
    std::vector<block_cq3_0x8> up_x8((size_t) rows/8*blocks_per_row);
    std::vector<uint8_t> gate_p256(ggml_cq3_0p256_size(rows, blocks_per_row));
    std::vector<uint8_t> up_p256(ggml_cq3_0p256_size(rows, blocks_per_row));
    std::vector<block_cq8_0> activation(columns/QK_CQ8_0);
    std::vector<float> source_row(columns), source_activation(columns);
    std::vector<float> ordinary_gate(rows), ordinary_up(rows);
    std::vector<float> output_x8_gate(rows), output_x8_up(rows);
    std::vector<float> output_p256_gate(rows), output_p256_up(rows);
    std::vector<int32_t> scratch(qwen38_cq3_p256_rows);
    std::vector<uint8_t> eviction(cache_eviction_bytes);

    for (int row = 0; row < rows; ++row) {
        qwen38_fill_cq3_bench_row(source_row.data(), columns, row, 0x13579bdu);
        quantize_row_cq3_0_ref(source_row.data(), gate_weights.data() + row*blocks_per_row, columns);
        qwen38_fill_cq3_bench_row(source_row.data(), columns, row, 0x2468aceu);
        quantize_row_cq3_0_ref(source_row.data(), up_weights.data() + row*blocks_per_row, columns);
    }
    for (int column = 0; column < columns; ++column) {
        source_activation[column] = (float) (((column*37 + 17) % 255) - 127)/53.0f;
    }
    quantize_row_cq8_0_ref(source_activation.data(), activation.data(), columns);
    ggml_repack_cq3_0_to_cq3_0x8(gate_weights.data(), gate_x8.data(), rows, blocks_per_row);
    ggml_repack_cq3_0_to_cq3_0x8(up_weights.data(), up_x8.data(), rows, blocks_per_row);
    const auto repack_start = std::chrono::steady_clock::now();
    ggml_repack_cq3_0_to_cq3_0p256(gate_weights.data(), gate_p256.data(), rows, blocks_per_row);
    ggml_repack_cq3_0_to_cq3_0p256(up_weights.data(), up_p256.data(), rows, blocks_per_row);
    const auto repack_stop = std::chrono::steady_clock::now();
    const double repack_us = std::chrono::duration<double, std::micro>(repack_stop - repack_start).count();

    auto run_ordinary = [&]() {
        qwen38_run_cq3_x8_control(rows, columns, gate_weights.data(), activation.data(), ordinary_gate.data());
        qwen38_run_cq3_x8_control(rows, columns, up_weights.data(), activation.data(), ordinary_up.data());
    };
    auto run_x8 = [&]() {
        ggml_gemv_cq3_0x8_cq8_0(columns, output_x8_gate.data(), gate_x8.data(), activation.data(), rows);
        ggml_gemv_cq3_0x8_cq8_0(columns, output_x8_up.data(), up_x8.data(), activation.data(), rows);
    };
    auto run_p256 = [&]() {
        ggml_gemv_cq3_0p256_cq8_0(columns, output_p256_gate.data(), gate_p256.data(),
                activation.data(), rows, scratch.data());
        ggml_gemv_cq3_0p256_cq8_0(columns, output_p256_up.data(), up_p256.data(),
                activation.data(), rows, scratch.data());
    };

    run_ordinary();
    run_x8();
    run_p256();
    assert(std::memcmp(output_x8_gate.data(), output_p256_gate.data(), rows*sizeof(float)) == 0);
    assert(std::memcmp(output_x8_up.data(), output_p256_up.data(), rows*sizeof(float)) == 0);
    float max_scaled_delta = 0.0f;
    for (int row = 0; row < rows; ++row) {
        max_scaled_delta = std::max(max_scaled_delta,
                qwen38_cq3_scaled_delta(ordinary_gate[row], output_p256_gate[row]));
        max_scaled_delta = std::max(max_scaled_delta,
                qwen38_cq3_scaled_delta(ordinary_up[row], output_p256_up[row]));
        assert(std::isfinite(output_p256_gate[row]));
        assert(std::isfinite(output_p256_up[row]));
    }
    assert(max_scaled_delta <= 5e-5f);

    for (int warmup = 0; warmup < warmup_pairs; ++warmup) {
        if ((warmup & 1) == 0) {
            run_x8(); run_p256(); run_ordinary(); run_p256();
        } else {
            run_p256(); run_x8(); run_p256(); run_ordinary();
        }
    }

    std::vector<double> x8_us, p256_for_x8_us, ordinary_us, p256_for_ordinary_us;
    std::vector<double> x8_over_p256_ratios, ordinary_over_p256_ratios;
    std::vector<double> x8_pair_frequency_before_mhz, x8_pair_frequency_after_mhz;
    std::vector<double> ordinary_pair_frequency_before_mhz, ordinary_pair_frequency_after_mhz;
    volatile float sink = 0.0f;
    auto measure = [&](auto && run, const std::vector<float> & gate, const std::vector<float> & up) {
        qwen38_evict_cq3_x16_cache(eviction);
        const auto start = std::chrono::steady_clock::now();
        run();
        const auto stop = std::chrono::steady_clock::now();
        sink += gate[0] + up[rows - 1];
        return std::chrono::duration<double, std::micro>(stop - start).count();
    };
    auto measure_x8_pair = [&](bool ab_order) {
        x8_pair_frequency_before_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        double control = 0.0;
        double candidate = 0.0;
        if (ab_order) {
            control = measure(run_x8, output_x8_gate, output_x8_up);
            candidate = measure(run_p256, output_p256_gate, output_p256_up);
        } else {
            candidate = measure(run_p256, output_p256_gate, output_p256_up);
            control = measure(run_x8, output_x8_gate, output_x8_up);
        }
        x8_pair_frequency_after_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        x8_us.push_back(control);
        p256_for_x8_us.push_back(candidate);
        x8_over_p256_ratios.push_back(control/candidate);
    };
    auto measure_ordinary_pair = [&](bool ab_order) {
        ordinary_pair_frequency_before_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        double control = 0.0;
        double candidate = 0.0;
        if (ab_order) {
            control = measure(run_ordinary, ordinary_gate, ordinary_up);
            candidate = measure(run_p256, output_p256_gate, output_p256_up);
        } else {
            candidate = measure(run_p256, output_p256_gate, output_p256_up);
            control = measure(run_ordinary, ordinary_gate, ordinary_up);
        }
        ordinary_pair_frequency_after_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        ordinary_us.push_back(control);
        p256_for_ordinary_us.push_back(candidate);
        ordinary_over_p256_ratios.push_back(control/candidate);
    };

    for (int repetition = 0; repetition < measured_pairs; ++repetition) {
        const bool ab_order = (repetition & 1) == 0;
        const bool x8_family_first = ((repetition/2) & 1) == 0;
        if (x8_family_first) {
            measure_x8_pair(ab_order);
            measure_ordinary_pair(ab_order);
        } else {
            measure_ordinary_pair(ab_order);
            measure_x8_pair(ab_order);
        }
    }

    const double x8_ratio_median = qwen38_cq3_x16_median(x8_over_p256_ratios);
    const double ordinary_ratio_median = qwen38_cq3_x16_median(ordinary_over_p256_ratios);
    const double x8_ratio_mad_fraction = qwen38_cq3_x19_mad_fraction(x8_over_p256_ratios);
    const double ordinary_ratio_mad_fraction = qwen38_cq3_x19_mad_fraction(ordinary_over_p256_ratios);
    std::printf("{\"schema\":\"qwen38_27b_cq3_activation_stationary_p256_benchmark_v20\",");
    std::printf("\"rows_per_matrix\":%d,\"matrices\":2,\"columns\":%d,", rows, columns);
    std::printf("\"rows_per_panel\":%d,\"warmup_pairs\":%d,\"measured_pairs\":%d,",
            qwen38_cq3_p256_rows, warmup_pairs, measured_pairs);
    std::printf("\"cache_eviction_bytes\":%zu,\"candidate_repack_us\":%.3f,",
            cache_eviction_bytes, repack_us);
    std::printf("\"source_bytes\":%zu,\"x8_bytes\":%zu,\"candidate_bytes\":%zu,",
            2*gate_weights.size()*sizeof(block_cq3_0), 2*gate_x8.size()*sizeof(block_cq3_0x8),
            gate_p256.size() + up_p256.size());
    std::printf("\"x8_candidate_bit_exact\":true,\"max_scaled_delta_vs_ordinary\":%.9g,",
            (double) max_scaled_delta);
    std::printf("\"x8_us\":"); qwen38_print_cq3_x16_values(x8_us);
    std::printf(",\"candidate_for_x8_us\":"); qwen38_print_cq3_x16_values(p256_for_x8_us);
    std::printf(",\"ordinary_us\":"); qwen38_print_cq3_x16_values(ordinary_us);
    std::printf(",\"candidate_for_ordinary_us\":"); qwen38_print_cq3_x16_values(p256_for_ordinary_us);
    std::printf(",\"x8_over_candidate_pairwise_ratios\":"); qwen38_print_cq3_x16_values(x8_over_p256_ratios);
    std::printf(",\"ordinary_over_candidate_pairwise_ratios\":"); qwen38_print_cq3_x16_values(ordinary_over_p256_ratios);
    std::printf(",\"x8_pair_frequency_before_mhz\":"); qwen38_print_cq3_x16_values(x8_pair_frequency_before_mhz);
    std::printf(",\"x8_pair_frequency_after_mhz\":"); qwen38_print_cq3_x16_values(x8_pair_frequency_after_mhz);
    std::printf(",\"ordinary_pair_frequency_before_mhz\":"); qwen38_print_cq3_x16_values(ordinary_pair_frequency_before_mhz);
    std::printf(",\"ordinary_pair_frequency_after_mhz\":"); qwen38_print_cq3_x16_values(ordinary_pair_frequency_after_mhz);
    std::printf(",\"x8_over_candidate_pairwise_ratio_median\":%.9f,", x8_ratio_median);
    std::printf("\"ordinary_over_candidate_pairwise_ratio_median\":%.9f,", ordinary_ratio_median);
    std::printf("\"x8_over_candidate_pairwise_ratio_mad_fraction\":%.9f,", x8_ratio_mad_fraction);
    std::printf("\"ordinary_over_candidate_pairwise_ratio_mad_fraction\":%.9f,", ordinary_ratio_mad_fraction);
    std::printf("\"frequency_source\":\"sysfs_scaling_cur_freq_else_proc_cpuinfo_cpu_mhz\",\"sink\":%.9g}\n",
            (double) sink);
}
