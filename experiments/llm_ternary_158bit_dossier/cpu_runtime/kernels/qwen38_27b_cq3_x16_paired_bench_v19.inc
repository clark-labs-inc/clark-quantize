static double qwen38_cq3_x19_frequency_mhz() {
    const char * sysfs_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    if (FILE * file = std::fopen(sysfs_path, "r")) {
        double khz = 0.0;
        const int matched = std::fscanf(file, "%lf", &khz);
        std::fclose(file);
        if (matched == 1 && khz > 0.0) {
            return khz/1000.0;
        }
    }

    if (FILE * file = std::fopen("/proc/cpuinfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), file) != nullptr) {
            double mhz = 0.0;
            if (std::sscanf(line, "cpu MHz%*[^:]: %lf", &mhz) == 1 && mhz > 0.0) {
                std::fclose(file);
                return mhz;
            }
        }
        std::fclose(file);
    }
    return 0.0;
}

static double qwen38_cq3_x19_mad_fraction(const std::vector<double> & values) {
    const double median = qwen38_cq3_x16_median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) {
        deviations.push_back(std::fabs(value - median));
    }
    return qwen38_cq3_x16_median(deviations)/median;
}

static void run_qwen38_27b_cq3_x16_paired_confirmation_v19() {
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
    std::vector<block_cq3_0x16> gate_x16((size_t) rows/16*blocks_per_row);
    std::vector<block_cq3_0x16> up_x16((size_t) rows/16*blocks_per_row);
    std::vector<block_cq8_0> activation(columns/QK_CQ8_0);
    std::vector<float> source_row(columns), source_activation(columns);
    std::vector<float> ordinary_gate(rows), ordinary_up(rows);
    std::vector<float> output_x8_gate(rows), output_x8_up(rows);
    std::vector<float> output_x16_gate(rows), output_x16_up(rows);
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
    ggml_repack_cq3_0_to_cq3_0x16(gate_weights.data(), gate_x16.data(), rows, blocks_per_row);
    ggml_repack_cq3_0_to_cq3_0x16(up_weights.data(), up_x16.data(), rows, blocks_per_row);

    auto run_ordinary = [&]() {
        qwen38_run_cq3_x8_control(rows, columns, gate_weights.data(), activation.data(), ordinary_gate.data());
        qwen38_run_cq3_x8_control(rows, columns, up_weights.data(), activation.data(), ordinary_up.data());
    };
    auto run_x8 = [&]() {
        ggml_gemv_cq3_0x8_cq8_0(columns, output_x8_gate.data(), gate_x8.data(), activation.data(), rows);
        ggml_gemv_cq3_0x8_cq8_0(columns, output_x8_up.data(), up_x8.data(), activation.data(), rows);
    };
    auto run_x16 = [&]() {
        ggml_gemv_cq3_0x16_cq8_0(columns, output_x16_gate.data(), gate_x16.data(), activation.data(), rows);
        ggml_gemv_cq3_0x16_cq8_0(columns, output_x16_up.data(), up_x16.data(), activation.data(), rows);
    };

    run_ordinary();
    run_x8();
    run_x16();
    assert(std::memcmp(output_x8_gate.data(), output_x16_gate.data(), rows*sizeof(float)) == 0);
    assert(std::memcmp(output_x8_up.data(), output_x16_up.data(), rows*sizeof(float)) == 0);
    float max_scaled_delta = 0.0f;
    for (int row = 0; row < rows; ++row) {
        max_scaled_delta = std::max(max_scaled_delta,
                qwen38_cq3_scaled_delta(ordinary_gate[row], output_x16_gate[row]));
        max_scaled_delta = std::max(max_scaled_delta,
                qwen38_cq3_scaled_delta(ordinary_up[row], output_x16_up[row]));
        assert(std::isfinite(output_x16_gate[row]));
        assert(std::isfinite(output_x16_up[row]));
    }
    assert(max_scaled_delta <= 5e-5f);

    for (int warmup = 0; warmup < warmup_pairs; ++warmup) {
        if ((warmup & 1) == 0) {
            run_x8(); run_x16(); run_ordinary(); run_x16();
        } else {
            run_x16(); run_x8(); run_x16(); run_ordinary();
        }
    }

    std::vector<double> x8_us, x16_for_x8_us, ordinary_us, x16_for_ordinary_us;
    std::vector<double> x8_over_x16_ratios, ordinary_over_x16_ratios;
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
            candidate = measure(run_x16, output_x16_gate, output_x16_up);
        } else {
            candidate = measure(run_x16, output_x16_gate, output_x16_up);
            control = measure(run_x8, output_x8_gate, output_x8_up);
        }
        x8_pair_frequency_after_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        x8_us.push_back(control);
        x16_for_x8_us.push_back(candidate);
        x8_over_x16_ratios.push_back(control/candidate);
    };
    auto measure_ordinary_pair = [&](bool ab_order) {
        ordinary_pair_frequency_before_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        double control = 0.0;
        double candidate = 0.0;
        if (ab_order) {
            control = measure(run_ordinary, ordinary_gate, ordinary_up);
            candidate = measure(run_x16, output_x16_gate, output_x16_up);
        } else {
            candidate = measure(run_x16, output_x16_gate, output_x16_up);
            control = measure(run_ordinary, ordinary_gate, ordinary_up);
        }
        ordinary_pair_frequency_after_mhz.push_back(qwen38_cq3_x19_frequency_mhz());
        ordinary_us.push_back(control);
        x16_for_ordinary_us.push_back(candidate);
        ordinary_over_x16_ratios.push_back(control/candidate);
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

    const double x8_ratio_median = qwen38_cq3_x16_median(x8_over_x16_ratios);
    const double ordinary_ratio_median = qwen38_cq3_x16_median(ordinary_over_x16_ratios);
    const double x8_ratio_mad_fraction = qwen38_cq3_x19_mad_fraction(x8_over_x16_ratios);
    const double ordinary_ratio_mad_fraction = qwen38_cq3_x19_mad_fraction(ordinary_over_x16_ratios);
    std::printf("{\"schema\":\"qwen38_27b_cq3_x16_paired_confirmation_benchmark_v19\",");
    std::printf("\"rows_per_matrix\":%d,\"matrices\":2,\"columns\":%d,", rows, columns);
    std::printf("\"warmup_pairs\":%d,\"measured_pairs\":%d,\"cache_eviction_bytes\":%zu,",
            warmup_pairs, measured_pairs, cache_eviction_bytes);
    std::printf("\"source_bytes\":%zu,\"x8_bytes\":%zu,\"x16_bytes\":%zu,",
            2*gate_weights.size()*sizeof(block_cq3_0), 2*gate_x8.size()*sizeof(block_cq3_0x8),
            2*gate_x16.size()*sizeof(block_cq3_0x16));
    std::printf("\"x8_x16_bit_exact\":true,\"max_scaled_delta_vs_ordinary\":%.9g,", (double) max_scaled_delta);
    std::printf("\"x8_us\":"); qwen38_print_cq3_x16_values(x8_us);
    std::printf(",\"x16_for_x8_us\":"); qwen38_print_cq3_x16_values(x16_for_x8_us);
    std::printf(",\"ordinary_us\":"); qwen38_print_cq3_x16_values(ordinary_us);
    std::printf(",\"x16_for_ordinary_us\":"); qwen38_print_cq3_x16_values(x16_for_ordinary_us);
    std::printf(",\"x8_over_x16_pairwise_ratios\":"); qwen38_print_cq3_x16_values(x8_over_x16_ratios);
    std::printf(",\"ordinary_over_x16_pairwise_ratios\":"); qwen38_print_cq3_x16_values(ordinary_over_x16_ratios);
    std::printf(",\"x8_pair_frequency_before_mhz\":"); qwen38_print_cq3_x16_values(x8_pair_frequency_before_mhz);
    std::printf(",\"x8_pair_frequency_after_mhz\":"); qwen38_print_cq3_x16_values(x8_pair_frequency_after_mhz);
    std::printf(",\"ordinary_pair_frequency_before_mhz\":"); qwen38_print_cq3_x16_values(ordinary_pair_frequency_before_mhz);
    std::printf(",\"ordinary_pair_frequency_after_mhz\":"); qwen38_print_cq3_x16_values(ordinary_pair_frequency_after_mhz);
    std::printf(",\"x8_over_x16_pairwise_ratio_median\":%.9f,", x8_ratio_median);
    std::printf("\"ordinary_over_x16_pairwise_ratio_median\":%.9f,", ordinary_ratio_median);
    std::printf("\"x8_over_x16_pairwise_ratio_mad_fraction\":%.9f,", x8_ratio_mad_fraction);
    std::printf("\"ordinary_over_x16_pairwise_ratio_mad_fraction\":%.9f,", ordinary_ratio_mad_fraction);
    std::printf("\"frequency_source\":\"sysfs_scaling_cur_freq_else_proc_cpuinfo_cpu_mhz\",\"sink\":%.9g}\n",
            (double) sink);
}
