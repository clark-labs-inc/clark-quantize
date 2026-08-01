use std::time::Instant;

use clark_air_browser_runtime::ternary_matmul_core;
use serde::Serialize;

#[derive(Serialize)]
struct Benchmark {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    batch_size: usize,
    group_size: usize,
    repetitions: usize,
    median_milliseconds: f64,
    effective_gigaweights_per_second: f64,
    checksum: f64,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    batch_size: usize,
}

fn pack(trits: &[u32]) -> Vec<u32> {
    trits
        .chunks(20)
        .map(|chunk| {
            let mut packed = 0;
            let mut place = 1;
            for trit in chunk {
                packed += trit * place;
                place *= 3;
            }
            packed
        })
        .collect()
}

fn benchmark(case: Case) -> Benchmark {
    let out_features = case.out_features;
    let in_features = case.in_features;
    let batch_size = case.batch_size;
    let group_size: usize = 60;
    let repetitions: usize = 21;
    let weight_count = out_features * in_features;
    let trits: Vec<u32> = (0..weight_count)
        .map(|index| ((index * 7 + index / 17 + 2) % 3) as u32)
        .collect();
    let packed = pack(&trits);
    let scales: Vec<f32> = (0..weight_count.div_ceil(group_size))
        .map(|index| 0.015625 + (index % 31) as f32 / 512.0)
        .collect();
    let input: Vec<f32> = (0..batch_size * in_features)
        .map(|index| ((index * 11 % 29) as f32 - 14.0) / 13.0)
        .collect();
    let bias = vec![0.0; out_features];
    let run = || {
        ternary_matmul_core(
            &input,
            &packed,
            &scales,
            &bias,
            batch_size,
            out_features,
            in_features,
            group_size,
        )
        .expect("benchmark dimensions must be valid")
    };
    let _warmup = run();
    let mut timings = Vec::with_capacity(repetitions);
    let mut output = Vec::new();
    for _ in 0..repetitions {
        let started = Instant::now();
        output = run();
        timings.push(started.elapsed().as_secs_f64());
    }
    timings.sort_by(f64::total_cmp);
    let median = timings[timings.len() / 2];
    let weight_uses = (batch_size * weight_count) as f64;
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index + 1) as f64 * *value as f64)
        .sum();
    Benchmark {
        name: case.name,
        out_features,
        in_features,
        batch_size,
        group_size,
        repetitions,
        median_milliseconds: median * 1000.0,
        effective_gigaweights_per_second: weight_uses / median / 1e9,
        checksum,
    }
}

fn main() {
    // These are the actual selected projection shapes in BackdropKit's
    // 1,005,348-parameter Swin2SR model. Sixty-four rows represent one 8x8
    // attention window; 1,024 rows exercise a realistic multi-window crop.
    let cases = [
        Case {
            name: "attention_60x60_one_window",
            out_features: 60,
            in_features: 60,
            batch_size: 64,
        },
        Case {
            name: "mlp_up_120x60_one_window",
            out_features: 120,
            in_features: 60,
            batch_size: 64,
        },
        Case {
            name: "mlp_down_60x120_one_window",
            out_features: 60,
            in_features: 120,
            batch_size: 64,
        },
        Case {
            name: "attention_60x60_multi_window",
            out_features: 60,
            in_features: 60,
            batch_size: 1024,
        },
    ];
    let report: Vec<_> = cases.into_iter().map(benchmark).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize benchmark")
    );
}
