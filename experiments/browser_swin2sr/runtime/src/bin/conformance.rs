use clark_air_browser_runtime::ternary_matmul_core;

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

fn main() {
    let out_features: usize = 7;
    let in_features: usize = 61;
    let batch_size: usize = 3;
    let group_size: usize = 17;
    let weight_count = out_features * in_features;
    let trits: Vec<u32> = (0..weight_count)
        .map(|index| ((index * 7 + 2) % 3) as u32)
        .collect();
    let scales: Vec<f32> = (0..weight_count.div_ceil(group_size))
        .map(|index| 0.125 + index as f32 / 64.0)
        .collect();
    let inputs: Vec<f32> = (0..batch_size * in_features)
        .map(|index| ((index * 11 % 29) as f32 - 14.0) / 13.0)
        .collect();
    let bias: Vec<f32> = (0..out_features)
        .map(|index| index as f32 / 10.0 - 0.25)
        .collect();
    let output = ternary_matmul_core(
        &inputs,
        &pack(&trits),
        &scales,
        &bias,
        batch_size,
        out_features,
        in_features,
        group_size,
    )
    .expect("portable packed matmul fixture must run");

    let checksum: f32 = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index + 1) as f32 * value)
        .sum();
    assert!((checksum - 49.611_057).abs() < 1e-4);
    println!(
        "portable ternary matmul passed: outputs={} checksum={checksum:.6}",
        output.len()
    );
}
