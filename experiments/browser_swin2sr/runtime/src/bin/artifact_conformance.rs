use std::{env, fs};

use clark_air_browser_runtime::{ternary_matmul_affine_core, BrowserArtifact};
use serde::Serialize;

#[derive(Serialize)]
struct Receipt {
    tensor: String,
    shape: Vec<usize>,
    group_size: usize,
    batch_size: usize,
    output: Vec<f32>,
    checksum: f64,
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 4 {
        eprintln!("usage: artifact_conformance MODEL_JSON MODEL_BIN TENSOR");
        std::process::exit(2);
    }
    let manifest = fs::read(&arguments[1]).expect("read model manifest");
    let blob = fs::read(&arguments[2]).expect("read model binary");
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob).expect("load artifact");
    let tensor = artifact
        .packed_ternary(&arguments[3])
        .expect("load packed ternary tensor");
    assert_eq!(tensor.shape.len(), 2, "conformance tensor must be a matrix");
    let out_features = tensor.shape[0];
    let in_features = tensor.shape[1];
    let batch_size = 3;
    let input: Vec<f32> = (0..batch_size * in_features)
        .map(|index| ((index * 11 % 29) as f32 - 14.0) / 13.0)
        .collect();
    let output = ternary_matmul_affine_core(
        &input,
        &tensor.packed,
        &tensor.scales,
        &tensor.offsets,
        &[],
        batch_size,
        out_features,
        in_features,
        tensor.group_size,
    )
    .expect("run artifact tensor");
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index + 1) as f64 * *value as f64)
        .sum();
    let receipt = Receipt {
        tensor: arguments[3].clone(),
        shape: tensor.shape,
        group_size: tensor.group_size,
        batch_size,
        output,
        checksum,
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).expect("serialize receipt")
    );
}
