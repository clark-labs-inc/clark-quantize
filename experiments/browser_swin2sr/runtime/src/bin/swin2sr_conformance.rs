use std::{env, fs, path::PathBuf, time::Instant};

use clark_air_browser_runtime::{swin2sr_x2_core, BrowserArtifact};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Fixture {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input: Vec<f32>,
    expected: Vec<f32>,
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 3 {
        return Err("usage: swin2sr_conformance ARTIFACT_DIR FIXTURE_JSON".into());
    }
    let artifact_directory = PathBuf::from(&arguments[1]);
    let manifest = fs::read(artifact_directory.join("model.json"))
        .map_err(|error| format!("read manifest: {error}"))?;
    let blob = fs::read(artifact_directory.join("model.bin"))
        .map_err(|error| format!("read model blob: {error}"))?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(&arguments[2]).map_err(|error| format!("read fixture: {error}"))?,
    )
    .map_err(|error| format!("parse fixture: {error}"))?;
    if fixture.input_shape.len() != 4
        || fixture.output_shape.len() != 4
        || fixture.input_shape[0] != 1
        || fixture.input_shape[1] != 3
    {
        return Err("fixture shape is not one NCHW RGB image".into());
    }
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob)?;
    let started = Instant::now();
    let output = swin2sr_x2_core(
        &artifact,
        &fixture.input,
        fixture.input_shape[2],
        fixture.input_shape[3],
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    if output.len() != fixture.expected.len() {
        return Err("runtime output length does not match fixture".into());
    }
    let mut maximum_error = 0.0_f32;
    let mut total_error = 0.0_f64;
    for (actual, expected) in output.iter().zip(&fixture.expected) {
        let error = (actual - expected).abs();
        maximum_error = maximum_error.max(error);
        total_error += error as f64;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "native-rust-scalar",
            "elapsed_seconds": elapsed,
            "input_shape": fixture.input_shape,
            "output_shape": fixture.output_shape,
            "maximum_absolute_error": maximum_error,
            "mean_absolute_error": total_error / output.len() as f64,
        }))
        .map_err(|error| format!("serialize result: {error}"))?
    );
    Ok(())
}
