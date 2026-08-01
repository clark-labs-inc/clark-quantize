use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clark_air_browser_runtime::decode_onnx_external_data_core;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 4 {
        return Err(
            "usage: decode_external_data MANIFEST BLOB LAYOUT OUTPUT"
                .to_string()
                .into(),
        );
    }
    let manifest = fs::read(PathBuf::from(&arguments[0]))?;
    let blob = fs::read(PathBuf::from(&arguments[1]))?;
    let layout = fs::read(PathBuf::from(&arguments[2]))?;
    let started = Instant::now();
    let output = decode_onnx_external_data_core(&manifest, &blob, &layout)?;
    let elapsed = started.elapsed();
    fs::write(PathBuf::from(&arguments[3]), &output)?;
    println!(
        "{{\"external_data_bytes\":{},\"decode_milliseconds\":{:.3}}}",
        output.len(),
        elapsed.as_secs_f64() * 1000.0,
    );
    Ok(())
}
