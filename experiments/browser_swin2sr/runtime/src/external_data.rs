use std::collections::BTreeSet;

use serde::Deserialize;

use crate::artifact::BrowserArtifact;

const FORMAT: &str = "backdropkit-swin2sr-onnx-external-layout-v1";
const ONNX_FLOAT: usize = 1;

#[derive(Deserialize)]
struct ExternalLayout {
    format: String,
    external_data_bytes: usize,
    initializers: usize,
    entries: Vec<ExternalEntry>,
}

#[derive(Deserialize)]
struct ExternalEntry {
    state_name: String,
    initializer_name: String,
    layout: String,
    state_shape: Vec<usize>,
    initializer_shape: Vec<usize>,
    data_type: usize,
    offset: usize,
    length: usize,
}

fn checked_product(shape: &[usize], label: &str) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!("{label} shape must contain positive dimensions"));
    }
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{label} shape overflows"))
    })
}

fn validate_layout(
    layout: &ExternalLayout,
    artifact: &BrowserArtifact<'_>,
) -> Result<(), String> {
    if layout.format != FORMAT {
        return Err(format!("unsupported external-data layout: {}", layout.format));
    }
    if layout.external_data_bytes == 0 {
        return Err("external-data output must not be empty".to_string());
    }
    if layout.initializers != layout.entries.len() {
        return Err("external-data initializer count differs from its entries".to_string());
    }
    let mut state_names = BTreeSet::new();
    let mut initializer_names = BTreeSet::new();
    let mut ranges = Vec::with_capacity(layout.entries.len());
    for entry in &layout.entries {
        if !state_names.insert(entry.state_name.as_str()) {
            return Err(format!("duplicate external-data state tensor: {}", entry.state_name));
        }
        if !initializer_names.insert(entry.initializer_name.as_str()) {
            return Err(format!(
                "duplicate external-data initializer: {}",
                entry.initializer_name
            ));
        }
        if entry.data_type != ONNX_FLOAT {
            return Err(format!(
                "external-data initializer is not float32: {}",
                entry.initializer_name
            ));
        }
        let state_values = checked_product(&entry.state_shape, &entry.state_name)?;
        let initializer_values = checked_product(
            &entry.initializer_shape,
            &entry.initializer_name,
        )?;
        if state_values != initializer_values {
            return Err(format!(
                "state and initializer sizes differ: {}",
                entry.state_name
            ));
        }
        let expected_length = initializer_values
            .checked_mul(4)
            .ok_or_else(|| format!("initializer byte size overflows: {}", entry.initializer_name))?;
        if entry.length != expected_length {
            return Err(format!(
                "initializer byte length differs: {}",
                entry.initializer_name
            ));
        }
        match entry.layout.as_str() {
            "direct" if entry.state_shape != entry.initializer_shape => {
                return Err(format!("direct initializer shape differs: {}", entry.state_name));
            }
            "transpose"
                if entry.state_shape.len() != 2
                    || entry.initializer_shape
                        != [entry.state_shape[1], entry.state_shape[0]] =>
            {
                return Err(format!("transpose initializer shape differs: {}", entry.state_name));
            }
            "direct" | "transpose" => {}
            value => return Err(format!("unsupported initializer layout: {value}")),
        }
        let end = entry
            .offset
            .checked_add(entry.length)
            .ok_or_else(|| format!("initializer byte range overflows: {}", entry.initializer_name))?;
        if end > layout.external_data_bytes {
            return Err(format!(
                "initializer byte range exceeds external data: {}",
                entry.initializer_name
            ));
        }
        ranges.push((entry.offset, end, entry.initializer_name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    let mut expected_offset = 0;
    for (start, end, name) in ranges {
        if start != expected_offset {
            return Err(format!("external-data ranges overlap or have a gap before {name}"));
        }
        expected_offset = end;
    }
    if expected_offset != layout.external_data_bytes {
        return Err("external-data ranges do not cover the output".to_string());
    }
    let artifact_names: BTreeSet<_> = artifact.tensor_names().collect();
    if artifact_names != state_names {
        return Err("external-data layout does not cover the artifact tensors".to_string());
    }
    Ok(())
}

fn write_f32(output: &mut [u8], value_index: usize, value: f32) {
    let offset = value_index * 4;
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub fn decode_onnx_external_data_core(
    manifest_json: &[u8],
    blob: &[u8],
    layout_json: &[u8],
) -> Result<Vec<u8>, String> {
    let artifact = BrowserArtifact::from_bytes(manifest_json, blob)?;
    let layout: ExternalLayout = serde_json::from_slice(layout_json)
        .map_err(|error| format!("invalid external-data layout: {error}"))?;
    validate_layout(&layout, &artifact)?;
    let mut output = vec![0_u8; layout.external_data_bytes];
    for entry in &layout.entries {
        let tensor = artifact.tensor_f32(&entry.state_name)?;
        if tensor.shape != entry.state_shape {
            return Err(format!("artifact tensor shape differs: {}", entry.state_name));
        }
        let destination = &mut output[entry.offset..entry.offset + entry.length];
        if entry.layout == "direct" {
            for (index, value) in tensor.values.into_iter().enumerate() {
                write_f32(destination, index, value);
            }
        } else {
            let rows = entry.state_shape[0];
            let columns = entry.state_shape[1];
            for row in 0..rows {
                for column in 0..columns {
                    let source_index = row * columns + column;
                    let destination_index = column * rows + row;
                    write_f32(destination, destination_index, tensor.values[source_index]);
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::decode_onnx_external_data_core;

    fn artifact_fixture() -> (Vec<u8>, Vec<u8>) {
        let manifest = br#"{
          "format":"clark-air-browser-v1",
          "mode":"ternary",
          "group_size":3,
          "tensors":{
            "bias":{
              "kind":"raw",
              "shape":[2],
              "data":{"dtype":"float16","offset":8,"bytes":4}
            },
            "weight":{
              "kind":"ternary",
              "shape":[2,3],
              "pad":14,
              "codes":{"dtype":"uint32","offset":0,"bytes":4},
              "scales":{"dtype":"float16","offset":4,"bytes":4}
            }
          }
        }"#;
        let trits = [0_u32, 1, 2, 2, 0, 1];
        let packed = trits
            .into_iter()
            .enumerate()
            .map(|(index, trit)| trit * 3_u32.pow(index as u32))
            .sum::<u32>();
        let mut blob = packed.to_le_bytes().to_vec();
        blob.extend_from_slice(&0x4000_u16.to_le_bytes());
        blob.extend_from_slice(&0x3800_u16.to_le_bytes());
        blob.extend_from_slice(&0x3c00_u16.to_le_bytes());
        blob.extend_from_slice(&0xbc00_u16.to_le_bytes());
        (manifest.to_vec(), blob)
    }

    fn decode_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn reconstructs_direct_and_transposed_onnx_initializers() {
        let (manifest, blob) = artifact_fixture();
        let layout = br#"{
          "format":"backdropkit-swin2sr-onnx-external-layout-v1",
          "external_data_bytes":32,
          "initializers":2,
          "entries":[
            {
              "state_name":"weight","initializer_name":"matmul.weight",
              "layout":"transpose","state_shape":[2,3],
              "initializer_shape":[3,2],"data_type":1,"offset":0,"length":24
            },
            {
              "state_name":"bias","initializer_name":"matmul.bias",
              "layout":"direct","state_shape":[2],
              "initializer_shape":[2],"data_type":1,"offset":24,"length":8
            }
          ]
        }"#;

        let output = decode_onnx_external_data_core(&manifest, &blob, layout).unwrap();

        assert_eq!(
            decode_values(&output),
            [-2.0, 0.5, 0.0, -0.5, 2.0, 0.0, 1.0, -1.0]
        );
    }

    #[test]
    fn rejects_a_layout_with_a_gap() {
        let (manifest, blob) = artifact_fixture();
        let layout = br#"{
          "format":"backdropkit-swin2sr-onnx-external-layout-v1",
          "external_data_bytes":36,
          "initializers":2,
          "entries":[
            {
              "state_name":"weight","initializer_name":"weight",
              "layout":"direct","state_shape":[2,3],
              "initializer_shape":[2,3],"data_type":1,"offset":0,"length":24
            },
            {
              "state_name":"bias","initializer_name":"bias",
              "layout":"direct","state_shape":[2],
              "initializer_shape":[2],"data_type":1,"offset":28,"length":8
            }
          ]
        }"#;

        let error = decode_onnx_external_data_core(&manifest, &blob, layout).unwrap_err();

        assert!(error.contains("gap"));
    }
}
