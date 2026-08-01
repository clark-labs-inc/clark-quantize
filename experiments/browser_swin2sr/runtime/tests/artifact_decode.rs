use clark_air_browser_runtime::BrowserArtifact;

fn fixture() -> (Vec<u8>, Vec<u8>) {
    let manifest = br#"{
      "format":"clark-air-browser-v1","mode":"ternary","group_size":3,
      "tensors":{
        "weight":{"kind":"ternary","shape":[2,3],"pad":14,
          "codes":{"dtype":"uint32","offset":0,"bytes":4},
          "scales":{"dtype":"float16","offset":4,"bytes":4}},
        "bias":{"kind":"raw","shape":[2],
          "data":{"dtype":"float16","offset":8,"bytes":4}}
      }
    }"#;
    let trits = [0_u32, 1, 2, 2, 0, 1];
    let packed = trits.into_iter().enumerate()
        .map(|(index, trit)| trit * 3_u32.pow(index as u32)).sum::<u32>();
    let mut blob = packed.to_le_bytes().to_vec();
    blob.extend_from_slice(&0x4000_u16.to_le_bytes());
    blob.extend_from_slice(&0x3800_u16.to_le_bytes());
    blob.extend_from_slice(&0x3c00_u16.to_le_bytes());
    blob.extend_from_slice(&0xbc00_u16.to_le_bytes());
    (manifest.to_vec(), blob)
}

#[test]
fn loads_packed_words_and_fp16_scales() {
    let (manifest, blob) = fixture();
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob).expect("valid artifact");
    let tensor = artifact.packed_ternary("weight").expect("packed tensor");
    assert_eq!(tensor.shape, [2, 3]);
    assert_eq!(tensor.pad, 14);
    assert_eq!(tensor.group_size, 3);
    assert_eq!(tensor.packed, [318]);
    assert_eq!(tensor.scales, [2.0, 0.5]);
    assert_eq!(tensor.decode_f32().unwrap(), [-2.0, 0.0, 2.0, 0.5, -0.5, 0.0]);
}

#[test]
fn v2_uses_tensor_local_group_size() {
    let manifest = br#"{"format":"clark-air-browser-v2","mode":"ternary",
      "group_size":3,"tensors":{"weight":{"kind":"ternary","shape":[2,3],
      "pad":14,"group_size":2,"codes":{"dtype":"uint32","offset":0,"bytes":4},
      "scales":{"dtype":"float16","offset":4,"bytes":6}}}}"#;
    let trits = [0_u32, 1, 2, 2, 0, 1];
    let packed = trits.into_iter().enumerate()
        .map(|(index, trit)| trit * 3_u32.pow(index as u32)).sum::<u32>();
    let mut blob = packed.to_le_bytes().to_vec();
    blob.extend_from_slice(&0x3c00_u16.to_le_bytes());
    blob.extend_from_slice(&0x4000_u16.to_le_bytes());
    blob.extend_from_slice(&0x4200_u16.to_le_bytes());
    let artifact = BrowserArtifact::from_bytes(manifest, &blob).expect("valid v2 artifact");
    assert_eq!(artifact.packed_ternary("weight").unwrap().decode_f32().unwrap(),
        [-1.0, 0.0, 2.0, 2.0, -3.0, 0.0]);
}

#[test]
fn v2_decodes_optional_affine_group_offsets() {
    let manifest = br#"{"format":"clark-air-browser-v2","mode":"ternary",
      "group_size":3,"tensors":{"weight":{"kind":"ternary","shape":[2,3],
      "pad":14,"group_size":3,"codes":{"dtype":"uint32","offset":0,"bytes":4},
      "scales":{"dtype":"float16","offset":4,"bytes":4},
      "offsets":{"dtype":"float16","offset":8,"bytes":4}}}}"#;
    let trits = [0_u32, 1, 2, 2, 0, 1];
    let packed = trits.into_iter().enumerate()
        .map(|(index, trit)| trit * 3_u32.pow(index as u32)).sum::<u32>();
    let mut blob = packed.to_le_bytes().to_vec();
    blob.extend_from_slice(&0x4000_u16.to_le_bytes());
    blob.extend_from_slice(&0x3800_u16.to_le_bytes());
    blob.extend_from_slice(&0x3400_u16.to_le_bytes());
    blob.extend_from_slice(&0xb800_u16.to_le_bytes());
    let artifact = BrowserArtifact::from_bytes(manifest, &blob).expect("valid affine artifact");
    assert_eq!(artifact.packed_ternary("weight").unwrap().decode_f32().unwrap(),
        [-1.75, 0.25, 2.25, 0.0, -1.0, -0.5]);
}

#[test]
fn v4_decodes_packed_int7_across_byte_boundaries() {
    let manifest = br#"{"format":"clark-air-browser-v4","mode":"mixed_ternary_int7",
      "group_size":3,"tensors":{"weight":{"kind":"symmetric_int","shape":[2,1,1,2],
      "bit_width":7,"zero_point":63,"granularity":"per_output",
      "codes":{"dtype":"packed_uint","offset":0,"bytes":4},
      "scales":{"dtype":"float16","offset":4,"bytes":4}}}}"#;
    let unsigned = [0_u16, 63, 126, 64];
    let mut packed = vec![0_u8; 4];
    for (index, value) in unsigned.into_iter().enumerate() {
        for bit in 0..7 {
            packed[(index * 7 + bit) / 8] |=
                (((value >> bit) & 1) as u8) << ((index * 7 + bit) % 8);
        }
    }
    packed.extend_from_slice(&0x4000_u16.to_le_bytes());
    packed.extend_from_slice(&0x3800_u16.to_le_bytes());
    let artifact = BrowserArtifact::from_bytes(manifest, &packed).expect("valid v4 artifact");
    let tensor = artifact.packed_symmetric_int("weight").unwrap();
    assert_eq!(tensor.decode_f32().unwrap(), [-126.0, 0.0, 31.5, 0.5]);
    assert_eq!(artifact.tensor_f32("weight").unwrap().values,
        [-126.0, 0.0, 31.5, 0.5]);
}

#[test]
fn rejects_out_of_bounds_payload() {
    let (manifest, mut blob) = fixture();
    blob.truncate(5);
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob).expect("valid manifest");
    let error = artifact.packed_ternary("weight").err().expect("must reject");
    assert!(error.contains("exceeds artifact"));
}

#[test]
fn loads_raw_fp16_tensor_as_f32() {
    let (manifest, blob) = fixture();
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob).expect("valid manifest");
    assert_eq!(artifact.raw_f32("bias").unwrap().values, [1.0, -1.0]);
}

#[test]
fn tensor_transform_overrides_legacy_default() {
    let (manifest, blob) = fixture();
    let manifest = String::from_utf8(manifest).unwrap()
        .replace("\"group_size\":3,", "\"group_size\":3,\"input_transform\":{\"kind\":\"none\"},")
        .replace("\"kind\":\"ternary\",", "\"kind\":\"ternary\",\"input_transform\":{\"kind\":\"block_hadamard\",\"block_size\":4},");
    let artifact = BrowserArtifact::from_bytes(manifest.as_bytes(), &blob).unwrap();
    assert_eq!(artifact.hadamard_block_size("weight"), Some(4));
    assert_eq!(artifact.hadamard_block_size("bias"), None);
}

#[test]
fn rejects_invalid_tensor_transform() {
    let (manifest, blob) = fixture();
    let manifest = String::from_utf8(manifest).unwrap().replace(
        "\"kind\":\"ternary\",",
        "\"kind\":\"ternary\",\"input_transform\":{\"kind\":\"block_hadamard\",\"block_size\":3},",
    );
    let error = BrowserArtifact::from_bytes(manifest.as_bytes(), &blob)
        .err().expect("must reject");
    assert!(error.contains("weight") && error.contains("power of two"));
}
