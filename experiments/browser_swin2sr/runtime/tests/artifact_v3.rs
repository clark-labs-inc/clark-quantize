use clark_air_browser_runtime::BrowserArtifact;


fn fixture(index_byte: u8) -> (Vec<u8>, Vec<u8>) {
    let manifest = br#"{
      "format":"clark-air-browser-v3",
      "mode":"ternary",
      "group_size":3,
      "tensors":{
        "weight":{
          "kind":"ternary",
          "shape":[2,3],
          "pad":14,
          "group_size":2,
          "codes":{"dtype":"uint32","offset":0,"bytes":4},
          "scale_codec":"codebook_fp16",
          "scale_count":3,
          "scale_index_bits":1,
          "scale_codebook":{"dtype":"float16","offset":4,"bytes":4},
          "scale_indices":{"dtype":"packed_uint","offset":8,"bytes":1}
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
    blob.extend_from_slice(&0x3c00_u16.to_le_bytes());
    blob.extend_from_slice(&0x4000_u16.to_le_bytes());
    blob.push(index_byte);
    (manifest.to_vec(), blob)
}


#[test]
fn v3_decodes_bitpacked_scale_codebooks_once_at_load() {
    let (manifest, blob) = fixture(0b0000_0010);
    let artifact = BrowserArtifact::from_bytes(&manifest, &blob).unwrap();
    let tensor = artifact.packed_ternary("weight").unwrap();

    assert_eq!(tensor.scales, [1.0, 2.0, 1.0]);
    assert_eq!(tensor.decode_f32().unwrap(), [-1.0, 0.0, 2.0, 2.0, -1.0, 0.0]);
}


#[test]
fn v3_rejects_scale_indices_outside_the_codebook() {
    let (manifest, mut blob) = fixture(0b0000_0010);
    blob.drain(6..8);
    let manifest = String::from_utf8(manifest)
        .unwrap()
        .replace("\"offset\":8", "\"offset\":6")
        .replace("\"bytes\":4},\n          \"scale_indices", "\"bytes\":2},\n          \"scale_indices");
    let artifact = BrowserArtifact::from_bytes(manifest.as_bytes(), &blob).unwrap();
    let error = artifact.packed_ternary("weight").err().unwrap();
    assert!(error.contains("index exceeds"));
}
