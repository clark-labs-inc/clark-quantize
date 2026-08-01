struct Meta {
  shape: vec4<u32>,
  batch: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> dimensions: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let batch = id.y;
  if (row >= dimensions.shape.x || batch >= dimensions.batch.x) {
    return;
  }
  var sum = 0.0;
  if (dimensions.shape.w != 0u) {
    sum = bias[row];
  }
  let row_offset = row * dimensions.shape.y;
  var packed_index = row_offset / 20u;
  let packed_offset = row_offset % 20u;
  var word = packed[packed_index];
  for (var position = 0u; position < packed_offset; position += 1u) {
    word /= 3u;
  }
  for (var column = 0u; column < dimensions.shape.y; column += 1u) {
    let index = row_offset + column;
    if (column > 0u && index % 20u == 0u) {
      packed_index += 1u;
      word = packed[packed_index];
    }
    let code = f32(word % 3u) - 1.0;
    word /= 3u;
    sum += input[batch * dimensions.shape.y + column]
      * code
      * scales[index / dimensions.shape.z];
  }
  output[batch * dimensions.shape.x + row] = sum;
}
