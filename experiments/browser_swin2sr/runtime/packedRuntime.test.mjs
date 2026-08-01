import assert from "node:assert/strict";
import test from "node:test";

import {
  createPackedSwinWebGpuRunner,
  selectPackedSwinRunner,
} from "./packedRuntime.js";

function artifactFixture() {
  const layout = new TextEncoder().encode(JSON.stringify({
    format: "backdropkit-swin2sr-onnx-external-layout-v1",
    external_data_name: "weights.bin",
    external_data_bytes: 4,
  }));
  return {
    backend: "wasm-simd-packed-artifact",
    blobBytes: new Uint8Array([2]),
    externalDataBytes: new Uint8Array([3, 4, 5, 6]),
    graphBytes: new Uint8Array([1]),
    layoutBytes: layout,
    manifestBytes: new Uint8Array([0]),
    swin2srX2: (...arguments_) => arguments_,
  };
}

test("WebGPU runner mounts decoded weights and returns reconstruction data", async () => {
  const calls = [];
  const output = new Float32Array([1, 2, 3]);
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { data, dims, type });
    }
  }
  const ort = {
    Tensor,
    InferenceSession: {
      async create(graph, options) {
        calls.push({ graph, options });
        return {
          async run(feeds) {
            calls.push({ feeds });
            return { reconstruction: { data: output } };
          },
        };
      },
    },
  };

  const runner = await createPackedSwinWebGpuRunner(ort, artifactFixture());
  const result = await runner.run(new Float32Array(12), 2, 2);

  assert.equal(runner.backend, "webgpu-external-data");
  assert.equal(calls[0].options.executionProviders[0], "webgpu");
  assert.equal(calls[0].options.externalData[0].path, "weights.bin");
  assert.deepEqual(calls[1].feeds.pixel_values.dims, [1, 3, 2, 2]);
  assert.equal(result, output);
});

test("a WebGPU initialization failure falls back without losing packed execution", async () => {
  const fallbackErrors = [];
  const artifact = artifactFixture();
  const ort = {
    InferenceSession: {
      async create() {
        throw new Error("no adapter");
      },
    },
  };

  const runner = await selectPackedSwinRunner({
    artifact,
    onFallback: (error) => fallbackErrors.push(error.message),
    ort,
    tryWebGpu: true,
  });
  const input = new Float32Array(3);
  const result = runner.run(input, 1, 1);

  assert.equal(runner.backend, "wasm-simd-packed-artifact");
  assert.deepEqual(fallbackErrors, ["no adapter"]);
  assert.equal(result[0], artifact.manifestBytes);
  assert.equal(result[1], artifact.blobBytes);
  assert.equal(result[2], input);
});
