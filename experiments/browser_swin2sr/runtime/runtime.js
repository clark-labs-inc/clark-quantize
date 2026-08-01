import initWasm, { ternary_matmul as wasmTernaryMatmul } from "./pkg/clark_air_browser_runtime.js";

const WORKGROUP_SIZE = 64;
let wasmReady;
let webGpuReady;

function maxAbsoluteError(actual, expected) {
  let maximum = 0;
  for (let index = 0; index < expected.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
  }
  return maximum;
}

function createBuffer(device, typedArray, usage) {
  const size = Math.max(4, Math.ceil(typedArray.byteLength / 4) * 4);
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  const target = new Uint8Array(buffer.getMappedRange());
  target.set(new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength));
  buffer.unmap();
  return buffer;
}

async function runWasm(fixture) {
  wasmReady ??= initWasm();
  await wasmReady;
  const batchSize = fixture.batch_size ?? 1;
  const started = performance.now();
  const output = wasmTernaryMatmul(
    new Float32Array(fixture.input),
    new Uint32Array(fixture.packed),
    new Float32Array(fixture.scales),
    new Float32Array(fixture.bias),
    batchSize,
    fixture.shape[0],
    fixture.shape[1],
    fixture.group_size,
  );
  return { backend: "wasm", elapsedMs: performance.now() - started, output };
}

async function createWebGpuRuntime() {
  if (!navigator.gpu) {
    throw new Error("WebGPU API is unavailable");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("WebGPU adapter is unavailable");
  }
  const device = await adapter.requestDevice();
  const shader = await fetch("./src/ternary.wgsl", { cache: "no-store" })
    .then((response) => response.text());
  const module = device.createShaderModule({ code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map((message) => message.message).join("; "));
  }
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  device.lost.then(() => {
    webGpuReady = undefined;
  });
  return { device, pipeline };
}

async function getWebGpuRuntime() {
  webGpuReady ??= createWebGpuRuntime();
  try {
    return await webGpuReady;
  } catch (error) {
    webGpuReady = undefined;
    throw error;
  }
}

async function runWebGpu(fixture) {
  const { device, pipeline } = await getWebGpuRuntime();
  const input = createBuffer(
    device,
    new Float32Array(fixture.input),
    GPUBufferUsage.STORAGE,
  );
  const packed = createBuffer(
    device,
    new Uint32Array(fixture.packed),
    GPUBufferUsage.STORAGE,
  );
  const scales = createBuffer(
    device,
    new Float32Array(fixture.scales),
    GPUBufferUsage.STORAGE,
  );
  const bias = createBuffer(
    device,
    new Float32Array(fixture.bias),
    GPUBufferUsage.STORAGE,
  );
  const batchSize = fixture.batch_size ?? 1;
  const outputSize = batchSize * fixture.shape[0] * Float32Array.BYTES_PER_ELEMENT;
  const output = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const meta = createBuffer(
    device,
    new Uint32Array([
      fixture.shape[0],
      fixture.shape[1],
      fixture.group_size,
      fixture.bias.length > 0 ? 1 : 0,
      batchSize,
      0,
      0,
      0,
    ]),
    GPUBufferUsage.UNIFORM,
  );
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: packed } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: meta } },
    ],
  });

  const started = performance.now();
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(fixture.shape[0] / WORKGROUP_SIZE), batchSize);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputSize);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const values = new Float32Array(readback.getMappedRange().slice(0));
    return {
      backend: "webgpu",
      elapsedMs: performance.now() - started,
      output: values,
    };
  } finally {
    if (mapped) {
      readback.unmap();
    }
    for (const buffer of [input, packed, scales, bias, output, readback, meta]) {
      buffer.destroy();
    }
  }
}

export async function runFixture(fixture, preference = "auto") {
  let result;
  let fallbackReason = null;
  if (preference !== "wasm") {
    try {
      result = await runWebGpu(fixture);
    } catch (error) {
      fallbackReason = String(error?.message ?? error);
    }
  }
  result ??= await runWasm(fixture);
  return {
    backend: result.backend,
    elapsedMs: result.elapsedMs,
    maxAbsoluteError: maxAbsoluteError(result.output, fixture.expected),
    outputPreview: Array.from(result.output.slice(0, 8)),
    expectedPreview: fixture.expected.slice(0, 8),
    fallbackReason,
  };
}
