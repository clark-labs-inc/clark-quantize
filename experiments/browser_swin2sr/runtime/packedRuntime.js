import { fetchPackedArtifactBundle } from "./packedBundle.js";

let runtimePromise;

async function initialize(modulePath, backend) {
  const module = await import(modulePath);
  await module.default();
  return {
    backend,
    decodeExternalData: module.decode_onnx_external_data,
    swin2srX2: module.swin2sr_x2,
  };
}

export async function loadPackedSwinRuntime({ allowSimd = true } = {}) {
  runtimePromise ??= allowSimd
    ? initialize(
      "./pkg-simd/clark_air_browser_runtime.js",
      "wasm-simd-packed-artifact",
    ).catch(() => initialize(
      "./pkg/clark_air_browser_runtime.js",
      "wasm-scalar-packed-artifact",
    ))
    : initialize(
      "./pkg/clark_air_browser_runtime.js",
      "wasm-scalar-packed-artifact",
    );
  try {
    return await runtimePromise;
  } catch (error) {
    runtimePromise = undefined;
    throw error;
  }
}

export async function loadPackedSwinArtifact(
  bundleUrl,
  { allowSimd = true, fetchOptions } = {},
) {
  const [runtime, artifact] = await Promise.all([
    loadPackedSwinRuntime({ allowSimd }),
    fetchPackedArtifactBundle(bundleUrl, fetchOptions),
  ]);
  return { ...runtime, ...artifact };
}

function externalDataLayout(artifact, externalDataBytes) {
  if (!artifact.graphBytes?.byteLength || !artifact.layoutBytes?.byteLength) {
    throw new Error("The packed AI model has no WebGPU graph");
  }
  const layout = JSON.parse(new TextDecoder().decode(artifact.layoutBytes));
  if (
    layout.format !== "backdropkit-swin2sr-onnx-external-layout-v1"
    || !layout.external_data_name
    || layout.external_data_bytes !== externalDataBytes.byteLength
  ) {
    throw new Error("The packed AI model has an invalid WebGPU layout");
  }
  return layout;
}

export async function createPackedSwinWebGpuRunner(ort, artifact) {
  const externalDataBytes = artifact.externalDataBytes ?? artifact.decodeExternalData(
    artifact.manifestBytes,
    artifact.blobBytes,
    artifact.layoutBytes,
  );
  const layout = externalDataLayout(artifact, externalDataBytes);
  const session = await ort.InferenceSession.create(artifact.graphBytes, {
    executionProviders: ["webgpu"],
    externalData: [{
      data: externalDataBytes,
      path: layout.external_data_name,
    }],
    graphOptimizationLevel: "all",
  });
  return {
    backend: "webgpu-external-data",
    async run(input, height, width) {
      const output = await session.run({
        pixel_values: new ort.Tensor("float32", input, [1, 3, height, width]),
      });
      return output.reconstruction.data;
    },
    release() {
      return session.release?.();
    },
  };
}

function packedWasmRunner(artifact) {
  return {
    backend: artifact.backend,
    run(input, height, width) {
      return artifact.swin2srX2(
        artifact.manifestBytes,
        artifact.blobBytes,
        input,
        height,
        width,
      );
    },
    release() {},
  };
}

export async function selectPackedSwinRunner({
  artifact,
  onFallback,
  ort,
  tryWebGpu = true,
}) {
  if (tryWebGpu && ort && artifact.graphBytes?.byteLength) {
    try {
      return await createPackedSwinWebGpuRunner(ort, artifact);
    } catch (error) {
      onFallback?.(error);
    }
  }
  return packedWasmRunner(artifact);
}

export async function createPackedSwinRunner({
  allowSimd = true,
  bundleUrl,
  fetchOptions,
  onFallback,
  ort,
  preferWebGpu = true,
}) {
  const artifact = await loadPackedSwinArtifact(bundleUrl, {
    allowSimd,
    fetchOptions,
  });
  return selectPackedSwinRunner({
    artifact,
    onFallback,
    ort,
    tryWebGpu: preferWebGpu
      && typeof navigator !== "undefined"
      && Boolean(navigator.gpu),
  });
}
