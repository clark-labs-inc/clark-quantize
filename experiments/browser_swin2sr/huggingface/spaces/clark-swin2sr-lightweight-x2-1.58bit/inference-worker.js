const ORT_VERSION = "1.27.0";
const ORT_MODULE =
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
const ORT_WASM_PATH =
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_REVISION = "b357304af178a6e31a5482958f1adf702f99be82";
const MODEL_ROOT =
  "https://huggingface.co/clark-labs/clark-swin2sr-lightweight-x2-1.58bit";
const MODEL_GZIP_URL =
  `${MODEL_ROOT}/resolve/${MODEL_REVISION}/onnx/model.onnx.gz`;
const MODEL_URL =
  `${MODEL_ROOT}/resolve/${MODEL_REVISION}/onnx/model.onnx`;

let ort = null;
let session = null;
let backend = null;

function postStatus(percent, title, detail) {
  self.postMessage({ type: "status", percent, title, detail });
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function fetchModelBytes() {
  if ("DecompressionStream" in self) {
    const response = await fetch(MODEL_GZIP_URL, { cache: "force-cache" });
    if (!response.ok) throw new Error(`Model download failed (${response.status})`);
    const total = Number(response.headers.get("content-length")) || 808_002;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      postStatus(
        Math.min(32, (received / total) * 32),
        "Downloading 808 KB local AI",
        `${formatBytes(received)} cached on this device.`,
      );
    }

    const compressed = new Blob(chunks);
    const stream = compressed.stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
  }

  postStatus(
    8,
    "Downloading local AI",
    "This browser uses the compatible uncompressed model.",
  );
  const response = await fetch(MODEL_URL, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  return response.arrayBuffer();
}

async function initialize(forceWasm) {
  if (session) return { backend };

  const [ortModule, modelBytes] = await Promise.all([
    import(ORT_MODULE),
    fetchModelBytes(),
  ]);
  ort = ortModule;
  ort.env.wasm.wasmPaths = ORT_WASM_PATH;
  ort.env.wasm.numThreads =
    self.crossOriginIsolated && navigator.hardwareConcurrency > 1
      ? Math.min(4, navigator.hardwareConcurrency)
      : 1;

  backend = "WASM";
  if (navigator.gpu && !forceWasm) {
    try {
      postStatus(40, "Starting WebGPU", "Compiling the model for your GPU.");
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (adapter) {
        session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        });
        backend = "WebGPU";
      }
    } catch (error) {
      console.info("WebGPU unavailable in the worker; using WASM.", error);
    }
  }

  if (!session) {
    postStatus(52, "Starting compatible mode", "Preparing the WASM CPU engine.");
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  postStatus(72, "Warming local AI", "Running a tiny private first pass.");
  const warmPixels = new Float32Array(3 * 8 * 8);
  const warmInput = new ort.Tensor("float32", warmPixels, [1, 3, 8, 8]);
  await session.run({ pixel_values: warmInput });
  return { backend };
}

async function runTile({ pixels, width, height }) {
  if (!session || !ort) throw new Error("Local AI is not ready");
  const input = new ort.Tensor(
    "float32",
    new Float32Array(pixels),
    [1, 3, height, width],
  );
  const result = await session.run({ pixel_values: input });
  const values = result.reconstruction.data;
  const output = values.slice().buffer;
  return { output };
}

self.addEventListener("message", async (event) => {
  const { id, type } = event.data;
  try {
    if (type === "initialize") {
      const result = await initialize(event.data.forceWasm);
      self.postMessage({ type: "response", id, result });
      return;
    }
    if (type === "run") {
      const result = await runTile(event.data);
      self.postMessage({ type: "response", id, result }, [result.output]);
      return;
    }
    throw new Error(`Unknown worker request: ${type}`);
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      message: error?.message || "Local AI worker failed",
    });
  }
});
