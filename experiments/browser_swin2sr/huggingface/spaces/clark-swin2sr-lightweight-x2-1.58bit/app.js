const FORCE_WASM =
  new URLSearchParams(window.location.search).get("backend") === "wasm";
const TILE_SIZE = 256;
const TILE_OVERLAP = 24;
const PREVIEW_EDGE = 1400;
const MAX_CANVAS_EDGE = 16_384;

const elements = Object.fromEntries(
  [
    "runtimePill",
    "runtimeLabel",
    "dropZone",
    "dropCopy",
    "fileInput",
    "sourcePreview",
    "replaceButton",
    "fileSummary",
    "fileName",
    "fileDetails",
    "runButton",
    "cancelButton",
    "progressCard",
    "progressPercent",
    "progressTrack",
    "progressFill",
    "statusTitle",
    "statusDetail",
    "outputDimensions",
    "bicubicCanvas",
    "modelCanvas",
    "bicubicEmpty",
    "modelEmpty",
    "tileProgress",
    "downloadBicubic",
    "downloadModel",
    "toast",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  source: null,
  sourceUrl: null,
  sourceName: "image",
  worker: null,
  workerRequests: new Map(),
  nextWorkerRequestId: 1,
  modelReady: false,
  backend: null,
  modelPromise: null,
  running: false,
  cancelRequested: false,
  finished: false,
};

function setProgress(percent, title, detail) {
  const rounded = Math.max(0, Math.min(100, Math.round(percent)));
  elements.progressPercent.textContent = `${rounded}%`;
  elements.progressFill.style.width = `${rounded}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(rounded));
  elements.statusTitle.textContent = title;
  elements.statusDetail.textContent = detail;
}

function setRuntime(label, mode = "") {
  elements.runtimeLabel.textContent = label;
  elements.runtimePill.classList.toggle("ready", mode === "ready");
  elements.runtimePill.classList.toggle("error", mode === "error");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 4200);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 1) return "a few seconds";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.ceil(seconds % 60);
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function safeBaseName(name) {
  return (
    name
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-z0-9_-]+/gi, "-")
      .replace(/^-+|-+$/g, "") || "image"
  );
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function rejectWorkerRequests(error) {
  for (const request of state.workerRequests.values()) {
    window.clearTimeout(request.timeout);
    request.reject(error);
  }
  state.workerRequests.clear();
}

function stopInferenceWorker(error) {
  state.worker?.terminate();
  state.worker = null;
  state.modelReady = false;
  rejectWorkerRequests(error);
}

function createInferenceWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(
    new URL("./inference-worker.js?v=20260801-worker1", import.meta.url),
    { type: "module" },
  );
  worker.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "status") {
      setProgress(message.percent, message.title, message.detail);
      return;
    }
    const request = state.workerRequests.get(message.id);
    if (!request) return;
    state.workerRequests.delete(message.id);
    window.clearTimeout(request.timeout);
    if (message.type === "error") {
      request.reject(new Error(message.message));
    } else {
      request.resolve(message.result);
    }
  });
  worker.addEventListener("error", () => {
    stopInferenceWorker(new Error("Local AI worker stopped"));
  });
  state.worker = worker;
  return worker;
}

function sendWorkerRequest(type, payload = {}, transfer = [], timeoutMs = 120_000) {
  const worker = createInferenceWorker();
  const id = state.nextWorkerRequestId;
  state.nextWorkerRequestId += 1;
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      stopInferenceWorker(new Error("Local AI took too long and was stopped"));
    }, timeoutMs);
    state.workerRequests.set(id, { resolve, reject, timeout });
    worker.postMessage({ ...payload, id, type }, transfer);
  });
}

async function initializeModel() {
  try {
    setRuntime("Loading local AI…");
    const { backend } = await sendWorkerRequest("initialize", {
      forceWasm: FORCE_WASM,
    });
    state.backend = backend;
    state.modelReady = true;
    setRuntime(`${backend} ready`, "ready");
    setProgress(
      100,
      "Local AI is ready",
      `${backend} runs away from the page so controls stay responsive.`,
    );
    elements.runButton.disabled = !state.source;
    return { backend };
  } catch (error) {
    console.error(error);
    setRuntime("Local AI could not start", "error");
    setProgress(
      0,
      "Could not start local AI",
      "Reload the page or try a current Chrome, Edge, Firefox, or Safari browser.",
    );
    throw error;
  }
}

async function runInference(pixels, width, height) {
  const result = await sendWorkerRequest(
    "run",
    { pixels: pixels.buffer, width, height },
    [pixels.buffer],
    180_000,
  );
  return new Float32Array(result.output);
}

function startModelWarmup() {
  if (!state.modelPromise) state.modelPromise = initializeModel();
  return state.modelPromise;
}

async function decodeImage(blob) {
  if ("createImageBitmap" in window) {
    try {
      return await createImageBitmap(blob, { imageOrientation: "from-image" });
    } catch {
      return createImageBitmap(blob);
    }
  }

  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function sourcePixelLimit() {
  const memory = Number(navigator.deviceMemory) || 8;
  return memory <= 4 ? 4_000_000 : 8_000_000;
}

async function selectImage(blob, name = "image.png") {
  if (state.running) {
    showToast("Cancel the current upscale before choosing another image.");
    return;
  }
  if (!blob || !blob.type.startsWith("image/")) {
    showToast("Choose a PNG, JPEG, WebP, or another browser-supported image.");
    return;
  }

  try {
    setProgress(100, "Reading image locally", "Nothing is being uploaded.");
    const decoded = await decodeImage(blob);
    const width = decoded.width;
    const height = decoded.height;
    const pixels = width * height;
    const maxPixels = sourcePixelLimit();

    if (
      width * 2 > MAX_CANVAS_EDGE ||
      height * 2 > MAX_CANVAS_EDGE ||
      pixels > maxPixels
    ) {
      if (typeof decoded.close === "function") decoded.close();
      const megapixels = (maxPixels / 1_000_000).toFixed(0);
      throw new Error(
        `This device can safely process up to ${megapixels} megapixels ` +
        `and ${MAX_CANVAS_EDGE / 2}px per side at 2×.`,
      );
    }

    if (state.source && typeof state.source.close === "function") {
      state.source.close();
    }
    if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);

    state.source = decoded;
    state.sourceName = safeBaseName(name);
    state.sourceUrl = URL.createObjectURL(blob);
    state.finished = false;
    state.cancelRequested = false;

    elements.sourcePreview.src = state.sourceUrl;
    elements.sourcePreview.alt = `Selected image: ${name}`;
    elements.sourcePreview.hidden = false;
    elements.dropCopy.hidden = true;
    elements.replaceButton.hidden = false;
    elements.fileSummary.hidden = false;
    elements.fileName.textContent = name;
    elements.fileDetails.textContent =
      `${width} × ${height} · ${formatBytes(blob.size)}`;
    elements.outputDimensions.textContent =
      `${width} × ${height} → ${width * 2} × ${height * 2}`;

    drawBicubicPreview(decoded);
    prepareModelCanvas(width * 2, height * 2);
    elements.downloadBicubic.disabled = false;
    elements.downloadModel.disabled = true;
    elements.runButton.disabled = !state.modelReady;
    setProgress(
      state.modelReady ? 100 : 82,
      "Bicubic preview is ready",
      state.modelReady
        ? `Select “Make it 2× sharper” for the ${state.backend} result.`
        : "Local AI is still warming up and will be ready shortly.",
    );
    startModelWarmup().catch(() => {});
  } catch (error) {
    showToast(error.message || "This image could not be opened.");
  }
}

function drawBicubicPreview(source) {
  const outputWidth = source.width * 2;
  const outputHeight = source.height * 2;
  const scale = Math.min(1, PREVIEW_EDGE / Math.max(outputWidth, outputHeight));
  const canvas = elements.bicubicCanvas;
  canvas.width = Math.max(1, Math.round(outputWidth * scale));
  canvas.height = Math.max(1, Math.round(outputHeight * scale));
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  canvas.hidden = false;
  elements.bicubicEmpty.hidden = true;
}

function prepareModelCanvas(width, height) {
  const canvas = elements.modelCanvas;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  canvas.hidden = true;
  elements.modelEmpty.hidden = false;
  elements.tileProgress.hidden = true;
  elements.tileProgress.firstElementChild.style.width = "0%";
}

function createWorkCanvas(width, height) {
  if ("OffscreenCanvas" in window) return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function makeInputPixels(bounds) {
  const patchWidth = bounds.right - bounds.left;
  const patchHeight = bounds.bottom - bounds.top;
  const paddedWidth = Math.ceil(patchWidth / 8) * 8;
  const paddedHeight = Math.ceil(patchHeight / 8) * 8;
  const canvas = createWorkCanvas(paddedWidth, paddedHeight);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(
    state.source,
    bounds.left,
    bounds.top,
    patchWidth,
    patchHeight,
    0,
    0,
    patchWidth,
    patchHeight,
  );

  if (paddedWidth > patchWidth) {
    context.drawImage(
      canvas,
      patchWidth - 1,
      0,
      1,
      patchHeight,
      patchWidth,
      0,
      paddedWidth - patchWidth,
      patchHeight,
    );
  }
  if (paddedHeight > patchHeight) {
    context.drawImage(
      canvas,
      0,
      patchHeight - 1,
      paddedWidth,
      1,
      0,
      patchHeight,
      paddedWidth,
      paddedHeight - patchHeight,
    );
  }

  const rgba = context.getImageData(0, 0, paddedWidth, paddedHeight).data;
  const planeSize = paddedWidth * paddedHeight;
  const planar = new Float32Array(planeSize * 3);
  for (let index = 0, pixel = 0; index < planeSize; index += 1, pixel += 4) {
    planar[index] = rgba[pixel] / 255;
    planar[planeSize + index] = rgba[pixel + 1] / 255;
    planar[planeSize * 2 + index] = rgba[pixel + 2] / 255;
  }

  return {
    pixels: planar,
    paddedWidth,
    paddedHeight,
  };
}

function paintOutputTile(values, tile, paddedWidth, paddedHeight) {
  const outputWidth = paddedWidth * 2;
  const outputHeight = paddedHeight * 2;
  const outputPlaneSize = outputWidth * outputHeight;
  const cropX = (tile.coreLeft - tile.sourceLeft) * 2;
  const cropY = (tile.coreTop - tile.sourceTop) * 2;
  const coreWidth = (tile.coreRight - tile.coreLeft) * 2;
  const coreHeight = (tile.coreBottom - tile.coreTop) * 2;
  const pixels = new Uint8ClampedArray(coreWidth * coreHeight * 4);
  for (let y = 0; y < coreHeight; y += 1) {
    const sourceRow = (cropY + y) * outputWidth + cropX;
    const targetRow = y * coreWidth;
    for (let x = 0; x < coreWidth; x += 1) {
      const sourceIndex = sourceRow + x;
      const targetIndex = (targetRow + x) * 4;
      pixels[targetIndex] =
        Math.max(0, Math.min(255, Math.round(values[sourceIndex] * 255)));
      pixels[targetIndex + 1] =
        Math.max(
          0,
          Math.min(255, Math.round(values[outputPlaneSize + sourceIndex] * 255)),
        );
      pixels[targetIndex + 2] =
        Math.max(
          0,
          Math.min(
            255,
            Math.round(values[outputPlaneSize * 2 + sourceIndex] * 255),
          ),
        );
      pixels[targetIndex + 3] = 255;
    }
  }

  const imageData = new ImageData(pixels, coreWidth, coreHeight);
  const context = elements.modelCanvas.getContext("2d");
  context.putImageData(imageData, tile.coreLeft * 2, tile.coreTop * 2);
}

function buildTiles(width, height) {
  const tiles = [];
  for (let top = 0; top < height; top += TILE_SIZE) {
    const coreBottom = Math.min(top + TILE_SIZE, height);
    for (let left = 0; left < width; left += TILE_SIZE) {
      const coreRight = Math.min(left + TILE_SIZE, width);
      tiles.push({
        coreLeft: left,
        coreTop: top,
        coreRight,
        coreBottom,
        sourceLeft: Math.max(0, left - TILE_OVERLAP),
        sourceTop: Math.max(0, top - TILE_OVERLAP),
        sourceRight: Math.min(width, coreRight + TILE_OVERLAP),
        sourceBottom: Math.min(height, coreBottom + TILE_OVERLAP),
      });
    }
  }
  return tiles;
}

async function runUpscale() {
  if (!state.source || state.running) return;
  state.running = true;
  state.cancelRequested = false;
  state.finished = false;
  elements.runButton.disabled = true;
  elements.cancelButton.hidden = false;
  elements.downloadModel.disabled = true;
  elements.modelCanvas.hidden = false;
  elements.modelEmpty.hidden = true;
  elements.tileProgress.hidden = false;
  prepareModelCanvas(state.source.width * 2, state.source.height * 2);
  elements.modelCanvas.hidden = false;
  elements.modelEmpty.hidden = true;
  elements.tileProgress.hidden = false;

  try {
    const { backend } = await startModelWarmup();
    const tiles = buildTiles(state.source.width, state.source.height);
    const started = performance.now();

    for (let index = 0; index < tiles.length; index += 1) {
      if (state.cancelRequested) {
        setProgress(
          (index / tiles.length) * 100,
          "Upscaling stopped",
          `${index} of ${tiles.length} tiles completed. Your image stayed local.`,
        );
        return;
      }

      const tile = tiles[index];
      const bounds = {
        left: tile.sourceLeft,
        top: tile.sourceTop,
        right: tile.sourceRight,
        bottom: tile.sourceBottom,
      };
      const { pixels, paddedWidth, paddedHeight } = makeInputPixels(bounds);
      const tileStarted = performance.now();
      setProgress(
        (index / tiles.length) * 100,
        `Sharpening tile ${index + 1} of ${tiles.length}`,
        `${backend} is working away from the page. Controls remain available.`,
      );
      const heartbeat = window.setInterval(() => {
        const tileSeconds = (performance.now() - tileStarted) / 1000;
        setProgress(
          (index / tiles.length) * 100,
          `Sharpening tile ${index + 1} of ${tiles.length}`,
          `${backend} · ${formatDuration(tileSeconds)} on this tile · controls remain available`,
        );
      }, 1000);
      let output;
      try {
        output = await runInference(pixels, paddedWidth, paddedHeight);
      } finally {
        window.clearInterval(heartbeat);
      }
      paintOutputTile(output, tile, paddedWidth, paddedHeight);

      const completed = index + 1;
      const elapsedSeconds = (performance.now() - started) / 1000;
      const remainingSeconds =
        (elapsedSeconds / completed) * (tiles.length - completed);
      const percent = (completed / tiles.length) * 100;
      setProgress(
        percent,
        `Sharpening tile ${completed} of ${tiles.length}`,
        `${backend} · ${formatDuration(elapsedSeconds)} elapsed` +
          (remainingSeconds > 0
            ? ` · about ${formatDuration(remainingSeconds)} left`
            : ""),
      );
      elements.tileProgress.firstElementChild.style.width = `${percent}%`;
      await nextFrame();
    }

    const elapsedSeconds = (performance.now() - started) / 1000;
    state.finished = true;
    elements.downloadModel.disabled = false;
    setProgress(
      100,
      "Clark 2× result is ready",
      `${state.backend} finished locally in ${formatDuration(elapsedSeconds)}.`,
    );
    showToast("Sharper full-resolution PNG ready to download.");
  } catch (error) {
    console.error(error);
    setProgress(
      0,
      "Upscaling stopped",
      "The browser ran out of resources or the local AI engine failed.",
    );
    showToast("Could not finish this image. Try a smaller source or reload.");
  } finally {
    state.running = false;
    elements.cancelButton.hidden = true;
    elements.cancelButton.disabled = false;
    elements.cancelButton.textContent = "Cancel after this tile";
    elements.runButton.disabled = !state.source || !state.modelReady;
  }
}

async function canvasToBlob(canvas) {
  if ("convertToBlob" in canvas) {
    return canvas.convertToBlob({ type: "image/png" });
  }
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("PNG encoding failed"))),
      "image/png",
    );
  });
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function downloadBicubic() {
  if (!state.source) return;
  elements.downloadBicubic.disabled = true;
  try {
    setProgress(100, "Encoding bicubic PNG", "Building the full-resolution file.");
    const canvas = createWorkCanvas(state.source.width * 2, state.source.height * 2);
    const context = canvas.getContext("2d");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(state.source, 0, 0, canvas.width, canvas.height);
    const blob = await canvasToBlob(canvas);
    downloadBlob(blob, `${state.sourceName}-bicubic-2x.png`);
    setProgress(100, "Bicubic PNG downloaded", "The file was created locally.");
  } catch {
    showToast("This browser could not encode the full-resolution PNG.");
  } finally {
    elements.downloadBicubic.disabled = false;
  }
}

async function downloadModel() {
  if (!state.finished) return;
  elements.downloadModel.disabled = true;
  try {
    setProgress(100, "Encoding Clark PNG", "The full-resolution file stays local.");
    const blob = await canvasToBlob(elements.modelCanvas);
    downloadBlob(blob, `${state.sourceName}-clark-ternary-2x.png`);
    setProgress(100, "Clark PNG downloaded", "No upload and no watermark.");
  } catch {
    showToast("This browser could not encode the full-resolution PNG.");
  } finally {
    elements.downloadModel.disabled = false;
  }
}

function openPicker(event) {
  if (event?.target === elements.replaceButton) return;
  elements.fileInput.click();
}

elements.dropZone.addEventListener("click", openPicker);
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openPicker(event);
  }
});
elements.replaceButton.addEventListener("click", (event) => {
  event.stopPropagation();
  elements.fileInput.click();
});
elements.fileInput.addEventListener("change", () => {
  const file = elements.fileInput.files?.[0];
  if (file) selectImage(file, file.name);
  elements.fileInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) selectImage(file, file.name);
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      const response = await fetch(button.dataset.example);
      if (!response.ok) throw new Error("Example could not load");
      const blob = await response.blob();
      const name = button.dataset.example.split("/").pop();
      await selectImage(blob, name);
    } catch {
      showToast("The example image could not be opened.");
    }
  });
});

elements.runButton.addEventListener("click", runUpscale);
elements.cancelButton.addEventListener("click", () => {
  state.cancelRequested = true;
  elements.cancelButton.disabled = true;
  elements.cancelButton.textContent = "Stopping…";
});
elements.downloadBicubic.addEventListener("click", downloadBicubic);
elements.downloadModel.addEventListener("click", downloadModel);
window.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !state.running && state.source) runUpscale();
  if (event.key === "Escape" && state.running) state.cancelRequested = true;
});

startModelWarmup().catch(() => {});
