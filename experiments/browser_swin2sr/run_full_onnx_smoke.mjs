#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const result = { backend: "wasm", runs: 3 };
  for (let index = 0; index < argv.length; index += 2) {
    result[argv[index].replace(/^--/, "")] = argv[index + 1];
  }
  result.runs = Number(result.runs);
  if (!result.fixture || !result.ort || !["wasm", "webgpu"].includes(result.backend)) {
    throw new Error(
      "usage: --fixture FILE --ort ORT_ESM [--backend wasm|webgpu] [--runs N]",
    );
  }
  return result;
}

function errors(actual, expected) {
  let maximum = 0;
  let total = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maximum = Math.max(maximum, error);
    total += error;
  }
  return { maximum, mean: total / expected.length };
}

const args = parseArgs(process.argv.slice(2));
const fixture = JSON.parse(await fs.readFile(args.fixture, "utf8"));
const ort = await import(pathToFileURL(path.resolve(args.ort)).href);
const moduleRoot = path.dirname(path.resolve(args.ort));
ort.env.wasm.wasmPaths = `${moduleRoot}/`;
ort.env.wasm.numThreads = 1;
const modelBytes = await fs.readFile(fixture.model);
const sessionStarted = performance.now();
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: [args.backend],
  graphOptimizationLevel: "all",
});
const sessionMs = performance.now() - sessionStarted;
const input = new ort.Tensor(
  "float32",
  new Float32Array(fixture.input),
  fixture.input_shape,
);
const timings = [];
let output;
for (let iteration = 0; iteration < args.runs; iteration += 1) {
  const started = performance.now();
  output = await session.run({ [fixture.input_name]: input });
  timings.push(performance.now() - started);
}
const outputTensor = output[fixture.output_name];
const measured = errors(outputTensor.data, fixture.expected);
process.stdout.write(`${JSON.stringify({
  backend: args.backend,
  modelBytes: modelBytes.byteLength,
  sessionMs,
  inferenceMs: timings,
  maxAbsoluteError: measured.maximum,
  meanAbsoluteError: measured.mean,
  outputShape: outputTensor.dims,
}, null, 2)}\n`);
