#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    result[argv[index].replace(/^--/, "")] = argv[index + 1];
  }
  if (!result.fixture || !result.ort) {
    throw new Error("usage: --fixture FILE --ort ORT_NODE_ESM");
  }
  return result;
}

function maximumError(actual, expected) {
  let maximum = 0;
  for (let index = 0; index < expected.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
  }
  return maximum;
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
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});
const sessionMs = performance.now() - sessionStarted;
const input = new ort.Tensor(
  "float32",
  new Float32Array(fixture.input),
  fixture.input_shape,
);
const started = performance.now();
const outputs = await session.run({ input });
const inferenceMs = performance.now() - started;
const values = outputs.output.data;
const result = {
  backend: "onnxruntime-web-wasm",
  sessionMs,
  inferenceMs,
  maxAbsoluteError: maximumError(values, fixture.expected),
  modelBytes: modelBytes.byteLength,
  outputShape: outputs.output.dims,
};
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
