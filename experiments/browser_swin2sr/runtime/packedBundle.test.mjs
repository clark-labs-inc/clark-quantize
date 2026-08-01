import assert from "node:assert/strict";
import test from "node:test";

import {
  decodePackedArtifactBundle,
  parsePackedArtifactBundle,
} from "./packedBundle.js";

const magic = [0x42, 0x44, 0x4b, 0x50, 0x56, 0x33, 0x00, 0x01];

function bundle(manifest, binary, graph = new Uint8Array(), layout = new Uint8Array()) {
  const version = graph.length ? 2 : 1;
  const header = version === 2 ? 24 : 16;
  const output = new Uint8Array(
    header + manifest.length + binary.length + graph.length + layout.length,
  );
  output.set([...magic.slice(0, 7), version]);
  const view = new DataView(output.buffer);
  view.setUint32(8, manifest.length, true);
  view.setUint32(12, binary.length, true);
  if (version === 2) {
    view.setUint32(16, graph.length, true);
    view.setUint32(20, layout.length, true);
  }
  output.set(manifest, header);
  output.set(binary, header + manifest.length);
  output.set(graph, header + manifest.length + binary.length);
  output.set(layout, header + manifest.length + binary.length + graph.length);
  return output;
}

test("packed artifact bundle keeps manifest and binary byte-exact", async () => {
  const manifest = new TextEncoder().encode('{"format":"clark-air-browser-v3"}');
  const binary = new Uint8Array([0, 1, 2, 127, 255]);
  const parsed = await decodePackedArtifactBundle(bundle(manifest, binary));
  assert.deepEqual(parsed.manifestBytes, manifest);
  assert.deepEqual(parsed.blobBytes, binary);
  assert.equal(parsed.graphBytes.length, 0);
  assert.equal(parsed.layoutBytes.length, 0);
});

test("WebGPU bundle keeps graph and external-data layout byte-exact", () => {
  const manifest = new Uint8Array([1]);
  const binary = new Uint8Array([2]);
  const graph = new Uint8Array([3, 4]);
  const layout = new Uint8Array([5, 6, 7]);
  const parsed = parsePackedArtifactBundle(bundle(manifest, binary, graph, layout));
  assert.deepEqual(parsed.graphBytes, graph);
  assert.deepEqual(parsed.layoutBytes, layout);
});

test("packed artifact bundle rejects truncation", () => {
  const manifest = new Uint8Array([1, 2]);
  const binary = new Uint8Array([3, 4]);
  const truncated = bundle(manifest, binary).slice(0, -1);
  assert.throws(() => parsePackedArtifactBundle(truncated), /truncated/);
});

test("packed artifact bundle decompresses deterministic gzip", async () => {
  const manifest = new Uint8Array([1, 2, 3]);
  const binary = new Uint8Array([4, 5, 6]);
  const source = bundle(manifest, binary);
  const compressed = await new Response(
    new Blob([source]).stream().pipeThrough(new CompressionStream("gzip")),
  ).arrayBuffer();
  const parsed = await decodePackedArtifactBundle(compressed);
  assert.deepEqual(parsed.manifestBytes, manifest);
  assert.deepEqual(parsed.blobBytes, binary);
});
