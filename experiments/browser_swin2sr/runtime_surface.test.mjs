import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync } from "node:zlib";

import {
  decompressModelArtifact,
  detectRuntimeSurface,
  runtimeSurface,
} from "./runtime_surface.mjs";

const chromeMac = {
  userAgent: "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/140 Safari/537.36",
  vendor: "Google Inc.",
  platform: "MacIntel",
};
const safariMac = {
  userAgent: "Mozilla/5.0 (Macintosh) AppleWebKit/605.1 Version/26.0 Safari/605.1",
  vendor: "Apple Computer, Inc.",
  platform: "MacIntel",
};
const firefoxMac = {
  userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X) Gecko/20100101 Firefox/142.0",
  vendor: "",
  platform: "MacIntel",
};
const chromeAndroid = {
  userAgent: "Mozilla/5.0 (Linux; Android 16) AppleWebKit/537.36 Chrome/140 Mobile Safari/537.36",
  vendor: "Google Inc.",
  platform: "Linux armv8l",
};
const safariIos = {
  userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1 Version/26.0 Mobile/15E148 Safari/604.1",
  vendor: "Apple Computer, Inc.",
  platform: "iPhone",
  maxTouchPoints: 5,
};

test("detects desktop browser families from navigator evidence", () => {
  assert.equal(detectRuntimeSurface(chromeMac), "chromium");
  assert.equal(detectRuntimeSurface(safariMac), "safari");
  assert.equal(detectRuntimeSurface(firefoxMac), "firefox");
});

test("detects mobile platform and browser together", () => {
  assert.equal(detectRuntimeSurface(chromeAndroid), "android-chrome");
  assert.equal(detectRuntimeSurface(safariIos), "ios-safari");
});

test("surface labels cannot be supplied independently of the navigator", () => {
  assert.equal(runtimeSurface(chromeMac, "webgpu"), "chromium-webgpu");
  assert.throws(() => runtimeSurface(chromeMac, "cpu"), /unsupported/);
});

test("decompresses the deterministic gzip model package", async () => {
  const expected = new TextEncoder().encode("packed ternary model");
  const compressed = gzipSync(expected, { level: 9, mtime: 0 });
  const actual = await decompressModelArtifact(compressed, "gzip");

  assert.deepEqual(new Uint8Array(actual), expected);
});

test("fails closed when compressed model support is unavailable", async () => {
  await assert.rejects(
    decompressModelArtifact(new Uint8Array([1, 2, 3]), "gzip", null),
    /unavailable/,
  );
});
