const MAGIC_PREFIX = new Uint8Array([0x42, 0x44, 0x4b, 0x50, 0x56, 0x33, 0x00]);

function hasMagic(bytes) {
  return MAGIC_PREFIX.every((value, index) => bytes[index] === value)
    && (bytes[7] === 1 || bytes[7] === 2);
}

export function parsePackedArtifactBundle(payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  if (bytes.byteLength < 16 || !hasMagic(bytes)) {
    throw new Error("BackdropKit packed artifact bundle is invalid");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = bytes[7];
  const headerBytes = version === 2 ? 24 : 16;
  if (bytes.byteLength < headerBytes) {
    throw new Error("BackdropKit packed artifact bundle is truncated");
  }
  const manifestLength = view.getUint32(8, true);
  const binaryLength = view.getUint32(12, true);
  const graphLength = version === 2 ? view.getUint32(16, true) : 0;
  const layoutLength = version === 2 ? view.getUint32(20, true) : 0;
  const manifestEnd = headerBytes + manifestLength;
  const binaryEnd = manifestEnd + binaryLength;
  const graphEnd = binaryEnd + graphLength;
  if (graphEnd + layoutLength !== bytes.byteLength) {
    throw new Error("BackdropKit packed artifact bundle is truncated");
  }
  return {
    manifestBytes: bytes.slice(headerBytes, manifestEnd),
    blobBytes: bytes.slice(manifestEnd, binaryEnd),
    graphBytes: bytes.slice(binaryEnd, graphEnd),
    layoutBytes: bytes.slice(graphEnd),
  };
}

async function decompressGzip(payload) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser cannot decompress the packed AI model");
  }
  const stream = new Blob([payload]).stream().pipeThrough(
    new DecompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function decodePackedArtifactBundle(payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const expanded = hasMagic(bytes) ? bytes : await decompressGzip(bytes);
  return parsePackedArtifactBundle(expanded);
}

export async function fetchPackedArtifactBundle(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Packed AI model download failed (${response.status})`);
  }
  return decodePackedArtifactBundle(await response.arrayBuffer());
}
