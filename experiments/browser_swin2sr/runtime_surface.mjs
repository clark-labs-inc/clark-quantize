export function detectRuntimeSurface(navigatorLike) {
  const userAgent = String(navigatorLike?.userAgent ?? "");
  const platform = String(navigatorLike?.platform ?? "");
  const touchPoints = Number(navigatorLike?.maxTouchPoints ?? 0);
  const isIpadDesktopUa = /Macintosh/i.test(userAgent) && touchPoints > 1;
  const isIos = /iPhone|iPad|iPod/i.test(userAgent) || isIpadDesktopUa;
  const isAndroid = /Android/i.test(userAgent);

  if (isIos) {
    if (/CriOS/i.test(userAgent)) return "ios-chrome";
    if (/FxiOS/i.test(userAgent)) return "ios-firefox";
    if (/EdgiOS/i.test(userAgent)) return "ios-edge";
    return "ios-safari";
  }
  if (isAndroid) {
    if (/Firefox/i.test(userAgent)) return "android-firefox";
    if (/EdgA/i.test(userAgent)) return "android-edge";
    if (/Chrome|Chromium/i.test(userAgent)) return "android-chrome";
    return "android-browser";
  }
  if (/Firefox/i.test(userAgent)) return "firefox";
  if (/Edg\//i.test(userAgent)) return "edge";
  if (/Chrome|Chromium/i.test(userAgent)) return "chromium";
  if (/Safari/i.test(userAgent) && /Apple/i.test(navigatorLike?.vendor ?? "")) {
    return "safari";
  }
  if (/Mac/i.test(platform) && /Safari/i.test(userAgent)) return "safari";
  return "unknown";
}


export function runtimeSurface(navigatorLike, backend) {
  if (!["webgpu", "wasm"].includes(backend)) {
    throw new Error(`unsupported runtime backend: ${backend}`);
  }
  return `${detectRuntimeSurface(navigatorLike)}-${backend}`;
}


export async function decompressModelArtifact(
  bytes,
  compression = "none",
  DecompressionStreamClass = globalThis.DecompressionStream,
) {
  const payload = bytes instanceof ArrayBuffer
    ? bytes
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  if (compression === "none") return payload;
  if (compression !== "gzip") {
    throw new Error(`unsupported model compression: ${compression}`);
  }
  if (typeof DecompressionStreamClass !== "function") {
    throw new Error("gzip model decompression is unavailable");
  }
  const stream = new Blob([payload])
    .stream()
    .pipeThrough(new DecompressionStreamClass("gzip"));
  return new Response(stream).arrayBuffer();
}
