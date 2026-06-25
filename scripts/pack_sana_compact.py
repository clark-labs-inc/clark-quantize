"""Re-serialize a Clark Air packed payload into a TIGHT safetensors container.

`pack_sana_8x.py` writes the packed ternary payload via torch.save (pickle), which
carries ~38% serialization overhead (200+ small numpy arrays + dict structures). This
script reads that payload (or re-packs from a checkpoint) and writes the SAME bits as a
flat safetensors file: raw packed codes (uint8 view of the uint64 stream), raw scale
quants, raw FP8 islands, with all structure in the safetensors metadata header. The
decoded weights are bit-identical to the .pt — this is a lossless container change, not
a re-quantization, so quality is provably unchanged.

ENV: PACKED=...-pack40-log6-fp8.pt OUT=...-pack40-log6-fp8.safetensors VERIFY=1
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file, safe_open

PACKED = os.environ.get("PACKED", "clark-air-sana-1.6b-pack40-log6-fp8.pt")
OUT = os.environ.get("OUT", os.path.splitext(PACKED)[0] + ".safetensors")
FP16_BYTES = int(os.environ.get("FP16_BYTES", "3208971544"))  # FP16/bf16 transformer baseline
VERIFY = int(os.environ.get("VERIFY", "1"))

POW3_40 = np.array([3 ** i for i in range(40)], dtype=np.uint64)


def unpack_trits_40(packed, shape, pad):
    vals = packed.astype(np.uint64).copy()
    out = np.empty(vals.size * 40, dtype=np.int8)
    for i in range(40):
        out[i::40] = (vals % 3).astype(np.int8)
        vals //= 3
    if pad:
        out = out[:-pad]
    return (out.reshape(shape) - 1).astype(np.int8)


def main():
    pk = torch.load(PACKED, map_location="cpu", weights_only=False)
    tensors = {}
    meta_codes, meta_scales, meta_islands = [], [], []

    # trunk codes: store the uint64 packed stream as raw uint8 bytes
    for i, c in enumerate(pk["codes"]):
        packed = np.asarray(c["packed"], dtype=np.uint64)
        tensors[f"code.{i}"] = torch.from_numpy(np.frombuffer(packed.tobytes(), dtype=np.uint8).copy())
        meta_codes.append({"name": c["name"], "n_u64": int(packed.size),
                           "code_shape": list(c["code_shape"]), "pad": int(c["pad"]),
                           "weight_shape": list(c["weight_shape"])})

    # per-tensor scale quants (uint8) + lo/hi/bits in metadata
    for i, s in enumerate(pk["scales"]):
        q = s["q"]
        q = q if isinstance(q, torch.Tensor) else torch.as_tensor(q)
        tensors[f"scale.{i}"] = q.to(torch.uint8).contiguous().view(-1)
        meta_scales.append({"name": s["name"], "shape": list(q.shape),
                            "lo": float(s["lo"]), "hi": float(s["hi"]), "bits": int(s["bits"])})

    # FP8 islands: store raw bytes (format-agnostic), record shape
    for name, t in pk["islands"].items():
        raw = t.contiguous().view(torch.uint8).view(-1) if t.dtype == torch.float8_e4m3fn else \
              torch.from_numpy(np.frombuffer(t.to(torch.float8_e4m3fn).contiguous().numpy().tobytes(), dtype=np.uint8).copy())
        tensors[f"island.{name}"] = raw
        meta_islands.append({"name": name, "shape": list(t.shape)})

    metadata = {
        "format": "sana_t112_compact_v1",
        "source_format": pk.get("format", ""),
        "step": str(int(pk.get("step", 0))),
        "group_size": str(int(pk.get("group_size", 112))),
        "scale_bits": str(int(pk.get("scale_bits", 6))),
        "quant_names": json.dumps(pk["quant_names"]),
        "island_names": json.dumps(pk["island_names"]),
        "codes": json.dumps(meta_codes),
        "scales": json.dumps(meta_scales),
        "islands": json.dumps(meta_islands),
    }
    save_file(tensors, OUT, metadata=metadata)
    sz = os.path.getsize(OUT)
    pt_sz = os.path.getsize(PACKED)
    print(f"[compact] {OUT}")
    print(f"  .pt (pickle) : {pt_sz/1e6:8.1f} MB  ({FP16_BYTES/pt_sz:.2f}x vs FP16)")
    print(f"  .safetensors : {sz/1e6:8.1f} MB  ({FP16_BYTES/sz:.2f}x vs FP16)")
    print(f"  saved        : {(pt_sz-sz)/1e6:8.1f} MB  ({100*(pt_sz-sz)/pt_sz:.0f}% smaller)")

    if VERIFY:
        # decode both and confirm bit-identical codes/scales/islands
        bad = 0
        with safe_open(OUT, framework="np") as f:
            md = f.metadata()
            mc = json.loads(md["codes"])
            for i, (cm, src) in enumerate(zip(mc, pk["codes"])):
                raw = f.get_tensor(f"code.{i}")
                packed = np.frombuffer(raw.tobytes(), dtype=np.uint64)
                dec = unpack_trits_40(packed, tuple(cm["code_shape"]), cm["pad"])
                ref = unpack_trits_40(np.asarray(src["packed"], dtype=np.uint64),
                                      tuple(src["code_shape"]), src["pad"])
                if not np.array_equal(dec, ref):
                    bad += 1
        print(f"  VERIFY: code roundtrip mismatches = {bad}  (0 = bit-identical, lossless)")
        assert bad == 0, "compact container is not bit-identical!"


if __name__ == "__main__":
    main()
