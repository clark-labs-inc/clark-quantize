"""Build the reference-keyed, QKV/KV-pre-fused GemLite pack for ComfyUI.

Reads our diffusers-keyed `gemlite_pack.safetensors`, rewrites it into the NVlabs reference
layout (scripts/clark_air_gemlite/remap.py), and writes `clark_air_sana_gemlite_comfy.safetensors`.
If ComfyUI_ExtraModels' `SanaMS` is importable, asserts the pack covers every island the
skeleton needs and that every quantized trunk module has codes.

ENV: PACK=clark-air-sana-1.6b-gemlite-2bit/transformer/gemlite_pack.safetensors
     OUT=clark_air_sana_gemlite_comfy.safetensors  CHECK=1
Run with PYTHONPATH=<ComfyUI>:<ComfyUI>/custom_nodes/ComfyUI_ExtraModels for the skeleton check.
"""
import os, sys
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clark_air_gemlite"))
from remap import build_reference_pack, reference_quant_modules  # noqa: E402

PACK = os.environ.get("PACK", "clark-air-sana-1.6b-gemlite-2bit/transformer/gemlite_pack.safetensors")
OUT = os.environ.get("OUT", "clark_air_sana_gemlite_comfy.safetensors")
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "20"))
# islands the reference has but our diffusers weights don't: null-caption bank (unused at
# inference with a real negative prompt) and the sincos pos-embed buffer (use_pe=False).
SKELETON_OPTIONAL = {"y_embedder.y_embedding", "pos_embed"}


def main():
    sd = load_file(PACK)
    ref = build_reference_pack(sd, NUM_LAYERS)
    print(f"[remap] {len(ref)} tensors ({sum('::' not in k for k in ref)} islands + "
          f"{sum(k.endswith('::codes2b') for k in ref)} quant trunk)", flush=True)

    if os.environ.get("CHECK", "1") == "1":
        try:
            from Sana.conf import sana_conf
            from Sana.models.sana_multi_scale import SanaMS
        except Exception as e:
            print(f"[check] skipped (SanaMS not importable: {type(e).__name__}: {e})", flush=True)
        else:
            skel = set(SanaMS(**sana_conf["SanaMS_1600M_P1_D20"]["unet_config"]).state_dict().keys())
            quant = set(reference_quant_modules(NUM_LAYERS))
            qweight = {m + ".weight" for m in quant}
            qbias = {m + ".bias" for m in quant}
            pack_keys = set(ref)

            problems = []
            for m in quant:  # every quantized module: skeleton weight + our codes
                if m + ".weight" not in skel: problems.append(f"skeleton missing {m}.weight")
                if m + "::codes2b" not in pack_keys: problems.append(f"pack missing {m}::codes2b")
                if m + ".bias" in skel and m + ".bias" not in pack_keys:
                    problems.append(f"pack missing bias {m}.bias")
            for k in skel:  # every island the skeleton needs must be in the pack
                if k in qweight or k in qbias or k in SKELETON_OPTIONAL: continue
                if k not in pack_keys: problems.append(f"pack missing island {k}")
            for k in ref:  # every island we emit must be a real skeleton key
                if "::" in k or k.endswith(".bias"): continue
                if k not in skel: problems.append(f"pack has unknown island {k}")
            # shape sanity on islands
            ssd = SanaMS(**sana_conf["SanaMS_1600M_P1_D20"]["unet_config"]).state_dict()
            for k in ref:
                if "::" in k: continue
                if k in ssd and tuple(ref[k].shape) != tuple(ssd[k].shape):
                    problems.append(f"shape {k}: pack {tuple(ref[k].shape)} vs skel {tuple(ssd[k].shape)}")

            if problems:
                print("[check] FAILED:", flush=True)
                for p in problems[:40]: print("   -", p, flush=True)
                raise SystemExit(1)
            print(f"[check] OK — {len(quant)} quant modules, "
                  f"{len(skel) - len(qweight) - len(qbias) - len(SKELETON_OPTIONAL & skel)} islands covered", flush=True)

    save_file(ref, OUT)
    sz = os.path.getsize(OUT) / 1e6
    print(f"[done] {OUT}  ({sz:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
