"""Self-contained test: queue an all-ClarkAir-nodes workflow (no ExtraModels), fetch the image."""
import json, time, os, urllib.request, urllib.error

API = "http://127.0.0.1:8188"
GRAPH = {
    "1": {"class_type": "ClarkAirGemmaLoader", "inputs": {"model_name": "unsloth/gemma-2-2b-it-bnb-4bit", "device": "cuda", "dtype": "BF16"}},
    "2": {"class_type": "ClarkAirGemmaEncode", "inputs": {"text": "a corgi astronaut floating in space, photorealistic, 8k", "gemma": ["1", 0]}},
    "3": {"class_type": "ClarkAirGemmaEncode", "inputs": {"text": "", "gemma": ["1", 0]}},
    "4": {"class_type": "ClarkAirSanaLoader", "inputs": {"pack_name": "clark_air_sana_gemlite_comfy.safetensors", "model_variant": "SanaMS_1600M_P1_D20", "dtype": "bfloat16"}},
    "5": {"class_type": "ClarkAirSanaEmptyLatent", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["5", 0], "seed": 9000, "steps": 20, "cfg": 4.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    "7": {"class_type": "ClarkAirVAELoader", "inputs": {"vae_name": "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"}},
    "8": {"class_type": "ClarkAirVAEDecode", "inputs": {"samples": ["6", 0], "vae": ["7", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ClarkAirSC"}},
}
try:
    req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": GRAPH, "client_id": "sc"}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=30))
except urllib.error.HTTPError as e:
    print("[/prompt REJECTED]", e.code, e.read().decode()[:900]); raise SystemExit(1)
pid = r["prompt_id"]; print("[queued]", pid, flush=True)
img = None
for _ in range(260):
    h = json.load(urllib.request.urlopen(API + f"/history/{pid}", timeout=30))
    if pid in h:
        e = h[pid]
        for o in e.get("outputs", {}).values():
            if o.get("images"): img = o["images"][0]; break
        if img: break
        st = e.get("status", {})
        if st.get("status_str") == "error": print("[ERROR]", json.dumps(st)[:1200]); raise SystemExit(1)
    time.sleep(2)
print("[image]", img, flush=True)
if img:
    p = os.path.join("/workspace/ComfyUI/output", img.get("subfolder", ""), img["filename"])
    print("[exists]", os.path.exists(p), p)
else:
    print("[timeout]"); raise SystemExit(1)
