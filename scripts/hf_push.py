"""Inspect or push a Clark Air artifact to HuggingFace.
Token resolution (first hit wins): HF_TOKEN env -> HUGGINGFACE_TOKEN/HF_API_TOKEN env -> /workspace/.env -> ~/.env -> ./.env
ACTION=inspect : list existing repo files (to find stale/proprietary content to remove)
ACTION=upload  : upload TRANSFORMER dir + README to REPO (keeps repo private; never flips visibility)
ENV: REPO=clark-labs/clark-air-sana-1.6b-1.58bit ARTIFACT=out/clark-air-sana-1.6b-unpacked ACTION=inspect
"""
import os

def _load_hf_token():
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        tok = os.environ.get(var)
        if tok and tok.strip():
            return tok.strip()
    for env_path in ("/workspace/.env", os.path.expanduser("~/.env"), ".env"):
        try:
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(("HF_TOKEN=", "HUGGINGFACE_TOKEN=", "HF_API_TOKEN=")):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    try:  # last resort: token cached by `huggingface-cli login`
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass
    raise SystemExit("[fatal] no HF token: export HF_TOKEN=..., add HF_TOKEN=... to /workspace/.env or ~/.env, or run `huggingface-cli login`")

os.environ["HF_TOKEN"] = _load_hf_token()
from huggingface_hub import HfApi, whoami
REPO=os.environ.get("REPO","clark-labs/clark-compress-sana-1.6b")
ART=os.environ.get("ARTIFACT","out/clark-air-sana-ternary-unpacked")
ACTION=os.environ.get("ACTION","inspect")
CLEAN=os.environ.get("CLEAN","0")=="1"  # delete stale old-draft files (only the original repo needs this)
api=HfApi(token=os.environ["HF_TOKEN"])
print("[auth]", whoami(token=os.environ["HF_TOKEN"]).get("name"), flush=True)
if ACTION=="upload":
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True); print(f"[repo] ensured {REPO} (private)", flush=True)
try:
    info=api.repo_info(REPO, repo_type="model")
    print(f"[repo] {REPO} | private={info.private}", flush=True)
    files=api.list_repo_files(REPO, repo_type="model")
    print(f"[existing files] {len(files)}:", flush=True)
    for f in files: print("   ", f, flush=True)
except Exception as e:
    print(f"[repo] {REPO} not accessible / new: {type(e).__name__}: {str(e)[:120]}", flush=True)

if ACTION=="upload":
    # 1) delete stale/proprietary old-draft files (original repo only)
    if CLEAN:
        for f in ["clark_compress_mixed.safetensors","load_clark_compress.py","manifest.json"]:
            try: api.delete_file(path_in_repo=f, repo_id=REPO, repo_type="model"); print(f"[delete] {f}", flush=True)
            except Exception as e: print(f"[delete] {f} skip ({type(e).__name__})", flush=True)
    # 2) upload new marketing README + bf16 transformer + coherent demo
    api.upload_file(path_or_fileobj=os.path.join(ART,"README.md"), path_in_repo="README.md", repo_id=REPO, repo_type="model")
    print("[upload] README.md", flush=True)
    demo=os.environ.get("DEMO","")
    if demo and os.path.exists(demo):
        api.upload_file(path_or_fileobj=demo, path_in_repo="demo.png", repo_id=REPO, repo_type="model"); print("[upload] demo.png", flush=True)
    api.upload_folder(folder_path=os.path.join(ART,"transformer"), path_in_repo="transformer", repo_id=REPO, repo_type="model")
    print("[upload] transformer/", flush=True)
    print("[done] uploaded — repo STILL PRIVATE (flip to public manually after review)", flush=True)
