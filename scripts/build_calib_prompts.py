"""Build a diverse text-prompt calibration pool from HF (prompts only, no images) for data-free
QAT distillation. Sources: PartiPrompts + DrawBench + Stable-Diffusion-Prompts + procedural synthetic.
Dedupes, balanced-samples into calib/val/hard splits, writes JSONL. Robust to source failures.
Usage: python build_calib_prompts.py --out pool.jsonl --n_calib 256 --n_val 64 --n_hard 32
"""
import argparse, json, random, re
from collections import defaultdict

BAD={"nude","nudity","porn","sex","nsfw","explicit","gore","corpse","beheading"}
def clean(x):
    if x is None: return None
    x=re.sub(r"\s+"," ",str(x).strip()).replace(chr(0),"")
    if len(x)<5 or len(x)>300: return None
    lx=x.lower()
    if any(w in lx for w in BAD): return None
    return x

def add(rows,p,src,cat="unknown"):
    if isinstance(p,list):
        for q in p: add(rows,q,src,cat)
        return
    p=clean(p)
    if p: rows.append({"prompt":p,"source":src,"category":cat})

def try_load(rows, fn, name):
    try:
        n0=len(rows); fn(rows); print(f"  {name}: +{len(rows)-n0}",flush=True)
    except Exception as e:
        print(f"  {name}: FAILED ({type(e).__name__}: {str(e)[:80]})",flush=True)

def load_parti(rows):
    from datasets import load_dataset
    for ex in load_dataset("nateraw/parti-prompts",split="train"):
        add(rows, ex.get("Prompt"), "parti", ex.get("Category","parti"))
def load_drawbench(rows):
    from datasets import load_dataset
    ds=load_dataset("shunk031/DrawBench",split="train")
    for ex in ds:
        p=ex.get("prompts") or ex.get("prompt") or ex.get("text")
        add(rows, p, "drawbench", ex.get("category","drawbench"))
def load_sd(rows,limit=20000):
    from datasets import load_dataset
    n=0
    for ex in load_dataset("Gustavosta/Stable-Diffusion-Prompts",split="train"):
        add(rows, ex.get("Prompt") or ex.get("text"), "sd_prompts","real_t2i"); n+=1
        if n>=limit: break

def synthetic(n=2000,seed=0):
    rng=random.Random(seed)
    subj=["red panda astronaut","ceramic dragon","glass lighthouse","old robot","striped hot-air balloon",
        "half-empty wine bottle","harp without strings","small tabby cat","wooden chair","bowl of noodles",
        "yellow wall with text","vintage sports car","honeybee on a flower","snowy mountain village"]
    rel=["next to","above","below","inside","holding","chasing","reflected in","partly hidden behind"]
    sty=["photorealistic","oil painting","watercolor","cinematic lighting","macro photo","flat illustration","studio portrait"]
    plc=["in a kitchen","on a beach at sunset","in a snowy forest","on a city street","on a wooden table","under a cloudy sky"]
    txt=['with the words "KA-BOOM"','with a sign that says "OPEN"','with no text','with three visible objects','without strings']
    out=[]
    for _ in range(n):
        s1=rng.choice(subj)
        if rng.random()<0.5:
            p=f"{rng.choice(sty)} image of a {s1} {rng.choice(rel)} a {rng.choice(subj)}, {rng.choice(plc)}, {rng.choice(txt)}"
        else:
            p=f"{rng.choice(sty)} image of a {s1}, {rng.choice(plc)}, {rng.choice(txt)}"
        out.append({"prompt":p,"source":"synthetic","category":"synthetic"})
    return out

def dedupe(rows):
    seen=set(); out=[]
    for r in rows:
        k=re.sub(r"[^a-z0-9]+"," ",r["prompt"].lower()).strip()
        if k in seen: continue
        seen.add(k); out.append(r)
    return out

def balanced_sample(rows,n,seed=0):
    rng=random.Random(seed); buckets=defaultdict(list)
    for r in rows: buckets[(r["source"],r["category"])].append(r)
    for b in buckets.values(): rng.shuffle(b)
    sel=[]; keys=list(buckets.keys()); rng.shuffle(keys)
    while len(sel)<n and keys:
        nk=[]
        for k in keys:
            if buckets[k] and len(sel)<n: sel.append(buckets[k].pop()); nk.append(k)
        keys=nk; rng.shuffle(keys)
    rng.shuffle(sel); return sel[:n]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="sana_calib.jsonl")
    ap.add_argument("--n_calib",type=int,default=256); ap.add_argument("--n_val",type=int,default=64)
    ap.add_argument("--n_hard",type=int,default=32); ap.add_argument("--seed",type=int,default=123)
    a=ap.parse_args(); random.seed(a.seed); rows=[]
    print("loading sources:",flush=True)
    try_load(rows,load_parti,"parti"); try_load(rows,load_drawbench,"drawbench"); try_load(rows,load_sd,"sd_prompts")
    rows.extend(synthetic(2000,a.seed)); print(f"  synthetic: +2000",flush=True)
    rows=dedupe(rows); print(f"deduped pool: {len(rows)}",flush=True)
    total=a.n_calib+a.n_val+a.n_hard; sample=balanced_sample(rows,total,a.seed)
    for i,r in enumerate(sample):
        r["split"]="calib" if i<a.n_calib else ("val" if i<a.n_calib+a.n_val else "hard")
    with open(a.out,"w") as f:
        for r in sample: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    c=defaultdict(int)
    for r in sample: c[(r["split"],r["source"])]+=1
    print(f"wrote {len(sample)} -> {a.out}",flush=True)
    for k,v in sorted(c.items()): print(" ",k,v,flush=True)

if __name__=="__main__": main()
