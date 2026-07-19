"""Benchmark a ternary KD checkpoint — GSM8K (generate+EM) and MMLU (logprob).

Evaluates the DEPLOYED artifact: the bf16 absmean-dequant of the trained masters
(exactly what packs to Q2_0), reconstructed on CPU then moved to GPU (~8.5 GB for
4B). A GPU preflight aborts if free memory is insufficient, so this never OOMs a
training run sharing the card — it lives in the free slack and caps itself.

  # ternary checkpoint:
  python scripts/eval_ternary_ckpt.py --model Qwen/Qwen3.5-4B \
      --ckpt /workspace/agentworld_runs/<run>/ckpt_1500.pt \
      --tasks gsm8k,mmlu --limit 200 --out /workspace/eval_ckpt_1500.json

  # base teacher, same samples (retention baseline; run when GPU frees up):
  python scripts/eval_ternary_ckpt.py --model Qwen/Qwen3.5-4B --baseline \
      --tasks gsm8k,mmlu --limit 200 --out /workspace/eval_base.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.ternary_kd import fused_absmean_dequant

GSM8K_SHOTS = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. "
     "How many clips did she sell altogether in April and May?",
     "In April she sold 48. In May she sold 48/2 = 24. Altogether 48+24 = 72.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?",
     "Per minute she earns 12/60 = $0.2. For 50 minutes she earned 0.2*50 = $10.\n#### 10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
     "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
     "How much more money does Betty need?",
     "Betty has 100/2 = $50. Grandparents give 2*15 = $30. Now she has 50+15+30 = $95. "
     "She needs 100-95 = $5 more.\n#### 5"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "Each time he writes 3*2 = 6 pages. Twice a week is 6*2 = 12 pages. In a year 12*52 = 624 pages.\n#### 624"),
]


def preflight_gpu(min_free_gb: float, device: str) -> None:
    if device == "cpu":
        return
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 2**30, total / 2**30
    print(f"GPU free {free_gb:.1f} / {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        raise SystemExit(
            f"ABORT: only {free_gb:.1f} GB free, need >= {min_free_gb:.1f}. "
            "A training run is likely using the card. Re-run when it frees up, "
            "or on a second GPU. (Not touching the training PID.)")
    # hard-cap this process so it cannot eat into the training allocation
    frac = min((free_gb - 0.75) / total_gb, 0.99)
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"capped eval process to {frac*total_gb:.1f} GB")


def load_deployed_model(base: str, ckpt_path: Path | None, device: str, dtype=torch.bfloat16):
    """Plain bf16 model with ternary Linears replaced by their absmean-dequant.

    baseline (ckpt_path=None) returns the untouched base model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_plain():
        try:
            return AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                base, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = load_plain()  # CPU
    if ckpt_path is not None:
        print(f"loading checkpoint {ckpt_path} (CPU) ...")
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("model", sd)
        state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
        own = model.state_dict()
        assign, n_deq, n_direct = {}, 0, 0
        for k, v in state.items():
            if k.endswith("weight_master"):
                target = k[: -len("weight_master")] + "weight"
                if target in own:
                    assign[target] = fused_absmean_dequant(v.float(), out_dtype=dtype).reshape(own[target].shape)
                    n_deq += 1
            elif k in own and own[k].shape == v.shape:
                assign[k] = v.to(dtype)
                n_direct += 1
        step = sd.get("step", "?")
        res = model.load_state_dict(assign, strict=False)
        print(f"checkpoint step {step}: dequantized {n_deq} ternary tensors, "
              f"copied {n_direct} frozen tensors, applied {len(assign)}; "
              f"{len(res.missing_keys)} left at base (norms/tied head).")
        del sd, state, assign
    # text-only eval: drop the vision tower + MTP head so they don't occupy the
    # tight GPU slack (they carry no text-eval signal; ~1-1.5 GB saved on a VL 4B)
    core = getattr(model, "model", model)
    for attr in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attr, None) is not None:
            setattr(core, attr, None); print(f"dropped {attr} for text eval")
    if getattr(model, "mtp", None) is not None:
        model.mtp = None; print("dropped mtp head for text eval")
    model.config.use_cache = True
    model.eval().to(device)
    if device != "cpu":
        print(f"model on GPU: {torch.cuda.memory_allocated()/2**30:.1f} GB")
    return model, tok


@torch.no_grad()
def eval_gsm8k(model, tok, device, limit, shots, max_new):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    preamble = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in GSM8K_SHOTS[:shots])
    correct = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = preamble + f"Question: {row['question']}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        text = text.split("Question:")[0]
        nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
        pred = nums[-1].replace(",", "").replace("$", "").rstrip(".") if nums else None
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-4
        except ValueError:
            ok = pred == gold
        correct += ok
        if (i + 1) % 25 == 0:
            print(f"  gsm8k {i+1}/{len(ds)}  acc={correct/(i+1):.3f}  "
                  f"({(i+1)/(time.time()-t0):.2f} q/s)", flush=True)
    return {"task": "gsm8k", "n": len(ds), "shots": shots, "accuracy": correct / len(ds)}


@torch.no_grad()
def eval_mmlu(model, tok, device, limit, shots, dataset):
    from datasets import load_dataset
    test = load_dataset(dataset, "all", split="test")
    dev = load_dataset(dataset, "all", split="dev")
    by_subject: dict[str, list] = {}
    for r in dev:
        by_subject.setdefault(r["subject"], []).append(r)
    if limit and limit < len(test):
        step = max(1, len(test) // limit)  # even stride across subjects, not first-N
        test = test.select(list(range(0, len(test), step))[:limit])
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in letters]

    def fmt(r, with_answer):
        q = r["question"].strip()
        opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(r["choices"]))
        s = f"{q}\n{opts}\nAnswer:"
        return s + (f" {letters[r['answer']]}\n\n" if with_answer else "")

    correct, t0 = 0, time.time()
    for i, row in enumerate(test):
        subj = row["subject"]
        ex = by_subject.get(subj, [])[:shots]
        head = f"The following are multiple choice questions (with answers) about {subj.replace('_',' ')}.\n\n"
        prompt = head + "".join(fmt(e, True) for e in ex) + fmt(row, False)
        ids = tok(prompt, return_tensors="pt").to(device)
        logits = model(**ids).logits[0, -1]
        pred = int(torch.tensor([logits[t] for t in letter_ids]).argmax())
        correct += pred == row["answer"]
        if (i + 1) % 50 == 0:
            print(f"  mmlu {i+1}/{len(test)}  acc={correct/(i+1):.3f}  "
                  f"({(i+1)/(time.time()-t0):.1f} q/s)", flush=True)
    return {"task": "mmlu", "n": len(test), "shots": shots, "dataset": dataset,
            "accuracy": correct / len(test)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--baseline", action="store_true", help="eval base model (ignore --ckpt)")
    ap.add_argument("--tasks", default="gsm8k,mmlu")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gsm8k-shots", type=int, default=4)
    ap.add_argument("--mmlu-shots", type=int, default=5)
    ap.add_argument("--mmlu-dataset", default="cais/mmlu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-free-gb", type=float, default=10.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    preflight_gpu(args.min_free_gb, args.device)

    ckpt = None if args.baseline else args.ckpt
    if not args.baseline and ckpt is None:
        raise SystemExit("pass --ckpt <path> or --baseline")
    model, tok = load_deployed_model(args.model, ckpt, args.device)

    report = {"model": args.model, "ckpt": str(ckpt), "baseline": args.baseline, "results": []}
    for task in args.tasks.split(","):
        task = task.strip()
        print(f"\n=== {task} ===", flush=True)
        if task == "gsm8k":
            r = eval_gsm8k(model, tok, args.device, args.limit, args.gsm8k_shots, args.max_new_tokens)
        elif task == "mmlu":
            r = eval_mmlu(model, tok, args.device, args.limit, args.mmlu_shots, args.mmlu_dataset)
        else:
            print(f"unknown task {task}, skipping"); continue
        print(f"  -> {task} accuracy = {r['accuracy']:.4f}  (n={r['n']})")
        report["results"].append(r)

    print("\n" + json.dumps(report["results"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
