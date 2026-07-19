"""Build the six per-arm uint32 token memmaps for signature-steered ternary KD.

Streams open datasets through the model tokenizer into {out}/{arm}.bin, one
document at a time with EOS separators. The first PROBE_TOKENS of each arm are
reserved by the trainer as that arm's frozen probe slice.

Sources are a swap-friendly table below; gated sets need `huggingface-cli login`.

Infra pilot without real data:
  python scripts/prepare_kd_arms.py --out /data/kd_arms --synthetic 2e6 --model Qwen/Qwen3.5-4B
writes random-token arms — meaningless for quality, perfect for validating the
full training loop, VRAM, and tok/s before the real corpus lands.

Real run (defaults, ~400M tokens/arm ≈ 2.4B total; the 5B-token phase-1 budget
then makes ~2 epochs, which is fine for KD):
  python scripts/prepare_kd_arms.py --out /data/kd_arms --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# arm -> list of (dataset, config, split, kind, field); kind: text | messages | glaive
ARM_SOURCES: dict[str, list[tuple[str, str | None, str, str, str]]] = {
    "knowledge": [("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "text")],
    "math": [("open-web-math/open-web-math", None, "train", "text", "text")],
    "code": [
        ("OpenCoder-LLM/opc-fineweb-code-corpus", None, "train", "text", "text"),
        ("HuggingFaceTB/smollm-corpus", "python-edu", "train", "text", "text"),
    ],
    "instruct": [("HuggingFaceTB/smoltalk", "all", "train", "messages", "messages")],
    "tools": [
        ("glaiveai/glaive-function-calling-v2", None, "train", "glaive", "chat"),
        ("Team-ACE/ToolACE", None, "train", "sharegpt", "conversations"),
        ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "sharegpt", "conversations"),
    ],
    "multilingual": [
        ("wikimedia/wikipedia", "20231101.de", "train", "text", "text"),
        ("wikimedia/wikipedia", "20231101.fr", "train", "text", "text"),
        ("wikimedia/wikipedia", "20231101.zh", "train", "text", "text"),
        ("wikimedia/wikipedia", "20231101.ru", "train", "text", "text"),
        ("wikimedia/wikipedia", "20231101.ja", "train", "text", "text"),
    ],
}


ROUND_ROBIN_ARMS = {"multilingual"}


def doc_text(row: dict, kind: str, field: str, tokenizer) -> str | None:
    if kind == "text":
        return row.get(field)
    if kind == "messages":
        msgs = row.get(field)
        if not msgs:
            return None
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            return "\n".join(m.get("content", "") for m in msgs)
    if kind == "glaive":
        return f"{row.get('system', '')}\n{row.get(field, '')}"
    if kind == "sharegpt":
        conv = row.get(field)
        if not conv:
            return None
        if isinstance(conv, str):
            return conv
        return "\n".join(f"{m.get('from', m.get('role', ''))}: {m.get('value', m.get('content', ''))}"
                         for m in conv)
    return None


def write_arm(arm: str, out: Path, tokenizer, target_tokens: int) -> int:
    from datasets import load_dataset

    eos = tokenizer.eos_token_id
    path = out / f"{arm}.bin"
    written = 0
    with open(path, "wb") as f:
        for dataset, config, split, kind, field in ARM_SOURCES[arm]:
            if written >= target_tokens:
                break
            # greedy fill by default: each source takes whatever the previous
            # ones left, so an exhausted small source (e.g. glaive at ~56M
            # tokens) no longer starves the arm. Language-balance arms keep an
            # even split instead.
            if arm in ROUND_ROBIN_ARMS:
                share = min(target_tokens // len(ARM_SOURCES[arm]), target_tokens - written)
            else:
                share = target_tokens - written
            got = 0
            try:
                ds = load_dataset(dataset, config, split=split, streaming=True)
            except Exception as e:
                print(f"[{arm}] SKIP {dataset}/{config}: {e}", file=sys.stderr)
                continue
            buf: list[int] = []
            for row in ds:
                text = doc_text(row, kind, field, tokenizer)
                if not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                buf.append(eos)
                if len(buf) >= 1_000_000:
                    f.write(np.asarray(buf, dtype=np.uint32).tobytes())
                    written += len(buf)
                    got += len(buf)
                    buf = []
                    print(f"[{arm}] {written/1e6:.0f}M tokens", flush=True)
                    if got >= share or written >= target_tokens:
                        break
            if buf and written < target_tokens:
                f.write(np.asarray(buf, dtype=np.uint32).tobytes())
                written += len(buf)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="tokenizer source")
    ap.add_argument("--tokens-per-arm", type=float, default=400e6)
    ap.add_argument("--arms", nargs="*", default=list(ARM_SOURCES))
    ap.add_argument("--synthetic", type=float, default=0,
                    help="write N random tokens per arm instead of real data (loop/VRAM pilot)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.synthetic:
        n = int(args.synthetic)
        vocab = int(tokenizer.vocab_size)
        rng = np.random.default_rng(0)
        for arm in args.arms:
            toks = rng.integers(0, vocab, size=n, dtype=np.uint32)
            toks.tofile(args.out / f"{arm}.bin")
            print(f"[{arm}] synthetic {n/1e6:.1f}M tokens (vocab {vocab})")
        return
    for arm in args.arms:
        n = write_arm(arm, args.out, tokenizer, int(args.tokens_per_arm))
        print(f"[{arm}] DONE: {n/1e6:.0f}M tokens -> {args.out / (arm + '.bin')}")


if __name__ == "__main__":
    main()
