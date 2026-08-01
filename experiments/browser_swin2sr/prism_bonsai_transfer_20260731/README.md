# Prisma Bonsai precision-transfer study

This study extracts falsifiable lessons from the exact released PrismML Bonsai
Image ternary checkpoint and tests whether they transfer to BackdropKit's tiny
Swin2SR model.

Upstream evidence is pinned to:

- PrismML model revision `bf047aa80569f193d2c4960033af6ba2629988fa`;
- transformer checkpoint SHA-256
  `a3a7df8a90374fea24afce3b36f00b4c728d0254717143d61f912a7b3070e7ac`;
- [official model card](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-gemlite-2bit);
- [official whitepaper](https://github.com/PrismML-Eng/Bonsai-Image-Demo/blob/main/bonsai-image-4b-whitepaper.pdf).

It does not assume that a 4B diffusion transformer's exception map applies to
Swin2SR. The transferable hypotheses are narrower:

1. quantize matrix-heavy trunks while preserving a measured high-precision
   island;
2. account for scales, zero-points, biases, norms, metadata, and runtime packing
   in effective bits per parameter;
3. distinguish stored weight precision from BF16/FP16 activation and FP32
   accumulation precision;
4. test whether symmetric ternary zero-points can be derived rather than stored;
5. use tensor-role and measured sensitivity gates before lowering precision.

The upstream transformer checkpoint is cached outside the repository at:

`/Users/stan/.cache/clark-quantize/prism-bonsai-image-bf047aa/state_dict.pt`

Its required SHA-256 is
`a3a7df8a90374fea24afce3b36f00b4c728d0254717143d61f912a7b3070e7ac`.

Run the inventory:

```bash
python3 experiments/browser_swin2sr/prism_bonsai_transfer_20260731/analyze_checkpoint.py \
  --checkpoint /Users/stan/.cache/clark-quantize/prism-bonsai-image-bf047aa/state_dict.pt \
  --expected-sha256 a3a7df8a90374fea24afce3b36f00b4c728d0254717143d61f912a7b3070e7ac \
  --out experiments/browser_swin2sr/prism_bonsai_transfer_20260731/packed-weight-inventory.json
```

See `SCIENTIFIC_JOURNAL.md` for interpretations, `scientist-brief.md` for the
frozen context supplied to Clark Scientist, and `SCIENTIST_DEEPSEEK_0731.md`
for the bounded OpenRouter/DeepSeek proposal receipt and deterministic audit.
