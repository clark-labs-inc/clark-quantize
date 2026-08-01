# Sana Image-Generation 1.58-bit Research Journal

Status date: 2026-07-30

## Scope and record policy

This dossier consolidates Clark's Sana 1.6B low-bit discoveries without moving
or renaming the root reports, trainers, release metadata, datasets, or runtime
artifacts. The linked source files remain authoritative.

Sana is an archived image-generation research line and a useful billion-scale
DiT reference. It is not the active optimization target; BackdropKit's Swin2SR
work is active.

## Verified findings

### 1. MSE-optimal ternary is a substantially better PTQ initializer than absmean

On the repository's historical Sana 1.6B test, absmean group-128 produced
`9.08 dB` PSNR versus FP16 and visual noise, while MSE-optimal group-128 reached
`9.52 dB` and recognizable output. MSE-optimal group-32 reached `12.01 dB`.

Activation-aware GPTQ and shared rotation improved the historical group-128
result to `17.01 dB`. These measurements cover three prompts at 512px, 20
steps, and seed 0; they establish a mechanism and narrow benchmark result, not
broad quality parity.

Evidence:

- [repository research README](../../README.md)
- [technical paper source](../../paper/clark-quantize.tex)
- [optimal ternary implementation](../../clark_quantize/ternary.py)
- [activation-aware implementation](../../clark_quantize/gptq.py)

### 2. Hard-forward master-weight QAT produced a compact coherent Sana artifact

The step-31,000 release manifest records:

- base model `Efficient-Large-Model/Sana_1600M_512px`;
- 200 ternary trunk modules, group size 112;
- 1,555,456,000 ternary trunk parameters;
- 47.7% zero codes;
- 13 trained and 23 frozen continuous islands;
- packed safetensors size `373,654,344` bytes;
- FP16 transformer size `3,208,971,544` bytes;
- measured compression `8.59x`;
- measured payload `1.8457` bits per weight;
- no bad modules in the pack round trip.

The result is a mixed artifact with explicit higher-precision islands, not a
claim that every parameter is globally 1.58 bits. Its release comparison uses
four prompts at identical seed, 512px, and 20 steps.

Evidence:

- [step-31,000 release manifest](../../releases/clark-air-sana-1.6b-s31000.json)
- [current Clark Air README](../../README.md)
- [release sharing README](../../SHARE_README.md)

### 3. The released optimizer was AdamW8bit plus explicit QAT mechanisms

Repository history audited in the optimizer report establishes that the
released Sana trainer used `bitsandbytes.optim.AdamW8bit`, followed by a
scheduled proximal pull and periodic closed-form scale refresh. The release
tag contains `muon`, but that historical tag is not evidence that Muon trained
the release.

The supported scientific lesson is hard-forward QAT from continuous masters,
with derived support and carefully managed scales. Muon remains a candidate to
test under matched exposure.

Evidence: [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md)

### 4. Explicit support editing was inferior to derived support

CTD, PrismLock, Mirror-CTD, forced density, and direct mask-variable branches
either froze useful support or moved it destructively. The successful line
derived hard support from moving continuous masters rather than optimizing
support as a separate discrete object.

Evidence:

- [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md)
- [universal compressor synthesis](../../CLARK_AIR_UNIVERSAL_COMPRESSOR.md)

### 5. Pure binary was not the successful image route

Recorded binary/ternary grids found that ternary preserved hard composition
better, while binary collapsed on difficult harp and horse/person prompts. The
product route used ternary plus protected higher-precision islands rather than
global binary.

Evidence:

- [universal compressor synthesis](../../CLARK_AIR_UNIVERSAL_COMPRESSOR.md)
- [binary preview model card](../../release/MODEL_CARD_binary.md)
- [render preferences](../../data/sana_render_rank_preferences.json)

### 6. Tight packing is real but independent from quality

The release pack stores 40 trits per uint64, six-bit log scales, and FP8
islands in a safetensors container. The manifest reports a lossless
re-serialization relative to the reference pack and zero bad round-trip
modules. This proves the storage representation and round trip; it does not by
itself prove visual equivalence to the FP16 teacher.

Evidence:

- [step-31,000 release manifest](../../releases/clark-air-sana-1.6b-s31000.json)
- [Sana packer](../../scripts/pack_sana_8x.py)
- [release loader](../../release/load_clark_compress.py)

## Stable negative results

- Replay, x0, and perceptual objectives preserved broad structure but softened
  or averaged detail.
- Bellman/value shaping mainly learned trajectory proximity rather than
  visible object and relation correctness.
- PTD reached a stable scalar band around `0.244-0.249` without a visual
  champion.
- Stein/Jeffreys stayed stable without a visual breakthrough.
- Self-oracle and CVF promotion failed stability or inversion gates.
- A no-checkpoint A100 speed probe failed on its first training step because a
  full teacher-copy recipe nearly filled 80 GB; checkpointing or rebatched
  execution is required.
- Sana's GLU-convolution FFN resisted every tested scalar ultra-low-bit PTQ
  treatment below roughly two effective bits while retaining near-FP16 output.

Evidence:

- [universal compressor synthesis](../../CLARK_AIR_UNIVERSAL_COMPRESSOR.md)
- [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md)
- [self-oracle research](../../SELF_ORACLE_RESEARCH.md)
- [repository research README](../../README.md)

## Unresolved questions

1. How much of the step-31,000 gain came from AdamW8bit, proximal pull, scale
   refresh, island learning rates, or their interaction?
2. Does the released checkpoint remain non-inferior on a broad,
   preregistered multi-seed prompt suite rather than four release prompts?
3. Which island allocation minimizes visual distortion per packed bit?
4. Can the full packed graph run end to end with numerical parity and measured
   speed/memory advantages on Apple Silicon, x86 CPU, and NVIDIA?
5. Can UI text, typography, and multilingual image-text rendering survive the
   low-bit artifact under OCR-oriented evaluation?

## Next gates if this inactive lane is ever resumed

1. Pin the exact base revision, release checkpoint, source revision, prompt
   manifest, and seeds.
2. Reproduce the release comparison plus a broad held-out suite with
   prompt-fidelity, typography/OCR, composition, perceptual, and worst-decile
   metrics.
3. Run a matched factorial separating AdamW, proximal pull, scale refresh,
   continuous islands, and any Muon variant.
4. Include FP16, PTQ ternary, released QAT, binary, and hardware-relevant 4-bit
   controls at equal prompts and seeds.
5. Verify packed-to-unpacked output equivalence and measure bytes, peak memory,
   latency, throughput, and energy on each claimed runtime.

## Journal maintenance

Append dated entries only when a reproducible artifact changes a finding.
Every entry should include base/source revisions, checkpoint hash, exact
prompt/data manifest, seeds, control artifacts, packed-bit ledger, and whether
the result is a smoke, development result, or sealed holdout.

## 2026-08-01 — Status review

No Sana model is being loaded, evaluated, or trained under the BackdropKit
program. This dossier remains an archived image-model reference for gradual
hard-forward QAT, precision islands, and packed accounting. Its unresolved
quality and runtime gates above are unchanged.
