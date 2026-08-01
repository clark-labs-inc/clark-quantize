# Sana Image-Generation 1.58-bit Artifact Index

Status date: 2026-07-30

This is a link-only index. Historical files remain in place so scripts,
imports, release paths, and provenance records are not disturbed.

## Primary result and release records

| Artifact | Role | Evidence boundary |
| --- | --- | --- |
| [step-31,000 release manifest](../../releases/clark-air-sana-1.6b-s31000.json) | Exact quantization surface, packing, footprint, prompt set, and release checkpoint metadata | Four-prompt comparison; not a broad quality-parity receipt |
| [current Clark Air README](../../README.md) | Product summary plus PTQ/GPTQ historical results | Mixes shipped release facts with narrow research benchmarks |
| [release sharing README](../../SHARE_README.md) | Packed/unpacked footprint explanation and usage | Descriptive release surface |
| [ternary model card](../../release/MODEL_CARD.md) | Compatibility artifact documentation | Older preview language reports a different compression stage |
| [binary model card](../../release/MODEL_CARD_binary.md) | Binary comparison artifact | Preview branch, not the recommended image route |
| [technical paper source](../../paper/clark-quantize.tex) | Derivation, ablations, and PTQ/GPTQ results | Historical narrow prompt suite |

## Scientific synthesis

| Artifact | Role | Caution |
| --- | --- | --- |
| [universal compressor synthesis](../../CLARK_AIR_UNIVERSAL_COMPRESSOR.md) | Sana conclusions, stable negative objectives, and transfer lessons | Explicitly freezes Sana as the main optimization line |
| [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md) | Reconstructs actual released optimizer path and evidence grades | Historical result where local original checkpoint/log may be absent |
| [self-oracle research](../../SELF_ORACLE_RESEARCH.md) | Oracle/ranking hypotheses and failure gates | Contains research proposals as well as results |
| [Prism-level compression analysis](../../PRISM_LEVEL_COMPRESSION.md) | Packing/quality decomposition and ternary theory | Cross-model synthesis, not a Sana release receipt |

## Core quantization and packing implementations

| Artifact | Role |
| --- | --- |
| [optimal ternary quantizer](../../clark_quantize/ternary.py) | Closed-form MSE-optimal ternary support and scales |
| [activation-aware ternary GPTQ](../../clark_quantize/gptq.py) | GPTQ and incoherence rotation |
| [Bonsai quantized modules](../../bonsai_quant.py) | Hard-forward master-weight QAT modules |
| [Sana replay QAT](../../scripts/train_sana_replay_qat.py) | Replay/distillation training branch |
| [Sana 9c full QAT](../../scripts/train_sana_qat_9c_full.py) | Historical full QAT branch |
| [Sana GPTQ runner](../../scripts/quantize_sana_gptq.py) | Activation-aware quantization execution |
| [compact packer](../../scripts/pack_sana_8x.py) | Pack40/log-scale/FP8 artifact generation |
| [GemLite packer](../../scripts/pack_sana_gemlite.py) | NVIDIA-oriented aligned two-bit-slot artifact |
| [release loader](../../release/load_clark_compress.py) | Packed release reconstruction |
| [render verifier](../../scripts/render_verify.py) | Checkpoint render comparison |

## Evaluation data and ranking records

| Artifact | Role |
| --- | --- |
| [hard prompt rubrics](../../data/sana_hard_prompt_rubrics.json) | Prompt-specific visual criteria |
| [render rank preferences](../../data/sana_render_rank_preferences.json) | Known pairwise render preferences |
| [self-oracle counterfactuals](../../data/sana_self_oracle_counterfactuals.json) | Prompt-contrastive counterfactuals |
| [CVF hard prompts](../../data/sana_cvf_hard_prompts.json) | Counterfactual validation prompts |
| [PTD pairs](../../data/sana_ptd_pairs.json) | Paired trajectory-distillation inputs |
| [PTD interactions](../../data/sana_ptd_interactions.json) | PTD interaction definitions |

## Runtime and compatibility artifacts

| Artifact | Role |
| --- | --- |
| [GemLite artifact README](../../clark-air-sana-1.6b-gemlite-2bit/README.md) | CUDA/GemLite packaging surface |
| [ComfyUI pack builder](../../scripts/build_comfy_pack.py) | ComfyUI integration pack |
| [ComfyUI smoke](../../smoke_node.py) | Node-loading smoke path |
| [Hugging Face push helper](../../scripts/hf_push.py) | Artifact inspection/publishing helper |

## Record authority

For packed size and checkpoint structure, prefer the release manifest. For a
quality claim, prefer a run-specific receipt with prompts and seeds over model
cards or marketing copy. Reopen the actual artifact and runtime before making a
current availability, checksum, or performance claim.
