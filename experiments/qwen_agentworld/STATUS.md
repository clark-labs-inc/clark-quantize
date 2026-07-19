# Qwen AgentWorld Experiment Status

Last updated: 2026-06-29

## Current State

The A100 node is staged for the Clark Air Qwen AgentWorld compression lane.

- Official eval repo: `/workspace/Qwen-AgentWorld`
- Dataset snapshot: `/workspace/AgentWorldBench`
- Model snapshot: `/workspace/models/Qwen-AgentWorld-35B-A3B`
- Experiment outputs: `/workspace/agentworld_runs`
- Local experiment scripts: `/workspace/clark-quantize/experiments/qwen_agentworld`

The GPU was returned to idle after the smoke run.

Important validation boundary:

- Official model context: `262144`
- Official eval max generation tokens: `32768`
- Official serving topology: tensor parallelism `4`
- Current A100 node: `1x NVIDIA A100-SXM4-80GB`

Therefore the current single-A100 results are reduced-context smoke/canary
results only. They must not be treated as final AgentWorldBench validation or
release-quality compression evidence. Full-fidelity BF16 and compressed-system
validation needs a 4-GPU node or an equivalent full-context serving backend.

Full-fidelity preflight on the current A100:

- Command:
  `bash experiments/qwen_agentworld/run_full_fidelity_eval.sh bf16 preflight_full_bf16_262k_32768`
- Log:
  `/workspace/agentworld_runs/preflight_full_fidelity.log`
- Result:
  exit code `42`, refused because the node has `1` GPU and the official
  boundary requires tensor parallelism `4`.
- Full-fidelity setup script for the next node:
  `/workspace/clark-quantize/experiments/qwen_agentworld/setup_full_fidelity_node.sh`
- Full-fidelity runner:
  `/workspace/clark-quantize/experiments/qwen_agentworld/run_full_fidelity_eval.sh`
- Setup preflight log:
  `/workspace/agentworld_runs/setup_full_fidelity_preflight.log`
- Setup preflight result:
  exit code `42`, refused before installing vLLM because the node has `1` GPU
  and needs `4`.

## Current Binary/Ternary Focus

The active compression path is binary/ternary only:

- Keep `linear_attention_projection` on ternary QAT patch, not raw PTQ.
- Keep full attention protected for now.
- Keep shared experts ternary/protected unless a future contextual measure says
  otherwise.
- Use binary as MoE expert spillover selected by measured teacher-behavior
  damage, not by route fraction alone.

## Latest Code-Homotopy Findings

Magnitude-aware code homotopy is now implemented as a one-A100 probe in:

- `/workspace/clark-quantize/experiments/qwen_agentworld/train_code_homotopy_qat.py`
- `/Users/stan/Documents/git/clark-quantize/experiments/qwen_agentworld/train_code_homotopy_qat.py`

It trains compact code-transition logits plus group log-scales directly, rather
than dense fp32 masters or LoRA adapters. The current implementation is still a
research probe: it bakes selected modules into dense BF16 `safetensors` patches
for gate testing, not a final packed code artifact. It now also supports
`--soft-forward-steps`, which trains a continuous expected-code relaxation for
an initial warmup before snapping back to hard codebook forward and saving hard
weights.

The main result so far is that code movement is real but sharply
layer-specific. Boundary-near candidate selection moves codes; bottom-absolute
candidate selection was mostly inert. Broad late-MLP movement is unsafe.

Current fixed10 gate control/champion remains:

- base stack:
  - `/workspace/agentworld_runs/qwen35_9b_lexical_ternary_qat_smoke/lexical_ternary_smoke_patch.safetensors`
  - `/workspace/agentworld_runs/qwen35_9b_qat_full_attention_g128/full_attention_ternary_patch.safetensors`
  - `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/linearattn_survival_l28_29_30_s20_len1024_patch.safetensors`
- gate:
  `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/gate_linearattn_survival_l28_29_30_s20_fixed10_len8192_t96.pt`
- taxonomy:
  `8 parseable_output`, `2 gibberish`, `0 repetition/structural collapse`.

Code-homotopy safety map under the current boundary settings:

| Run | Surface | Code Movement | Fixed10 Gate Read | Decision |
| --- | --- | ---: | --- | --- |
| CH0b | layers 28-30 MLP, bottom-abs candidates | ~0.06% | weak/no useful movement | reject |
| CH1 | layers 28-30 MLP, boundary candidates | ~59% | `<roblox:...>` repetition collapse | reject |
| CH2 | layers 28-30 MLP, moderate boundary | ~34% | repetition collapse | reject |
| CH3 | layer 30 MLP only | ~44% | `8 parseable_output`, `2 gibberish`, no repetition | keep as research artifact |
| CH4 | layer 29 MLP only | ~44% | structural junk | reject |
| CH5 | layer 28 MLP only | ~45% | repetition/structural/gibberish, stopped early | reject |
| CH6 | layer 31 MLP on top of CH3 | ~38% | repetition collapse by row 4, stopped early | reject |
| CH7 | layer 27 MLP only | ~44% | repetition collapse on row 1, stopped early | reject |
| CH8 | layer 27 MLP, note-style soft settings but hard forward | 0% by step 150 | inert; stopped early | reject/no-op |
| CH9 | layer 27 MLP, soft-forward warmup 60 steps | ~12.7% | `8 parseable_output`, `2 gibberish`, no repetition | keep as research artifact |
| CH10 | layer 28 MLP, soft-forward warmup 60 steps | ~12.7% | repetition collapse on rows 1-2 | reject |
| CH3+CH9 stack | layer 30 + layer 27 retained patches | mixed | structural junk on row 1 | reject/no-compose |
| CH11 | layer 29 MLP, soft-forward warmup 60 steps | ~12.7% | `9 parseable_output`, `1 gibberish`, no repetition | keep as research artifact |
| CH11 fixed100 | CH11 layer 29 patch, first 14/100 rows | ~12.7% | `11 parseable_output`, `3 repetition_collapse`, stopped early | reject as promotable |
| CH12 | layer 31 MLP, soft-forward warmup 60 steps | ~10.7% | repetition collapse on rows 3-4 | reject |

Retained useful code-homotopy artifacts:

- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/code_homotopy_ch3_boundary_l30_mlp_g128_c250k_s45_lr2e2_m02_tw1e4_patch.safetensors`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/report_code_homotopy_ch3_boundary_l30_mlp_g128_c250k_s45_lr2e2_m02_tw1e4.json`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/gate_code_homotopy_ch3_boundary_l30_mlp_g128_c250k_s45_lr2e2_m02_tw1e4_fixed10_len8192_t96.pt`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/code_homotopy_ch9_anneal_l27_mlp_g128_c250k_s120_lr5e3_m05_sf60_patch.safetensors`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/report_code_homotopy_ch9_anneal_l27_mlp_g128_c250k_s120_lr5e3_m05_sf60.json`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/gate_code_homotopy_ch9_anneal_l27_mlp_g128_c250k_s120_lr5e3_m05_sf60_fixed10_len8192_t96.pt`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/code_homotopy_ch11_anneal_l29_mlp_g128_c250k_s120_lr5e3_m05_sf60_patch.safetensors`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/report_code_homotopy_ch11_anneal_l29_mlp_g128_c250k_s120_lr5e3_m05_sf60.json`
- `/workspace/agentworld_runs/qwen35_air_precision_rescue_20260629/gate_code_homotopy_ch11_anneal_l29_mlp_g128_c250k_s120_lr5e3_m05_sf60_fixed10_len8192_t96.pt`

Rejected dense CH6/CH7/CH10/CH12 patches were deleted after their
reports/logs/gates were kept, to avoid disk creep. Current read:

- Do not treat "late MLP" as one safe bucket.
- Safe code movement is narrower than expected under hard-forward training:
  layer 30 survives, while layers 27/28/29/31 collapse when moved aggressively.
- Movement fraction alone is not predictive: layer 30 survived with about the
  same movement that caused layers 27/28/29/31 to collapse.
- CH8/CH9 show the schedule matters. With the note-style margin/LR under hard
  forward, layer 27 did not move at all. With soft-forward warmup then hard
  snap, layer 27 moved about `12.7%` of selected candidates and matched the
  champion taxonomy. This is the first evidence that annealed code homotopy can
  recover a previously unsafe layer.
- The annealed schedule is layer-specific: it rescued layer 27 and layer 29,
  but layer 28 still collapsed immediately and layer 31 collapsed on MCP rows.
  Layer 29 is the best individual annealed result so far at `9 parseable_output`
  and `1 gibberish` versus the control's `8/2`.
- CH11 does not scale cleanly beyond the fixed10 canary. On the fixed100 subset
  it hit three repetition collapses by row 14 (`<model_response>` loops and a
  `<click/>` loop), so it remains a useful research artifact but is not a
  promotable 7-7.5x candidate.
- Safe individual code moves do not compose naively. The CH3+CH9 stack failed
  row 1 with structural junk even though CH3 and CH9 each passed individually.
- Next useful work is to build a sparse-code artifact path for single retained
  patches, then try a more conservative CH11 variant before attempting larger
  gates again. If composition is revisited, it needs a joint annealed
  multi-layer training run, not post-hoc stacking of individually safe dense
  patches.

## Global All-2D Mirror-QAT Pivot

The active hypothesis is now a real global QAT run rather than more local
patch probes. Prism/Bonsai forensics suggest the successful artifact is trained
globally into a codebook-native basis: all 2D tensors low-bit, 1D norms
continuous, no final adapters. The one-A100 approximation is:

- all language 2D tensors run ternary g128 in forward;
- trainable masters use BF16, not FP32;
- optimizer is no-moment group-normalized mirror SGD;
- no Adam/Muon state for matrix masters;
- alpha ramps from BF16 master forward to hard ternary forward;
- teacher cache is used first; on-policy AgentWorld survival comes later.

Implementation update:

- `train_all2d_ternary_qat.py` now supports `--master-dtype bfloat16`,
  `--train-layers all`, and `--optimizer mirror_sgd`.
- It forces CUDA garbage collection after replacing modules. This changed the
  one-A100 fit picture materially: the first attempt sat near the memory limit,
  while the garbage-collected relaunch idled around the mid-20GB range before
  the first full forward/backward spike.
- Full all-module codebook loss is not memory-safe at this context. The first
  global run OOMed while materializing codebook losses. The active run keeps
  `--code-weight 0` and relies on the prox pull for codebook pressure.

Run diagnostics:

- `qwen35_global_t2_mirror_v1b_nocode_s5000_ctx1024` proved that the graph
  fits, but the learning rate was too high. It reached step 25 at alpha
  `0.0048` with `KL=15.14`, `top1=0.0`, `mass=0.0878`; the run was killed.
- `qwen35_global_t2_mirror_v1c_lr2e6_s5000_ctx1024` used the same setup with
  LR reduced from `2e-4` to `2e-6`. It stayed stable through step 40:
  step 10 `KL=0.0283`, step 20 `KL=0.2136`, step 30 `KL=0.0748`, step 40
  `KL=0.0896`. It was intentionally stopped early only because it lacked
  periodic checkpoints.
- `qwen35_global_t2_mirror_v1d_lr2e6_ckpt_s5000_ctx1024` added latest
  checkpointing and reproduced the stable early trajectory through step 40. It
  was stopped because the first checkpoint writer used atomic replacement and
  would have required two full latest-patch copies worth of free disk on later
  saves.
- Active run:
  `qwen35_global_t2_mirror_v1e_lr2e6_ckpt_s5000_ctx1024`
  in `/workspace/agentworld_runs/qwen35_global_t2_mirror_20260629`.
  This is the same stable LR/settings as v1c/v1d, with disk-safe overwrite-style
  latest checkpoints every 500 steps.
- Active run early read:
  step 1 `KL=0.00315`, step 10 `KL=0.0282`, step 20 `KL=0.0925`,
  step 30 `KL=0.0783`, step 40 `KL=0.0908`. The low-alpha fuse has not fired,
  and top-k mass remains high (`~0.985-0.992` on the repeated web row). This is
  the current long-run lane. Later check: step 50 `KL=0.0692`, step 60
  `KL=0.0686`; still stable. The current live process was launched before the
  `--log-first-cycle` improvement, so every `log_every=10` row lands on the
  same cache row. Future launches will log the first full row cycle by default.

This is not yet a promotable artifact. It is the first real test of whether a
single A100 can run global all-2D QAT with state-light optimization. If it
remains stable through the alpha ramp, the next gates are fixed10 generation at
hard alpha, then on-policy survival cache, then longer-context curriculum.

Latest QAT-linear strict 8x candidate:

- Policy:
  `/workspace/agentworld_runs/policy_qwen_agentworld_qatlinear_protect_fullattn_rebinary8x.json`
- Projection patch:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/linear_projection_ternary_patch.safetensors`
- Corrected teacher-cache eval:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/strict8x_eval.json`
- Metrics:
  - KL full: `0.62492`
  - KL top-k: `0.58739`
  - teacher top-k mass: `0.96840`
  - top1 agreement: `0.83929`
  - compression: `8.00005x`

Contextual binary replacement probes:

- Existing hot binary MoE backfill sample:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/group_damage_backfill_hot32_contextual.json`
  - avg KL: `0.05149`
  - max KL: `0.25329`
- Low-route shared-expert ternary-to-binary sample:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/group_damage_remaining_lowroute32_contextual.json`
  - avg KL: `0.76674`
  - max KL: `12.59307`
  - conclusion: shared experts are not safe binary replacement targets.
- Low-route MoE-only ternary-to-binary sample:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/group_damage_remaining_moe_lowroute64_contextual.json`
  - avg KL: `0.02598`
  - max KL: `0.20782`
  - conclusion: there is a safer MoE binary pool, but individual outliers still
    exist.
- Broader low-route MoE-only ternary-to-binary sample:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/group_damage_remaining_moe_lowroute256_contextual.json`
  - avg KL: `0.02882`
  - max KL: `0.39149`
  - conclusion: many MoE slices are near-zero binary damage, but route fraction
    still admits severe outliers. Binary MoE allocation must use measured
    damage, not route alone.

Manual equal-bit swaps were tested:

- `swap_hot32_low16_moe`:
  `/workspace/agentworld_runs/policy_qwen_agentworld_qatlinear_strict8x_swap_hot32_low16_moe.json`
  eval:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/swap_hot32_low16_eval.json`
  - KL full: `0.61880`
  - mass: `0.96751`
  - top1: `0.83571`
  - read: slight KL improvement, top1/mass regression; not a clean promotion.
- `swap_hot16_low8_moe`:
  `/workspace/agentworld_runs/policy_qwen_agentworld_qatlinear_strict8x_swap_hot16_low8_moe.json`
  eval:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/swap_hot16_low8_eval.json`
  - KL full: `0.63652`
  - mass: `0.96722`
  - top1: `0.82857`
  - read: worse than the current strict candidate.
- `swap_hot32_low32_down_moe`:
  `/workspace/agentworld_runs/policy_qwen_agentworld_qatlinear_strict8x_swap_hot32_low32_down_moe.json`
  eval:
  `/workspace/agentworld_runs/qwen_projection_qat_full_v2/swap_hot32_low32_down_eval.json`
  - KL full: `0.61567`
  - mass: `0.96931`
  - top1: `0.83214`
  - compression: `8.00005x`
  - no-judge official smoke:
    `/workspace/agentworld_runs/smoke_1x7_qatlinear_swap_hot32_low32_down_no_think_64/predictions.jsonl`
  - degeneracy:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/degeneracy_swap_hot32_low32_down_no_think_64.json`
    - collapsed: `0/7`
  - BF16 sequence comparison:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_swap_hot32_low32_down.json`
    - avg sequence ratio: `0.30079`
  - OpenRouter Qwen 3.5 Flash judge:
    `/workspace/agentworld_runs/smoke_1x7_qatlinear_swap_hot32_low32_down_no_think_64/judged.jsonl`
    `/workspace/agentworld_runs/smoke_1x7_qatlinear_swap_hot32_low32_down_no_think_64/score_openrouter_qwen35_flash.txt`
    - official score: `0.00`
  - read: best strict-8x cache KL so far and no generation collapse, but not
    system-useful on AgentWorldBench. Generations are alive but malformed, often
    emitting literal `<think>`/state-copy text instead of valid task actions.

Conclusion: ternary QAT plus binary MoE spillover is still the right line, but
cache/logit gates are not sufficient validation. The next binary/ternary step
should be either (a) QAT/polish under AgentWorld-format teacher behavior, or
(b) a larger measured allocator plus an official-smoke gate. Do not promote a
policy on KL/sequence similarity without AgentWorld judge improvement.

## Latest AgentWorld System-Performance Ablations

The strict 8x binary/ternary policies are alive at the token level but not
system-useful yet. A 7-domain, 64-token, no-thinking smoke judged with
OpenRouter `qwen/qwen3.5-flash-02-23` currently gives this ladder:

| Variant | Compression Surface | Global Compression | Smoke Score | Notes |
| --- | --- | ---: | ---: | --- |
| BF16 reference | none | `1.00x` | `7.86` | Current 64-token judge reference. |
| QAT projection patch only | ternary `linear_attention_projection` patch | low | `5.00` | Patch alone degrades most domains; only Web received partial credit. |
| MoE binary+ternary only | MoE experts from strict policy, no patch/head/embed | `5.92x` | `5.00` | Alive, but binary spillover hurts system output. |
| MoE ternary-only | only MoE expert ternary groups, binary groups BF16 | `2.34x` | `14.29` | Best system result so far; Terminal strong, Android/Web partial. |
| MoE ternary stage128 | measured MoE ternary subset plus 128 lowest-route formerly-binary MoE groups promoted to ternary | `2.36x` | `8.57` | Cheap gates passed, but judged score regressed. |
| MoE ternary stage256 | stage128 plus another 128 low-route MoE groups | `2.38x` | `11.43` | Best staged addition so far, but still below the base measured ternary subset. |
| MoE ternary stage512 | stage256 plus another 256 low-route MoE groups | `2.45x` | `10.71` | Non-monotonic regression from stage256. |
| MoE ternary stage1024 | stage512 plus another 512 low-route MoE groups | `2.58x` | `7.86` | Regression to BF16-reference smoke level. |
| MoE ternary stage2048 | stage1024 plus another 1024 low-route MoE groups | `2.90x` | `8.57` | Slightly above BF16 reference, still below stage512/base measured subset. |
| MoE+shared ternary-only | MoE ternary groups plus shared-expert ternary | `2.36x` | `6.43` | Shared experts regress the judged smoke despite tiny rate gain. |
| All policy MoE ternary | every policy-quantized MoE expert group ternary, no binary | `5.57x` | `5.00` | Blanket expert ternary is not safe; measured subset matters. |
| Strict swap combined | QAT projection + MoE binary/ternary + FP8 head/embed | `8.00x` | `0.00` | Alive but malformed; emits `<think>`/state-copy text. |

Relevant artifacts:

- QAT projection patch-only:
  - predictions:
    `/workspace/agentworld_runs/smoke_1x7_qatlinear_patch_only_no_think_64/predictions.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_1x7_qatlinear_patch_only_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap compare:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_qatlinear_patch_only.json`
- MoE binary+ternary only:
  - predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_only_swap_hot32_low32_down_no_think_64/predictions.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_1x7_moe_only_swap_hot32_low32_down_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap compare:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_moe_only_swap_hot32_low32_down.json`
- MoE ternary-only:
  - predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_only_swap_hot32_low32_down_no_think_64/predictions.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_only_swap_hot32_low32_down_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap compare:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_moe_ternary_only_swap_hot32_low32_down.json`
- Staged MoE ternary additions:
  - ignore:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage128_no_think_64/predictions.jsonl`
    was produced during an interrupted, misconfigured runner attempt that
    applied binary and ternary MoE groups. It has no judge/summary artifact and
    is not part of the ternary-only ladder.
  - policies:
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage128.json`
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage256.json`
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage512.json`
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage1024.json`
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage2048.json`
  - stage128 predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage128_tonly_no_think_64/predictions.jsonl`
  - stage128 score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage128_tonly_no_think_64/score_openrouter_qwen35_flash.txt`
  - stage128 cheap-gate summary:
    `/workspace/agentworld_runs/qwen_binary_ternary_runs/smoke_1x7_moe_ternary_stage128_tonly_no_think_64.summary.json`
  - stage256 predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage256_tonly_no_think_64/predictions.jsonl`
  - stage256 score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage256_tonly_no_think_64/score_openrouter_qwen35_flash.txt`
  - stage256 cheap-gate summary:
    `/workspace/agentworld_runs/qwen_binary_ternary_runs/smoke_1x7_moe_ternary_stage256_tonly_no_think_64.summary.json`
  - stage512 predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage512_no_think_64/predictions.jsonl`
  - stage512 score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage512_no_think_64/score_openrouter_qwen35_flash.txt`
  - stage1024 predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage1024_no_think_64/predictions.jsonl`
  - stage1024 score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage1024_no_think_64/score_openrouter_qwen35_flash.txt`
  - stage2048 predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage2048_no_think_64/predictions.jsonl`
  - stage2048 score:
    `/workspace/agentworld_runs/smoke_1x7_moe_ternary_stage2048_no_think_64/score_openrouter_qwen35_flash.txt`
- MoE+shared ternary-only:
  - predictions:
    `/workspace/agentworld_runs/smoke_1x7_moe_shared_ternary_only_no_think_64/predictions.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_1x7_moe_shared_ternary_only_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap compare:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_moe_shared_ternary_only.json`
- All policy MoE ternary:
  - policy:
    `/workspace/agentworld_runs/policy_qwen_agentworld_all_moe_ternary_no_binary.json`
  - predictions:
    `/workspace/agentworld_runs/smoke_1x7_all_moe_ternary_no_binary_no_think_64/predictions.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_1x7_all_moe_ternary_no_binary_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap compare:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_all_moe_ternary_no_binary.json`

Read: broad MoE ternary is currently the only low-bit surface that improves
the judged smoke over the stored BF16 64-token reference, but only for the
measured ternary subset. Adding shared experts or blanket-ternarizing all
policy MoE expert groups regresses the official smoke. The staged MoE ternary
ladder shows the same non-monotonic behavior: stage256 is the best staged
addition at `11.43`, while stage128/stage512/stage1024/stage2048 all regress
relative to the base measured ternary-only subset at `14.29`. Binary expert
spillover and ternary projection QAT both need AgentWorld-format-aware
training/polish before they can be used in an 8x system-performance candidate.
The next rate increase should be a measured binary/ternary allocator using
official-smoke gates, not route-only ternary expansion, all-MoE ternary, or
naive binary spillover.

## Larger 3x7 Validation Slice

A larger deterministic subset was added:

- Dataset:
  `/workspace/AgentWorldBench_smoke_3x7`
- Composition:
  `3` tasks from each of Android, MCP, OS, Search, SWE, Terminal, Web
  (`21` total).
- Runner:
  `/workspace/clark-quantize/experiments/qwen_agentworld/run_binary_ternary_smoke.sh`
- Judge:
  OpenRouter `qwen/qwen3.5-flash-02-23`
- Generation boundary:
  `MAX_MODEL_LEN=8192`, `MAX_TOKENS=64`, no chat-template thinking channel.

| Variant | Compression Surface | Smoke Score | Cheap Gate | Notes |
| --- | --- | ---: | --- | --- |
| BF16 reference | none | `6.43` | no collapse | 21 valid samples. |
| MoE ternary-only measured subset | only MoE expert ternary groups from the measured strict-swap policy | `4.29` | no collapse, BF16 sequence ratio `0.573`, exact `2/21` | Regresses on the larger slice despite being the 1x7 champion. |
| MoE ternary pruned highproxy64 KL<=0.0005 | source-filtered MoE ternary policy with 33/64 measured high-proxy groups pruned to BF16 | `3.57` | no collapse, BF16 sequence ratio `0.560`, exact `1/21` | Negative result; pruning measured high-proxy groups did not recover system score. |

Artifacts:

- BF16 3x7:
  - predictions:
    `/workspace/agentworld_runs/smoke_3x7_bf16_8192_no_think_64/predictions.jsonl`
  - judged:
    `/workspace/agentworld_runs/smoke_3x7_bf16_8192_no_think_64/judged.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_3x7_bf16_8192_no_think_64/score_openrouter_qwen35_flash.txt`
  - summary:
    `/workspace/agentworld_runs/qwen_binary_ternary_runs/smoke_3x7_bf16_8192_no_think_64.summary.json`
- MoE ternary-only 3x7:
  - predictions:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_only_no_think_64/predictions.jsonl`
  - judged:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_only_no_think_64/judged.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_only_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap-gate summary:
    `/workspace/agentworld_runs/qwen_binary_ternary_runs/smoke_3x7_moe_ternary_only_no_think_64.summary.json`
- MoE ternary pruned highproxy64 3x7:
  - measurement:
    `/workspace/agentworld_runs/qwen_projection_qat_full_v2/group_damage_3x7_active_moe_ternary_highproxy64.json`
  - policy:
    `/workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_prune3x7_highproxy64_kl0005.json`
  - predictions:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_prune_highproxy64_kl0005_no_think_64/predictions.jsonl`
  - judged:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_prune_highproxy64_kl0005_no_think_64/judged.jsonl`
  - score:
    `/workspace/agentworld_runs/smoke_3x7_moe_ternary_prune_highproxy64_kl0005_no_think_64/score_openrouter_qwen35_flash.txt`
  - cheap-gate summary:
    `/workspace/agentworld_runs/qwen_binary_ternary_runs/smoke_3x7_moe_ternary_prune_highproxy64_kl0005_no_think_64.summary.json`

Read: the larger slice weakens the previous 1x7 result. The measured MoE
ternary subset is still alive and non-collapsed, but it does not preserve
system performance on 3x7. Treat 1x7 as a fast canary only. Future
binary/ternary policies should clear 3x7 before any promotion, and the next
useful experiment is not more route-only ternary expansion or simple
high-proxy pruning; it is AgentWorld-behavior-aware ternary training or a
measured allocator that uses 3x7 official-smoke gates. Separately, final
validation must move to the official 262k/32768-token boundary.

## Completed Checks

1. Created isolated environment: `/workspace/agentworld-venv`.
2. Downloaded official `Qwen/AgentWorldBench` dataset.
3. Downloaded official `Qwen/Qwen-AgentWorld-35B-A3B` model.
4. Built a 7-sample domain-balanced smoke subset:
   `/workspace/AgentWorldBench_smoke_1x7`.
5. Loaded model config/tokenizer successfully:
   - `model_type = qwen3_5_moe`
   - `architectures = Qwen3_5MoeForConditionalGeneration`
   - `language_model_only = True`
6. Started a reduced-context Transformers OpenAI-compatible endpoint:
   - `MAX_MODEL_LEN=8192`
   - port `8000`
   - peak GPU memory about 70.5 GiB during smoke inference
7. Ran official `eval.py infer` on the 7-domain smoke subset:
   `/workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl`.
8. Collected router-utilization calibration on the 7-domain smoke subset:
   `/workspace/agentworld_runs/router_calibration_smoke_512.json`.
9. Built two projected 8x policy artifacts:
   - `/workspace/agentworld_runs/policy_qwen_agentworld_8x_uncalibrated.json`
   - `/workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json`
10. Ran a bounded whole-tensor packing smoke:
    `/workspace/agentworld_runs/pack_smoke_qwen_agentworld/manifest.json`.
11. Ran bounded expert-slice packer smokes:
    - `/workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts/manifest.json`
    - `/workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts/manifest.json`
12. Verified all three pack manifests can be loaded and reconstructed:
    - `/workspace/agentworld_runs/pack_smoke_qwen_agentworld/verify.json`
    - `/workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts/verify.json`
    - `/workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts/verify.json`
13. Served a bounded fake-quant endpoint and ran the official 7-domain
    inference smoke:
    `/workspace/agentworld_runs/smoke_1x7_fake_quant_32groups/predictions.jsonl`.
14. Compared BF16 and fake-quant smoke predictions without a paid judge:
    `/workspace/agentworld_runs/compare_bf16_vs_fake_quant_32groups.json`.
15. Ran a bounded teacher-behavior damage probe for policy groups:
    `/workspace/agentworld_runs/group_damage_smoke_12groups.json`.
16. Ran a bounded binary expert-slice damage probe:
    `/workspace/agentworld_runs/group_damage_smoke_binary_12groups.json`.
17. Built a corrected measured-damage mixed-rate policy:
    `/workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json`.
18. Compared router-proxy and corrected measured policies:
    `/workspace/agentworld_runs/policy_compare_router_vs_measured_smoke24_monotonic.json`.
19. Packed and verified expert slices from the corrected measured policy:
    `/workspace/agentworld_runs/pack_smoke_qwen_agentworld_measured24_monotonic_experts/verify.json`.
20. Packed and verified parent-tensor mixed-rate artifacts from the corrected
    measured policy:
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_smoke/verify.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_binary_smoke/verify.json`
21. Packed and verified an FP8 parent-tensor artifact from the corrected
    measured policy:
    `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_fp8_smoke/verify.json`.
22. Ran a no-limit dry run for the full corrected measured policy:
    `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full_dryrun/dry_run.json`.
23. Executed and verified the full corrected measured-policy parent-tensor
    pack:
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/verify.json`
24. Served a corrected measured-policy fake-quant endpoint and ran the official
    7-domain inference smoke:
    `/workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups/predictions.jsonl`.
25. Compared corrected measured-policy predictions against BF16 without a paid
    judge:
    `/workspace/agentworld_runs/compare_bf16_vs_fake_quant_measured24_monotonic_32groups.json`.
26. Served a packed-manifest-loaded endpoint and ran the official 7-domain
    inference smoke:
    `/workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/predictions.jsonl`.
27. Compared packed-manifest predictions against BF16 and fake-quant smoke
    predictions without a paid judge:
    - `/workspace/agentworld_runs/compare_bf16_vs_packed_manifest_32groups.json`
    - `/workspace/agentworld_runs/compare_fake_quant_vs_packed_manifest_32groups.json`
28. Served a packed-runtime endpoint that replaced one whole-tensor
    `torch.nn.Linear` with a packed-manifest-backed runtime module and ran the
    official 7-domain inference smoke:
    `/workspace/agentworld_runs/smoke_1x7_packed_runtime_1linear_cpu/predictions.jsonl`.
29. Compared packed-runtime predictions against BF16 and packed-manifest smoke
    predictions without a paid judge:
    - `/workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1linear_cpu.json`
    - `/workspace/agentworld_runs/compare_packed_manifest_vs_packed_runtime_1linear_cpu.json`
30. Served a larger 8-linear packed-runtime endpoint and ran a tiny direct
    liveness request:
    - `/workspace/agentworld_runs/packed_runtime_report_8linear_cpu.json`
    - `/workspace/agentworld_runs/packed_runtime_8linear_cpu_direct_smoke.json`
31. Served a fused MoE expert packed-runtime endpoint that replaced one full
    `Qwen3_5MoeExperts` module from packed parent tensors, then ran the
    official 7-domain inference smoke at a bounded 256-token output budget:
    - `/workspace/agentworld_runs/packed_runtime_report_1expert_layer_cpu.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_1expert_layer_cpu_256/predictions.jsonl`
    - `/workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1expert_layer_cpu_256.json`
32. Scaled the fused MoE expert runtime to four full expert modules and ran
    the same official 7-domain bounded inference smoke:
    - `/workspace/agentworld_runs/packed_runtime_report_4expert_layers_cpu.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_cpu_256/predictions.jsonl`
    - `/workspace/agentworld_runs/compare_bf16_vs_packed_runtime_4expert_layers_cpu_256.json`
33. Ran paid LLM judge scoring through OpenRouter using
    `qwen/qwen3.5-flash-02-23` on the core smoke artifacts:
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192/judged.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/judged.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_cpu_256/judged.jsonl`
    - `/workspace/agentworld_runs/judge_summary_openrouter_qwen35_flash_smoke_core.json`
34. Added and tested active-expert GPU cache mode for the 4-layer fused expert
    runtime:
    - `/workspace/agentworld_runs/packed_runtime_report_4expert_layers_active64.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_256/predictions.jsonl`
    - `/workspace/agentworld_runs/packed_runtime_stats_4expert_layers_active64_after_smoke.json`
    - `/workspace/agentworld_runs/compare_cpu_vs_active64_packed_runtime_4expert_layers_256.json`
35. Completed OpenRouter Qwen 3.5 Flash judge scoring for all distinct
    7-domain smoke prediction artifacts:
    `/workspace/agentworld_runs/judge_summary_openrouter_qwen35_flash_all_smoke.json`.
36. Fixed the reduced-context Transformers serving path to disable Qwen's
    chat-template thinking channel by default. The corrected BF16 no-thinking
    smoke is:
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think/judged.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think/score_openrouter_qwen35_flash.txt`
37. Re-ran the strongest measured-policy fake-quant smoke under the same
    no-thinking serving boundary:
    - `/workspace/agentworld_runs/fake_quant_report_measured24_monotonic_32groups_no_think.json`
    - `/workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups_no_think/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups_no_think/judged.jsonl`
    - `/workspace/agentworld_runs/compare_bf16_no_think_vs_fake_quant_measured24_monotonic_32groups_no_think.json`
38. Added a bounded no-thinking 256-token runtime comparison for the 4-layer
    active-cache packed-runtime candidate:
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think_256/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_no_think_256/predictions.jsonl`
    - `/workspace/agentworld_runs/packed_runtime_report_4expert_layers_active64_no_think.json`
    - `/workspace/agentworld_runs/packed_runtime_stats_4expert_layers_active64_no_think_after_smoke.json`
    - `/workspace/agentworld_runs/compare_bf16_no_think_256_vs_packed_runtime_4expert_layers_active64_no_think_256.json`
39. Attempted the same 4-layer active-cache packed runtime at the corrected
    8192-token no-thinking smoke length:
    - `/workspace/agentworld_runs/packed_runtime_report_4expert_layers_active64_no_think_8192.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_no_think_8192/predictions.jsonl`
    - `/workspace/agentworld_runs/serve_packed_runtime_4expert_layers_active64_no_think_8192.log`
    The run is not judgeable: 4/7 generations are empty after CUDA OOM in
    Transformers' MoE expert materialization path.
40. Replaced all 40 fused expert modules with the packed active-cache runtime
    and ran a direct 8192-context OS liveness request:
    - `/workspace/agentworld_runs/packed_runtime_report_all_experts_active8_no_think.json`
    - `/workspace/agentworld_runs/models_after_all_experts_active8_direct_os.json`
    - `/workspace/agentworld_runs/serve_packed_runtime_all_experts_active8_no_think.log`
    The direct request completed without the previous selected-expert OOM.
    The endpoint reported `replaced_expert_count = 40` and
    `removed_parameter_bytes = 64424509440`. This proves full expert
    replacement avoids the Transformers MoE materialization failure, but the
    Python-loop active-cache runtime is still a liveness/sanity path, not a
    production long-context serving kernel.
41. Added `active_lazy` packed-runtime cache mode for all-expert replacement.
    This avoids reconstructing and stacking all packed experts on CPU during
    server startup, and the endpoint reached readiness with all 40 experts
    replaced:
    - `/workspace/agentworld_runs/packed_runtime_report_all_experts_active_lazy8_no_think_256.json`
    - `/workspace/agentworld_runs/serve_packed_runtime_all_experts_active_lazy8_no_think_256.log`
    The official 7-domain 256-token smoke did not produce a first prediction
    row after roughly four minutes; the server sat CPU-bound with very low GPU
    utilization and high VRAM residency. The run was stopped and the GPU was
    returned to idle. Conclusion: lazy all-expert replacement fixes startup
    memory pressure, but per-token Python expert reconstruction is too slow for
    evaluation without a real packed MoE kernel or a better hot-expert cache.
42. Added a bounded lazy CPU expert cache for `active_lazy` packed-runtime
    serving and validated it on a one-row official AgentWorldBench smoke:
    - `/workspace/agentworld_runs/smoke_1x1_web_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/models_after_all_experts_active_lazy8_cpu256_no_think_64_1x1.json`
    - `/workspace/agentworld_runs/smoke_1x1_web_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64/score_openrouter_qwen35_flash.txt`
    - `/workspace/agentworld_runs/smoke_1x1_web_bf16_8192_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x1_web_bf16_8192_no_think_64/score_openrouter_qwen35_flash.txt`
    - `/workspace/agentworld_runs/compare_bf16_no_think_64_vs_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64_1x1_web.json`
    Configuration:
    - all 40 fused expert modules replaced
    - `cache = active_lazy`
    - GPU active cache limit = 8 experts per module
    - lazy CPU cache limit = 256 experts per module
    - `MAX_TOKENS = 64`
    Runtime stats after the one-row web smoke:
    - removed BF16 expert parameters: `64424509440` bytes
    - lazy CPU cache: `10113` expert slices, `63625494528` bytes
    - active GPU cache: `320` expert slices, `2013265920` bytes
    - cache hits/misses/evictions: `34883 / 25647 / 25327`
    - lazy CPU hits/misses/evictions: `15534 / 10113 / 0`
    The fair same-budget comparison against BF16 no-thinking 64-token output:
    - packed judged score: `15.00`
    - BF16 judged score: `15.00`
    - exact matches: `0/1`
    - average sequence ratio: `0.9954`
    Conclusion: all-expert packed reconstruction can preserve the one-row
    output closely, but the runtime is still not an eval-scale serving path.
    It used roughly 64GB CPU cache and took about 3.5 minutes for one web row,
    while BF16 produced the same-budget row in about 5 seconds after model
    load. The next serving milestone is a fused packed MoE kernel or a
    calibrated hot-expert preload/runtime that avoids Python expert loops.
43. Ran the same all-expert `active_lazy` + lazy CPU cache path on the full
    7-domain official smoke subset at a bounded 64-token budget:
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/compare_bf16_no_think_64_vs_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64.json`
    - `/workspace/agentworld_runs/models_after_all_experts_active_lazy8_cpu256_no_think_64_7x.json`
    - `/workspace/agentworld_runs/summary_bf16_vs_all_experts_active_lazy8_cpu256_no_think_64_7x.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64/score_openrouter_qwen35_flash.txt`
    - `/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think_64/score_openrouter_qwen35_flash.txt`
    Results:
    - packed rows: `7`, empty generations: `0`
    - BF16 rows: `7`, empty generations: `0`
    - cheap sequence ratio, BF16 vs packed: `0.6478`
    - OpenRouter Qwen 3.5 Flash judge, BF16: `7.86`
    - OpenRouter Qwen 3.5 Flash judge, packed all-expert runtime: `3.57`
    - final lazy CPU cache: `10240` expert slices,
      `64424509440` bytes
    - final active GPU cache: `320` expert slices, `2013265920` bytes
    Runtime behavior:
    - first packed row was slow because it populated the lazy CPU cache
    - later packed rows completed in roughly 35-46 seconds each
    - BF16 completed all seven 64-token rows in roughly 33 seconds after
      sample load
    Conclusion: the all-expert packed runtime is now end-to-end evaluable on
    the official smoke subset, but current PTQ/mixed-rate expert replacement is
    not quality-equivalent to BF16 and the Python-loop runtime is still far too
    slow. The next compression milestone should be teacher-behavior QAT or a
    better measured policy, not serving-kernel work alone.
44. Isolated compression-policy damage from packed-runtime damage by applying
    the same full expert packed manifest back into normal BF16 Transformers
    parameters, then running native MoE inference on the same 7-domain
    64-token smoke:
    - `/workspace/agentworld_runs/packed_manifest_report_all_experts_no_think_64.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_all_experts_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_all_experts_no_think_64/score_openrouter_qwen35_flash.txt`
    - `/workspace/agentworld_runs/compare_bf16_no_think_64_vs_packed_manifest_all_experts_no_think_64.json`
    - `/workspace/agentworld_runs/compare_packed_manifest_all_experts_vs_packed_runtime_all_experts_active_lazy8_cpu256_no_think_64.json`
    - `/workspace/agentworld_runs/summary_bf16_vs_manifest_vs_runtime_all_experts_no_think_64_7x.json`
    Manifest load result:
    - applied expert groups: `20480`
    - missing groups: `0`
    Three-way 7-domain score comparison with OpenRouter Qwen 3.5 Flash:
    - BF16 64-token baseline: `7.86`
    - native dequantized packed manifest: `5.71`
    - all-expert packed runtime: `3.57`
    Cheap output comparison:
    - BF16 vs dequantized manifest sequence ratio: `0.6075`
    - dequantized manifest vs packed runtime sequence ratio: `0.9595`
    - dequantized manifest vs packed runtime exact matches: `4/7`
    Conclusion: the packed runtime is close to the native dequantized manifest
    behavior, so the dominant quality loss comes from the current PTQ/mixed
    expert policy itself. The packed Python runtime still adds smaller
    residual drift and is much slower, but QAT / better teacher-behavior
    rate allocation is now the higher-value next branch than more serving
    glue.
45. Tested the older uncalibrated all-trunk-ternary 8x policy as a cleaner
    alternative to the measured mixed binary/ternary policy:
    - `/workspace/agentworld_runs/policy_qwen_agentworld_8x_uncalibrated.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_uncalibrated_full/manifest.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_uncalibrated_full/verify.json`
    - `/workspace/agentworld_runs/packed_manifest_report_uncalibrated_all_ternary_no_think_64.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_uncalibrated_all_ternary_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_uncalibrated_all_ternary_no_think_64/score_openrouter_qwen35_flash.txt`
    - `/workspace/agentworld_runs/compare_bf16_no_think_64_vs_packed_manifest_uncalibrated_all_ternary_no_think_64.json`
    - `/workspace/agentworld_runs/compare_measured_manifest_vs_uncalibrated_all_ternary_no_think_64.json`
    Pack and load result:
    - manifest complete: `true`
    - parent artifacts: `693`
    - groups: `21093`
    - artifact bytes: `7857262940`
    - source bytes: `69321221376`
    - artifact compression: `8.82x`
    - rates: `20830` ternary40_log8 groups, `261` BF16 groups, `2` FP8 groups
    - manifest verification: `21093/21093` groups, `ok = true`,
      max RMSE delta `1.49e-7`
    - native Transformers load: `applied_count = 21093`, `missing_count = 0`
    7-domain 64-token result:
    - rows: `7`, empty generations: `0`
    - OpenRouter Qwen 3.5 Flash judge: `0.00`
    - BF16 vs uncalibrated sequence ratio: `0.0804`
    - measured-manifest vs uncalibrated sequence ratio: `0.1038`
    Failure mode:
    - BF16 and measured-policy manifest still emit plausible AgentWorld text.
    - uncalibrated all-ternary emits repetitive digit/quote/comma fragments
      such as `100000...`, `',10', '1'`, and repeated blank quote tokens.
    Conclusion: the uncalibrated all-trunk-ternary policy is a clean negative.
    It reaches strict 8x storage and loads correctly, but it destroys language
    behavior. The measured mixed policy remains the better PTQ baseline despite
    using binary cold expert groups. The next useful branch is not naive
    all-ternary packing; it is teacher-behavior QAT plus Fisher/rate-distortion
    allocation before repacking.
46. Added ternary-vs-binary measurement support to
    `measure_group_damage.py` and used it to probe the low-bit boundary:
    - `/workspace/agentworld_runs/group_damage_hot_bf16_moe_12groups_ternary_binary.json`
    - `/workspace/agentworld_runs/group_damage_hot_ternary_moe_12groups_ternary_binary.json`
    - `/workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_hot48_monotonic.json`
    - `/workspace/agentworld_runs/policy_compare_measured_smoke24_vs_hot48_monotonic.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured_hot48_monotonic_full/manifest.json`
    - `/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured_hot48_monotonic_full/verify.json`
    - `/workspace/agentworld_runs/smoke_1x7_packed_manifest_measured_hot48_no_think_64/predictions.jsonl`
    - `/workspace/agentworld_runs/summary_measured_hot48_manifest_no_think_64_7x.json`
    Measurement setup:
    - select hot MoE expert groups by route fraction
    - measure both `ternary40_log8` and `binary_log8` for the same group
    - calibration: 3 AgentWorld smoke jobs, max length 512, last 8 logits
    Boundary measurement result:
    - hot BF16 groups forced low-bit: `24` measured rows,
      avg KL `0.0507`, max KL `0.1824`
    - hot ternary groups measured as ternary/binary: `24` measured rows,
      avg KL `0.0493`, max KL `0.1131`
    - ternary is generally safer than binary on hot routed experts, but local
      last-logit KL is noisy and not sufficient as a rollout-quality selector.
    Hot48 measured policy:
    - measured overrides: `72`
    - projected compression: `8.005x`
    - packed artifact compression: `8.416x`
    - changed groups vs previous measured24 policy: `23`
    - changes: `12` BF16 -> ternary, `11` ternary -> BF16
    - manifest verification: `21093/21093` groups, `ok = true`
    - native Transformers load: `applied_count = 21093`, `missing_count = 0`
    7-domain 64-token rollout:
    - rows: `7`, empty generations: `0`
    - BF16 vs hot48 sequence ratio: `0.1017`
    - measured24 manifest vs hot48 sequence ratio: `0.1013`
    - paid judge was intentionally skipped because all seven generations were
      visibly collapsed into repetitive digit/quote/blank fragments.
    Conclusion: this is a useful negative. The measurement path can compare
    binary versus ternary on the same groups, but local teacher-logit damage
    must be used as a diagnostic/guardrail, not as sufficient evidence to
    demote hot BF16 expert slices. PTQ-only ternary/binary policies still need
    either hard rollout gates or teacher-behavior QAT before demoting hot
    expert slices.

Smoke inference output:

- rows: 7
- tasks: android, mcp, os, search, swe, terminal, web
- empty generations: 0

Judge scoring was run after the user confirmed OpenRouter and Qwen 3.5 Flash.
The judge wrapper now passes the key via environment instead of exposing it as a
command-line argument.

## Model Inventory

Enhanced safetensors metadata inventory:

`/workspace/agentworld_runs/model_inventory.json`

Summary:

```text
total size:        64.560 GiB
target 8x size:     8.070 GiB
tensor count:      693

moe_expert                  60.000 GiB
linear_attention_projection  1.882 GiB
embedding                    0.947 GiB
lm_head                      0.947 GiB
full_attention               0.508 GiB
shared_expert                0.235 GiB
router                       0.039 GiB
linear_attention_state       0.002 GiB
norm                         ~0 GiB
other                        ~0 GiB
```

The compression problem is therefore expert-dominated. The first Clark Air
rate-distortion pass should estimate Fisher/router utilization for MoE expert
tensors before spending time on attention or tiny protected tensors.

## Router Calibration

Smoke calibration:

```text
artifact:   /workspace/agentworld_runs/router_calibration_smoke_512.json
jobs:       7
max length: 512 tokens
routers:    40
top-k:       8
```

Example hot experts from the smoke calibration:

```text
layer 0:  expert 62 = 3.82%, expert 231 = 2.17%, expert 124 = 1.87%
layer 10: expert 81 = 2.83%, expert 232 = 2.79%, expert 245 = 2.62%
layer 39: expert 70 = 7.49%, expert 30 = 3.51%, expert 34 = 3.29%
```

## Mixed-Rate Policy Artifacts

Uncalibrated proxy policy:

```text
artifact:     /workspace/agentworld_runs/policy_qwen_agentworld_8x_uncalibrated.json
projected:    7.531 GiB
compression:  8.573x
rates:        FP8 embeddings/head, BF16 protected tensors, ternary everything else
```

Router-calibrated smoke policy:

```text
artifact:     /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json
projected:    8.063 GiB
target:       8.070 GiB
compression:  8.007x
rates:
  BF16 groups:        547
  FP8 groups:           2
  ternary groups:   18,562
  binary groups:     1,982
```

The calibrated policy keeps some hot expert slices BF16, converts cold expert
slices to binary, and leaves most of the model ternary. Its damage estimates
are still proxy estimates; the next scientific upgrade is replacing proxy
damage with teacher-behavior empirical Fisher.

Corrected measured-damage smoke policy:

```text
artifact:     /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json
projected:    8.069 GiB
target:       8.070 GiB
compression:  8.001x
overrides:    24 measured rows
floors:       11 unmeasured lower-bit candidates raised by measured floors
rates:
  BF16 groups:        554
  FP8 groups:           2
  ternary groups:   18,555
  binary groups:     1,982
```

Comparison to router-proxy policy:

```text
artifact:        /workspace/agentworld_runs/policy_compare_router_vs_measured_smoke24_monotonic.json
changed groups:  15
changes:
  ternary40_log8 -> BF16:   11
  BF16 -> ternary40_log8:    4
```

There is also an older non-monotonic measured policy artifact in the output
directory. Do not use it as a candidate; it is kept only as evidence of the
allocator pitfall described below.

## Teacher-Behavior Damage Probe

Ternary/early-policy artifact:

`/workspace/agentworld_runs/group_damage_smoke_12groups.json`

Scope:

```text
calibration jobs: 3
max length:       256
groups measured:  12
rates:            ternary40_log8
buckets:          linear_attention_projection + moe_expert
missing params:   0
avg KL:           0.0121916905
max KL:           0.0208825485
top1 agreement:   1.0 for sampled rows
```

Binary expert-slice artifact:

`/workspace/agentworld_runs/group_damage_smoke_binary_12groups.json`

Scope:

```text
calibration jobs: 3
max length:       256
groups measured:  12
rates:            binary_log8
bucket:           moe_expert
missing params:   0
avg KL:           0.00000000745
max KL:           0.00000000745
```

The binary rows are cold expert slices in this tiny smoke set; the first row
has `route_fraction = 0.0` and zero logit MSE. Treat this as proof that the
measurement path works, not as enough evidence that binary experts are safe
globally.

This is the first non-judge teacher-behavior measurement for the Qwen policy.
It loads the BF16 teacher, fake-quantizes one policy group at a time, compares
last-token logits on AgentWorld smoke prompts, and restores the original
weights after each group. It is still a bounded smoke, not a full empirical
Fisher pass.

Hardware result: this probe completed on the single A100 80GB and returned the
GPU to idle afterward. A 2-GPU node is not required for this stage.

Allocator bug found and fixed: a measured ternary damage row can make an
unmeasured binary proxy look artificially cheap for the same group. The policy
builder now enforces a per-group measured-damage floor so unmeasured lower-bit
choices cannot undercut measured higher-bit choices. Measured candidates remain
authoritative.

## Packing Smoke

Artifact:

`/workspace/agentworld_runs/pack_smoke_qwen_agentworld/manifest.json`

Packed four whole-tensor policy choices:

```text
model.language_model.layers.0.linear_attn.in_proj_a.weight
  source 131,072 bytes -> packed 14,769 bytes
model.language_model.layers.0.linear_attn.in_proj_b.weight
  source 131,072 bytes -> packed 15,138 bytes
model.language_model.layers.0.linear_attn.in_proj_z.weight
  source 16,777,216 bytes -> packed 1,791,534 bytes
model.language_model.layers.0.linear_attn.out_proj.weight
  source 16,777,216 bytes -> packed 1,794,671 bytes
```

Expert-slice ternary smoke:

```text
artifact: /workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts/manifest.json
verify:   /workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts/verify.json
verified: 6 tensors/slices
ok:       true
```

Expert-slice binary smoke:

```text
artifact: /workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts/manifest.json
verify:   /workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts/verify.json
verified: 4 tensors/slices
ok:       true
```

Corrected measured-policy expert-slice smoke:

```text
artifact: /workspace/agentworld_runs/pack_smoke_qwen_agentworld_measured24_monotonic_experts/manifest.json
verify:   /workspace/agentworld_runs/pack_smoke_qwen_agentworld_measured24_monotonic_experts/verify.json
verified: 6 tensors/slices
ok:       true
max RMSE delta: 8.94e-08
```

Parent-tensor mixed-rate smoke:

```text
artifact: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_smoke/manifest.json
verify:   /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_smoke/verify.json
parent:   model.language_model.layers.0.mlp.experts.down_proj
groups:   24 expert slices in one parent artifact
rates:    9 BF16, 15 ternary
source:   50,331,648 bytes
artifact: 15,389,407 bytes
ok:       true
```

Parent-tensor binary smoke:

```text
artifact: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_binary_smoke/manifest.json
verify:   /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_binary_smoke/verify.json
parent:   model.language_model.layers.1.mlp.experts.down_proj
groups:   4 binary expert slices in one parent artifact
source:   8,388,608 bytes
artifact: 582,256 bytes
ok:       true
```

Parent-tensor FP8 smoke:

```text
artifact: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_fp8_smoke/manifest.json
verify:   /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_fp8_smoke/verify.json
parent:   lm_head.weight
groups:   1 FP8 tensor group
source:   1,017,118,720 bytes
artifact: 349,191,786 bytes
mode:     torch.float8_e4m3fn bytes
ok:       true
```

Full corrected-policy dry run:

```text
artifact: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full_dryrun/dry_run.json
parents:  693
groups:   21,093
source:   69,321,221,376 bytes
rates:
  BF16 groups:        554
  FP8 groups:           2
  ternary groups:   18,555
  binary groups:     1,982
```

Full corrected-policy parent pack:

```text
artifact: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json
verify:   /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/verify.json
complete: true
parents:  693
groups:   21,093
source:   69,321,221,376 bytes
artifact: 8,240,037,667 bytes
ratio:    8.413x
verify:   ok
verified: 21,093 groups
max RMSE delta: 1.49e-07
rates:
  BF16 groups:        554
  FP8 groups:           2
  ternary groups:   18,555
  binary groups:     1,982
```

This proves the Clark Air packer path works on actual Qwen tensors and on the
virtual MoE expert-slice policy groups used by the calibrated policy. The new
parent-tensor format groups multiple expert slices into one file, covers the
full corrected policy, and verifies every packed group. The remaining storage
gap is no longer packing; it is loading/running the packed artifact.

## Fake-Quant Endpoint Smoke

Endpoint report:

```text
artifact: /workspace/agentworld_runs/fake_quant_report_32groups_v2.json
applied:  32 groups
missing:  0 groups
scope:    linear_attention_projection + moe_expert
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_fake_quant_32groups/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
```

This is a behavioral endpoint proof, not a memory-saving runtime. The fake
quantized weights are dequantized back into normal model tensors, so GPU memory
usage remains essentially BF16-sized.

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_fake_quant_32groups.json
rows:                7
BF16 empty outputs:  0
fake empty outputs:  0
exact matches:       0
avg sequence ratio:  0.386
```

This comparison only checks transport and coarse output drift. It does not
replace AgentWorldBench judge scoring.

Corrected measured-policy endpoint report:

```text
artifact: /workspace/agentworld_runs/fake_quant_report_measured24_monotonic_32groups.json
applied:  32 groups
missing:  0 groups
scope:    linear_attention_projection + moe_expert
policy:   policy_qwen_agentworld_8x_measured_smoke24_monotonic.json
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
```

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_fake_quant_measured24_monotonic_32groups.json
rows:                7
BF16 empty outputs:  0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.241
```

This is still not a quality score. The smoke proves the corrected measured
policy can be applied to a real endpoint and produce non-empty official
predictions on all seven domains.

## Packed-Manifest Endpoint Smoke

Endpoint report:

```text
artifact: /workspace/agentworld_runs/packed_manifest_report_32groups.json
manifest: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json
applied:  32 groups
missing:  0 groups
scope:    linear_attention_projection + moe_expert
rates:    ternary40_log8 + binary_log8
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
```

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_packed_manifest_32groups.json
rows:                7
BF16 empty outputs:  0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.254
```

Non-judge comparison to measured fake-quant smoke:

```text
artifact:            /workspace/agentworld_runs/compare_fake_quant_vs_packed_manifest_32groups.json
rows:                7
fake empty outputs:  0
packed empty:        0
exact matches:       0
avg sequence ratio:  0.328
```

This proves the packed artifact can be used as the source of endpoint weights.
It still dequantizes into normal model parameters, so runtime GPU memory remains
BF16-sized. The next runtime milestone is replacing selected linear/expert
paths with packed-aware modules instead of copying dequantized weights into
ordinary tensors.

## Packed-Runtime Linear Endpoint Smoke

Endpoint report:

```text
artifact:                /workspace/agentworld_runs/packed_runtime_report_1linear_cpu.json
manifest:                /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json
replaced linears:         1
skipped:                  0
cache:                    cpu
removed parameter bytes:  33,554,432
scope:                    first whole-tensor linear_attention_projection group
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_packed_runtime_1linear_cpu/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
```

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1linear_cpu.json
rows:                7
BF16 empty outputs:  0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.289
```

Non-judge comparison to packed-manifest smoke:

```text
artifact:            /workspace/agentworld_runs/compare_packed_manifest_vs_packed_runtime_1linear_cpu.json
rows:                7
baseline empty:      0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.240
```

This is the first memory-saving runtime proof: the selected module no longer
stores its BF16 weight as a model parameter. It is intentionally not a fast
kernel. With `cache=cpu`, the module reconstructs/caches the dequantized weight
on CPU and copies it to GPU during forward, so long generations are slower.
This proves the replacement path and memory accounting, not the final serving
performance. The next runtime step is scaling replacement coverage with either
GPU-side cache for speed experiments or a real packed GEMM/expert runtime for
memory-saving performance.

Larger direct liveness smoke:

```text
artifact:                /workspace/agentworld_runs/packed_runtime_report_8linear_cpu.json
direct response:          /workspace/agentworld_runs/packed_runtime_8linear_cpu_direct_smoke.json
replaced linears:         8
skipped:                  0
cache:                    cpu
removed parameter bytes:  134,479,872
startup resident memory:  about 66.7 GiB on the single A100
result:                   non-empty response to direct local chat request
```

This confirms that the packed-runtime replacement path scales past one module
on the single A100. The response was only a liveness check, not a quality
measurement; it did not replace the official 7-domain smoke. A 2-GPU node is
not needed for this replacement-development lane. The useful hardware jump for
faithful Qwen serving remains the official-style 4-GPU, 262k-context setup.

## Packed-Runtime Fused Expert Endpoint Smoke

The Qwen3.5 MoE implementation stores expert weights as two fused parameters
inside each `Qwen3_5MoeExperts` module:

```text
gate_up_proj: [256, 1024, 2048]
down_proj:    [256, 2048, 512]
```

The packed-runtime expert smoke replaced one whole fused expert module when
both packed parent tensors were available in the manifest.

Endpoint report:

```text
artifact:                  /workspace/agentworld_runs/packed_runtime_report_1expert_layer_cpu.json
direct response:            /workspace/agentworld_runs/packed_runtime_1expert_layer_cpu_direct_smoke.json
replaced normal linears:     0
replaced expert modules:     1
skipped:                    0
cache:                      cpu
removed parameter bytes:    1,610,612,736
startup resident memory:    about 66.7 GiB on the single A100
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_packed_runtime_1expert_layer_cpu_256/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
budget:   MAX_TOKENS=256, TEMPERATURE=0
runtime:  about 2 minutes for 7 requests in the CPU-cache prototype
```

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1expert_layer_cpu_256.json
rows:                7
BF16 empty outputs:  0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.340
```

This is the first expert-dominated memory-saving runtime proof. It removes a
large fused expert layer from GPU parameter storage and still runs through the
official AgentWorld inference harness. It remains a smoke implementation: in
`cache=cpu` mode every hit expert copies its dequantized slice from CPU to GPU
during forward. The path is correct enough for bounded validation, but the real
serving milestone is a packed/fused expert kernel or a smarter active-expert
GPU cache.

Four-layer fused expert scaling smoke:

```text
artifact:                  /workspace/agentworld_runs/packed_runtime_report_4expert_layers_cpu.json
replaced expert modules:    4
skipped:                    0
cache:                      cpu
removed parameter bytes:    6,442,450,944
startup resident memory:    about 66.7 GiB on the single A100
```

Official AgentWorld inference smoke:

```text
artifact: /workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_cpu_256/predictions.jsonl
rows:     7
empty:    0
tasks:    android, mcp, os, search, swe, terminal, web
budget:   MAX_TOKENS=256, TEMPERATURE=0
runtime:  about 3 minutes for 7 requests in the CPU-cache prototype
```

Non-judge comparison to BF16 smoke:

```text
artifact:            /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_4expert_layers_cpu_256.json
rows:                7
BF16 empty outputs:  0
candidate empty:     0
exact matches:       0
avg sequence ratio:  0.390
```

This shows the fused-expert runtime scales beyond a single layer on the one-A100
development lane. The blocker is no longer basic replacement correctness; it is
serving efficiency. CPU-cache forward gets slower as more expert layers are
replaced because every active expert copy crosses CPU to GPU during generation.

## OpenRouter Qwen 3.5 Flash Judge Smoke

Judge configuration:

```text
judge base URL: https://openrouter.ai/api/v1
judge model:    qwen/qwen3.5-flash-02-23
temperature:    0
max tokens:     8192 for the expanded pass
summary:        /workspace/agentworld_runs/judge_summary_openrouter_qwen35_flash_all_smoke.json
```

Judged smoke results:

```text
run                                                     valid  failed  overall
smoke_1x7_bf16_8192_no_think                           7      0       16.43
smoke_1x7_fake_quant_measured24_monotonic_32groups_no_think 7  0       15.00
smoke_1x7_fake_quant_measured24_monotonic_32groups     7      0       14.29
smoke_1x7_packed_runtime_4expert_layers_active64_no_think_256 7 0     10.71
smoke_1x7_bf16_8192_no_think_256                       7      0       7.86
smoke_1x7_packed_runtime_1linear_cpu                   7      0       5.71
smoke_1x7_packed_runtime_1expert_layer_cpu_256         7      0       3.57
smoke_1x7_bf16_8192                                    7      0       2.14
smoke_1x7_packed_manifest_32groups                     7      0       2.14
smoke_1x7_fake_quant_32groups                          7      0       1.43
smoke_1x7_packed_runtime_4expert_layers_active64_256   7      0       1.43
smoke_1x7_packed_runtime_4expert_layers_cpu_256        7      0       1.43
```

Artifact paths:

```text
/workspace/agentworld_runs/judge_summary_openrouter_qwen35_flash_all_smoke.json
/workspace/agentworld_runs/*/judged.jsonl
/workspace/agentworld_runs/*/score_openrouter_qwen35_flash.txt
```

Read this as an end-to-end scoring smoke, not as final benchmark quality. The
original BF16 reduced-context Transformers endpoint also scored low on this
tiny 7-sample subset because the tokenizer chat template opened Qwen's
`<think>` channel and the server returned raw visible reasoning in
`message.content`. The corrected no-thinking BF16 run improves from 2.14 to
16.43 and produces closed `<predicted_observation>` tags for Android/Web plus a
clean terminal output. It still does not prove final AgentWorldBench quality
for the official full-context deployment.

The active-cache 4-layer run produced byte-identical predictions to the
CPU-cache 4-layer run, so its judged file was copied from the already judged
CPU-cache output instead of spending a duplicate paid judge pass.

Do not compare old compressed smoke scores directly against
`smoke_1x7_bf16_8192_no_think` as final compression evidence. Re-run each
compression candidate under the same no-thinking serving setting first.

The first apples-to-apples no-thinking compression comparison is:

```text
baseline:  smoke_1x7_bf16_8192_no_think
candidate: smoke_1x7_fake_quant_measured24_monotonic_32groups_no_think
score:     16.43 -> 15.00
compare:   exact matches 2/7, avg sequence ratio 0.635
```

This is a fake-quant behavioral check, not a memory-saving runtime check. The
next comparable runtime check should regenerate the packed-runtime expert-cache
candidate with no-thinking enabled before judging it.

The first no-thinking packed-runtime check uses the bounded 256-token lane so
it is comparable to the earlier runtime smoke:

```text
baseline:  smoke_1x7_bf16_8192_no_think_256
candidate: smoke_1x7_packed_runtime_4expert_layers_active64_no_think_256
score:     7.86 -> 10.71
compare:   exact matches 2/7, avg sequence ratio 0.809
runtime:   replaced 4 fused expert modules, removed 6,442,450,944 BF16 bytes
cache:     256 active experts resident, 1.61 GiB active GPU cache
hits:      89,551
misses:    28,339
evictions: 28,083
```

Treat the 256-token score as a runtime sanity gate, not a final quality claim:
the output cap truncates Android/Web tags and depresses absolute scores. The
useful signal is that the no-thinking packed-runtime active-cache path produces
non-empty official eval outputs, preserves two rows exactly, and remains
score-comparable to the corrected BF16 bounded baseline.

The attempted 8192-token packed-runtime active-cache run failed as a long-context
runtime gate:

```text
candidate: smoke_1x7_packed_runtime_4expert_layers_active64_no_think_8192
rows:      7
empty:     4
completed: android, mcp, web
failed:    os, search, swe, terminal
error:     CUDA OOM in transformers/integrations/moe.py selected expert weights
request:   tried to allocate about 99-109 GiB
live use:  about 65 GiB process memory on one A100 80GB
```

Do not judge this artifact. The failure is a runtime implementation limit, not
a scored compression-quality result. The fallback Transformers MoE path
materializes selected expert weights too broadly at long sequence length. A
proper long-context packed-runtime path needs either a custom packed MoE kernel,
smaller chunked prefill, or the official multi-GPU SGLang/vLLM serving stack.

## Research Note: Pure 1-bit AgentWorld Branch

The attached 1-bit analysis is useful as a candidate branch, especially because
it correctly emphasizes that Qwen-AgentWorld is expert-dominated. Current local
inventory agrees on the key storage fact:

```text
total reference bits: 554,569,771,008
moe_expert params:    32,212,254,720
router params:        20,971,520
```

Actionable pieces:

- Keep routers and DeltaNet state tensors protected; their storage impact is
  tiny and router damage would be high-leverage.
- Add expert-coverage calibration before any QAT branch; inactive experts will
  otherwise get weak distillation signal.
- Add a pure-binary g128 PTQ/damage branch as a research target, separate from
  the current mixed-rate 8x lane.

Do not replace the validated mixed-rate policy with pure 1-bit yet. The current
measured policy is an 8.001x mixed-rate artifact, not a 14x pure-binary
artifact:

```text
BF16 groups:      554
FP8 groups:         2
ternary groups: 18555
binary groups:   1982
```

Full 1-bit STE QAT is not a one-A100 job if it needs both FP16 teacher and
student training state. Treat it as a multi-GPU training branch after the
current PTQ/runtime evidence is stable.

## Active Expert Cache Runtime Smoke

The `active` cache mode keeps dequantized expert slices on CPU, promotes hit
experts to GPU on demand, and evicts with a per-module LRU cap.

Configuration:

```text
cache mode:          active
active cache limit:  64 experts per fused expert module
replaced modules:    4 Qwen3_5MoeExperts modules
removed BF16 bytes:  6,442,450,944
```

Artifacts:

```text
/workspace/agentworld_runs/packed_runtime_report_4expert_layers_active64.json
/workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_256/predictions.jsonl
/workspace/agentworld_runs/packed_runtime_stats_4expert_layers_active64_after_smoke.json
/workspace/agentworld_runs/compare_bf16_vs_packed_runtime_4expert_layers_active64_256.json
/workspace/agentworld_runs/compare_cpu_vs_active64_packed_runtime_4expert_layers_256.json
```

Official AgentWorld inference smoke:

```text
rows:     7
empty:    0
budget:   MAX_TOKENS=256, TEMPERATURE=0
runtime:  about 2.5 minutes for 7 requests
```

Cache stats after the smoke:

```text
active experts resident:       256 total across 4 modules
active GPU cache bytes:        1,610,612,736
cache hits:                    98,482
cache misses:                  29,972
evictions:                     29,716
per-module active cache bytes: 402,653,184
```

Output comparison:

```text
BF16 vs active64:
  rows: 7, candidate empty: 0, avg sequence ratio: 0.390

CPU-cache 4-layer vs active64:
  rows: 7, exact matches: 7, avg sequence ratio: 1.000
```

Because active64 produced byte-identical predictions to the already judged
CPU-cache 4-layer runtime on this smoke, no additional paid judge call was run
for active64. It inherits the same judged smoke result for this exact output.
The useful signal is runtime: active-cache preserved output and reduced the
7-request bounded smoke from about 3 minutes to about 2.5 minutes, at the cost
of 1.61 GiB active GPU cache.

## Important Implementation Notes

- `make_subset.py` must parse JSONL by file iteration, not `str.splitlines()`.
  AgentWorldBench rows can contain Unicode separator characters inside strings.
- The single-A100 lane is a reduced-context smoke/development lane only. The
  official Qwen deployment uses tensor parallelism and 262k context, which
  needs a larger node for faithful full-benchmark inference.
- The judge wrapper must not pass API keys as command-line arguments. Use
  `JUDGE_API_KEY` / `OPENAI_API_KEY` in the environment.
- The current bounded calibration, fake-quant, and packing development steps
  fit on one A100 80GB. Move to multi-GPU only for full-context serving,
  larger-batch QAT/calibration, or if a concrete OOM appears.
- Do not duplicate the model snapshot on this node. `/workspace` has limited
  headroom after staging the 65 GiB model.

## Next Gates

1. Improve the baseline serving path before treating absolute judge numbers as
   product-quality numbers:
   - compare against Qwen's recommended vLLM/SGLang settings on a larger node
   - inspect output formatting / response tags for the reduced-context endpoint
2. Implement AgentWorld calibration extraction:
   - broader teacher KL / logit agreement
   - hidden geometry
   - empirical Fisher diagonal for expert tensors
3. Replace proxy damage globally in the rate policy with empirical Fisher
   damage, preserving the measured-damage monotonic floor.
4. Decide the next runtime engineering path:
   - tune active-expert GPU cache capacity and layer coverage for smoke-scale
     validation
   - true packed/fused expert kernel for product serving
5. Serve the larger memory-saving packed endpoint and rerun the same official
   smoke subset.

## Ternary/Binary Long-Horizon Check

The current focus is the ternary/binary boundary, not more FP8/BF16 policy
reshuffling. The best measured mixed manifest remains the short-horizon
reference, but it does not pass longer generation.

Artifacts:

```text
/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json
/workspace/agentworld_runs/smoke_1x7_packed_manifest_measured24_no_think_256/predictions.jsonl
/workspace/agentworld_runs/compare_bf16_no_think_256_vs_packed_manifest_measured24_no_think_256.json
/workspace/agentworld_runs/summary_measured24_manifest_no_think_256_7x.json
```

Result:

```text
MAX_TOKENS:         256
temperature:        0
rows:               7
candidate empty:    0
avg sequence ratio: 0.0035
judge:              skipped
```

The paid OpenRouter judge was intentionally skipped. The 256-token outputs
visibly collapsed into repeated punctuation, zeros, blanks, or quote runs
across domains, while the BF16 baseline emitted normal AgentWorld-style
observations/actions. This makes the measured24 manifest a useful 64-token
PTQ/rate-allocation baseline, but not a sequence-stable compressed model.

Automated degeneration gate:

```text
script: experiments/qwen_agentworld/detect_generation_degeneracy.py
gate:   fail if >=25% rows trigger collapse heuristics
```

The gate checks empty/whitespace output, dominant repeated characters, long
single-character runs, tiny character sets, repeated character n-grams, and
punctuation-without-content. It is deliberately model-free and should run
before any paid judge call.

Calibration panel:

```text
bf16_64:          collapsed 0/7, failed false, entropy 4.734, repeat 0.082
bf16_256:         collapsed 0/7, failed false, entropy 4.928, repeat 0.092
measured24_64:    collapsed 0/7, failed false, entropy 4.738, repeat 0.084
measured24_256:   collapsed 7/7, failed true,  entropy 0.555, repeat 1.000
hot48_64:         collapsed 7/7, failed true,  entropy 0.693, repeat 1.000
all_ternary_64:   collapsed 6/7, failed true,  entropy 1.546, repeat 0.934
```

Gate artifacts:

```text
/workspace/agentworld_runs/degeneracy_bf16_no_think_64.json
/workspace/agentworld_runs/degeneracy_bf16_no_think_256.json
/workspace/agentworld_runs/degeneracy_measured24_manifest_no_think_64.json
/workspace/agentworld_runs/degeneracy_measured24_manifest_no_think_256.json
/workspace/agentworld_runs/degeneracy_measured_hot48_manifest_no_think_64.json
/workspace/agentworld_runs/degeneracy_uncalibrated_all_ternary_no_think_64.json
```

Conclusion:

- Ternary is still the safer low-bit rate than binary for hot/high-sensitivity
  groups, based on measured last-logit KL probes.
- Binary can remain in the allocator, but only for low-sensitivity groups that
  survive rollout gates.
- PTQ-only local damage is insufficient. Hot48 and measured24 both show that
  local last-logit KL can look acceptable while rollout behavior collapses.
- Next serious branch should be ternary/binary QAT with on-policy teacher KL
  or hidden-geometry/Fisher damage, then rate-distortion allocation. Do not run
  another paid judge for a policy that already collapses under 256-token
  greedy generation.

## No-Binary Ternary Diagnostic

To isolate whether the 256-token collapse was caused by the binary spillover,
we built a diagnostic policy from measured24 with every `binary_log8` group
converted to `ternary40_log8`.

Policy:

```text
/workspace/agentworld_runs/policy_qwen_agentworld_measured24_no_binary_all_ternary.json
source: measured24 monotonic policy
changed: 1,982 binary groups -> ternary groups
theoretical policy compression: 7.79x
rates after override:
  BF16 groups:      554
  FP8 groups:         2
  ternary groups: 20537
  binary groups:      0
```

Pack/verify:

```text
manifest: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_no_binary_all_ternary_full/manifest.json
groups:   21093 / 21093
verify:   ok, max_rmse_delta 1.49e-7
actual compressed artifact ratio: 8.18x
```

256-token smoke:

```text
run:                smoke_1x7_packed_manifest_measured24_no_binary_all_ternary_no_think_256
rows:               7
candidate empty:    0
avg sequence ratio: 0.0023
degeneration gate:  failed, collapsed 7/7
judge:              skipped
```

Artifacts:

```text
/workspace/agentworld_runs/smoke_1x7_packed_manifest_measured24_no_binary_all_ternary_no_think_256/predictions.jsonl
/workspace/agentworld_runs/degeneracy_measured24_no_binary_all_ternary_no_think_256.json
/workspace/agentworld_runs/compare_bf16_no_think_256_vs_measured24_no_binary_all_ternary_no_think_256.json
/workspace/agentworld_runs/summary_measured24_no_binary_all_ternary_no_think_256_7x.json
/workspace/agentworld_runs/failed_pack_records/measured24_no_binary_all_ternary/
```

The failed full pack directory was removed after preserving the lightweight
evidence bundle above, recovering about 8.0 GiB. The measured24 reference pack
was not touched.

Conclusion: binary spillover is not the sole cause of the long-horizon
collapse. A safer all-ternary replacement for the binary groups still degenerates
at 256 tokens. The next path is not another PTQ-only ternary/binary policy; it
is hard-forward ternary/binary QAT with rollout/on-policy teacher behavior gates.

## Teacher-Behavior Cache for Ternary/Binary QAT

Added a local, no-judge teacher-behavior cache/eval lane:

```text
experiments/qwen_agentworld/collect_teacher_behavior.py
experiments/qwen_agentworld/eval_teacher_behavior_cache.py
```

Purpose:

- collect BF16 Qwen teacher top-k logits on AgentWorld calibration prompts;
- evaluate packed ternary/binary students against the same cached teacher
  behavior without rerunning the teacher or paying a judge;
- provide a concrete optimization target for the next QAT/scale-training branch.

Teacher cache collected:

```text
cache:      /workspace/agentworld_runs/teacher_behavior_smoke_1x7_len512_last8_top256.pt
metadata:   /workspace/agentworld_runs/teacher_behavior_smoke_1x7_len512_last8_top256.pt.json
data:       /workspace/AgentWorldBench_smoke_1x7
rows:       7
max_length: 512
last_n:     8
top_k:      256
```

BF16 sanity eval:

```text
artifact: /workspace/agentworld_runs/teacher_behavior_eval_bf16_smoke_1x7_len512_last8_top256.json
teacher_topk_kl_full_mean:        0.01419
teacher_topk_kl_topk_mean:        0.00000
student_mass_on_teacher_topk:     0.98631
top1_agreement:                   1.00000
```

Measured24 packed-manifest eval:

```text
artifact: /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_smoke_1x7_len512_last8_top256.json
manifest report: /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_report.json
applied groups: 21093
teacher_topk_kl_full_mean:        9.66758
teacher_topk_kl_topk_mean:        6.47228
student_mass_on_teacher_topk:     0.08744
top1_agreement:                   0.05357
```

This cached metric agrees with the 256-token rollout gate: the measured24
ternary/binary PTQ manifest is far from the BF16 teacher distribution even
before paid judging. Use this objective as the cheap inner loop for the next
hard-forward QAT branch.

Next implementation step:

1. Scale the teacher cache beyond the 7-row smoke, still using local BF16 only.
2. Add a trainable metadata/scale QAT pass over selected ternary/binary groups
   that optimizes cached teacher top-k KL.
3. Repack and require both gates before judge spend:
   - cached teacher-behavior top-k KL/top1 gate;
   - 256-token degeneration gate.

## Corrected Left-Truncated Teacher-Behavior Gate

The first teacher-cache and measured-damage probes used the tokenizer default
right truncation. On AgentWorld prompts this overweights repeated
domain/system-prefix text and can miss the current observation/request tail.
Both cache/damage scripts now set:

```python
tokenizer.truncation_side = "left"
```

Superseded / weakened evidence:

- `/workspace/agentworld_runs/teacher_behavior_smoke_1x7_len512_last8_top256.pt`
- `/workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_smoke_1x7_len512_last8_top256.json`
- earlier `measure_group_damage.py` outputs that were collected before the
  left-truncation fix

Corrected balanced cache:

```text
cache:      /workspace/agentworld_runs/teacher_behavior_balanced_5x7_left512_last8_top256.pt
metadata:   /workspace/agentworld_runs/teacher_behavior_balanced_5x7_left512_last8_top256.pt.json
data:       /workspace/AgentWorldBench
rows:       35 (5 per domain)
max_length: 512
last_n:     8
top_k:      256
```

BF16 sanity eval on the corrected cache:

```text
artifact: /workspace/agentworld_runs/teacher_behavior_eval_bf16_balanced_5x7_left512_last8_top256.json
rows:                                35
teacher_topk_kl_full_mean:           0.0000918
teacher_topk_kl_topk_mean:           0.0000000
student_mass_on_teacher_topk_mean:   0.9999083
top1_agreement:                      1.0000000
```

Measured24 mixed ternary/binary manifest eval on the corrected cache:

```text
artifact:        /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_balanced_5x7_left512_last8_top256.json
manifest report: /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_balanced_5x7_left512_report.json
rows:                                35
teacher_topk_kl_full_mean:           10.81758
teacher_topk_kl_topk_mean:           8.46360
student_mass_on_teacher_topk_mean:   0.14903
top1_agreement:                      0.00000
```

Conclusion:

- The corrected cache validates the evaluator: BF16 nearly matches exactly.
- The current measured24 mixed ternary/binary PTQ manifest fails the teacher
  behavior gate decisively before any paid judge or long rollout.
- The no-binary all-ternary diagnostic already showed binary spillover is not
  the sole problem. The failure is PTQ low-bit sequence stability itself.
- The next mainline should focus on trainable hard-forward ternary/binary QAT:
  optimize teacher top-k KL / on-policy teacher KL on the corrected cache, then
  rerun the 256-token degeneration gate before any OpenRouter judge spend.

## Fixed-Code Low-Bit Scale Probe

Added a rate-scale override to the packed-manifest evaluator:

```text
experiments/qwen_agentworld/packed_manifest_loader.py
experiments/qwen_agentworld/eval_teacher_behavior_cache.py
```

This keeps the same packed ternary/binary codes and multiplies reconstructed
groups for selected rates before evaluating the corrected teacher-behavior
cache. It is a cheap discriminator for:

```text
bad scale metadata  vs.  bad low-bit support/code allocation
```

Probe:

```text
manifest: /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json
cache:    /workspace/agentworld_runs/teacher_behavior_balanced_5x7_left512_last8_top256.pt
scales:   ternary40_log8=0.75, binary_log8=0.75
artifact: /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_balanced_5x7_left512_scale075_last8_top256.json
report:   /workspace/agentworld_runs/teacher_behavior_eval_measured24_manifest_balanced_5x7_left512_scale075_report.json
```

Result:

```text
baseline measured24:
  KL full mean:        10.81758
  KL top-k mean:       8.46360
  teacher top-k mass:  0.14903
  top1 agreement:      0.00000

0.75x ternary/binary scales:
  KL full mean:        10.11894
  KL top-k mean:       8.96701
  teacher top-k mass:  0.61493
  top1 agreement:      0.00000
```

Interpretation:

- Shrinking low-bit scales improves mass on the teacher top-k support, so
  scale calibration is a real axis.
- It does not recover top-1 agreement or bring KL near BF16. Fixed PTQ codes
  remain behaviorally wrong.
- Do not continue with scale-only sweeps as the mainline. Use the rate-scale
  hook for diagnostics, then move to hard-forward ternary/binary QAT or
  behavioral Fisher reallocation.

## Ternary vs Binary Subset Gate

Ran the corrected teacher-behavior cache with only one low-bit rate applied at
a time. This isolates whether the measured24 failure comes from binary spillover
or from the much larger ternary trunk.

Artifacts:

```text
binary only:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_binary_only_balanced_5x7_left512_last8_top256.json
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_binary_only_balanced_5x7_left512_report.json

ternary only:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_only_balanced_5x7_left512_last8_top256.json
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_only_balanced_5x7_left512_report.json
```

Results:

```text
BF16 sanity:
  KL full mean:        0.0000918
  teacher top-k mass:  0.9999083
  top1 agreement:      1.00000

binary_log8 only:
  KL full mean:        0.0034788
  KL top-k mean:       0.0033764
  teacher top-k mass:  0.9998977
  top1 agreement:      0.98929

ternary40_log8 only:
  KL full mean:        11.05727
  KL top-k mean:       9.10461
  teacher top-k mass:  0.22056
  top1 agreement:      0.00000

full measured24:
  KL full mean:        10.81758
  KL top-k mean:       8.46360
  teacher top-k mass:  0.14903
  top1 agreement:      0.00000
```

Policy mass by rate:

```text
bf16:
  groups: 554
  params: 475,247,232
  largest bucket: moe_expert, 291 groups, 452,984,832 params

binary_log8:
  groups: 1,982
  params: 3,117,416,448
  bucket: moe_expert only

ternary40_log8:
  groups: 18,555
  params: 30,050,828,288
  moe_expert:                  18,207 groups, 28,641,853,440 params
  linear_attention_projection:    148 groups,  1,010,434,048 params
  full_attention:                  40 groups,    272,629,760 params
  shared_expert:                  160 groups,    125,911,040 params
```

Interpretation:

- Binary spillover is not the immediate teacher-cache failure. The selected
  binary expert slices are almost behavior-preserving on this gate.
- The ternary trunk alone reproduces the catastrophic full-policy failure.
- The next branch should target ternary allocation/training first:
  1. run ternary bucket ablations (`moe_expert`, attention projection,
     full attention, shared expert) on the corrected cache;
  2. protect or QAT the sensitive ternary buckets;
  3. only then reintroduce binary rate allocation for the bit target.
- This sharpens the mainline: do not discard binary globally; fix ternary
  support/code behavior and use binary only where behavior gates stay clean.

## Ternary Bucket Ablations

Ran corrected teacher-behavior cache ablations for every ternary bucket in the
measured24 policy. This identifies which ternary assignment actually breaks the
model.

Artifacts:

```text
full_attention:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_bucket_full_attention.json

linear_attention_projection:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_bucket_linear_attention_projection.json

shared_expert:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_bucket_shared_expert.json

moe_expert:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_ternary_bucket_moe_expert.json

protect linear attention projection candidate:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_candidate_protect_linear_attention_projection.json

protect all attention candidate:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_candidate_protect_all_attention.json

fp8 embedding/lm_head only:
  /workspace/agentworld_runs/teacher_behavior_eval_measured24_fp8_embedding_lm_head.json
```

Summary:

```text
bf16:
  KL 0.000092, top-k mass 0.999908, top1 1.000000

binary_only:
  KL 0.003479, top-k mass 0.999898, top1 0.989286

fp8_embed_head:
  KL 0.003261, top-k mass 0.999906, top1 0.985714

ternary_full_attention:
  KL 0.086130, top-k mass 0.999722, top1 0.946429

ternary_linear_attention_projection:
  KL 5.414197, top-k mass 0.487150, top1 0.332143

ternary_shared_expert:
  KL 0.074176, top-k mass 0.999963, top1 0.957143

ternary_moe_expert:
  KL 0.118171, top-k mass 0.994880, top1 0.946429

ternary_only:
  KL 11.057269, top-k mass 0.220561, top1 0.000000

protect_linear_attention_projection:
  KL 0.859504, top-k mass 0.943757, top1 0.714286

protect_all_attention:
  KL 0.292134, top-k mass 0.994187, top1 0.917857

full_measured24:
  KL 10.817585, top-k mass 0.149032, top1 0.000000
```

Interpretation:

- The catastrophic ternary failure is not broadly "MoE ternary." MoE-expert
  ternary alone is modest on this teacher-cache gate.
- `linear_attention_projection` is the main bad ternary bucket. It alone drops
  top1 to `0.332` and raises KL to `5.41`.
- Full attention and shared experts are mildly sensitive, but not catastrophic
  alone.
- FP8 embedding/lm_head and selected binary expert slices are clean.
- Protecting `linear_attention_projection` rescues the policy from catastrophic
  failure but is still too weak (`KL 0.86`, top1 `0.71`).
- Protecting all attention is much better (`KL 0.29`, top1 `0.918`) but still
  not BF16-clean enough to spend judge calls or run full AgentWorldBench.

Next branch:

1. Build a new rate policy that keeps all attention buckets BF16/FP8, keeps
   FP8 embedding/lm_head, keeps selected binary expert slices, and focuses
   ternary on MoE/shared expert weights.
2. Use the corrected teacher-cache gate before rollout:
   target should be much closer to binary-only / FP8-only behavior than to
   `protect_all_attention`.
3. If the protected-attention policy still has KL around `0.3`, run QAT or
   scale/support adaptation on MoE/shared ternary groups before AgentWorldBench
   judge spend.

Derived protected-attention policy:

```text
artifact:
  /workspace/agentworld_runs/policy_qwen_agentworld_measured24_protect_all_attention.json

changed:
  188 ternary attention groups -> BF16
  protected buckets: full_attention, linear_attention_projection

teacher-cache gate:
  source: /workspace/agentworld_runs/teacher_behavior_eval_measured24_candidate_protect_all_attention.json
  KL 0.292134, top-k mass 0.994187, top1 0.917857

projected compression:
  6.324x
  chosen_gib: 10.209
```

This policy is a behavior-improved diagnostic artifact, not the final 8x
candidate. Protecting all attention costs too many bits to hit the target
without additional rate allocation.

Next concrete allocator target:

- keep attention protected;
- keep FP8 embedding/lm_head, because that gate is clean;
- keep existing binary MoE slices, because that gate is clean;
- recover the 8x budget by selecting additional low-damage MoE/shared-expert
  groups for binary using the corrected teacher-cache gate or measured Fisher
  damage;
- only then run 256-token degeneration and OpenRouter judge scoring.

## Protected-Attention Bit Budget and Reallocation Probe

The protected-attention diagnostic is behaviorally useful but cannot reach
strict 8x with the current ternary/binary/FP8 menu.

Exact bit math:

```text
measured24:
  compression: 8.001x
  chosen_gib: 8.069

protect linear_attention_projection only:
  compression before extra binary: 6.619x
  extra bits needed for strict 8x: 14.466B
  remaining ternary params: 29.040B
  max extra binary saving: 17.424B
  strict 8x feasible by further binary conversion: yes

protect all attention:
  compression before extra binary: 6.324x
  extra bits needed for strict 8x: 18.372B
  remaining MoE/shared ternary params: 28.768B
  max extra binary saving: 17.261B
  strict 8x feasible by further binary conversion: no
  compression if all remaining MoE/shared ternary becomes binary: 7.874x
```

Derived strict-8x policy:

```text
artifact:
  /workspace/agentworld_runs/policy_qwen_agentworld_protect_linear_attn_rebinary8x.json

policy:
  linear_attention_projection -> BF16
  15,407 lowest existing-damage MoE/shared ternary groups -> binary
  final compression: 8.0001x
  chosen_gib: 8.0699
```

Corrected teacher-cache eval:

```text
artifact:
  /workspace/agentworld_runs/teacher_behavior_eval_policy_protect_linear_attn_rebinary8x.json

KL full mean:        4.54063
KL top-k mean:       4.11906
teacher top-k mass:  0.74885
top1 agreement:      0.41786
```

Conclusion: recovering 8x by mass-converting MoE/shared expert groups to binary
breaks the teacher-cache gate. The selected original binary expert slices are
safe, but broad binary expansion is not.

Also tested whether the sensitive linear-attention projection bucket prefers
binary over ternary:

```text
probe policy:
  /workspace/agentworld_runs/policy_probe_linear_attention_projection_binary_only.json

eval:
  /workspace/agentworld_runs/teacher_behavior_eval_policy_probe_linear_attention_projection_binary_only.json

binary linear_attention_projection:
  KL full mean:        5.97238
  KL top-k mean:       4.26327
  teacher top-k mass:  0.23093
  top1 agreement:      0.12143

ternary linear_attention_projection:
  KL full mean:        5.41420
  top1 agreement:      0.33214
```

Conclusion: linear-attention projection is not a good binary spillover target;
binary is worse than ternary. This bucket needs protection or QAT.

Updated strategy:

1. Keep `linear_attention_projection` protected for now.
2. Do not protect all attention if strict 8x is required; it is bit-infeasible
   under the current format.
3. The next real branch is QAT for `linear_attention_projection` and maybe
   `full_attention`, not more PTQ reallocation.
4. Use the corrected teacher-cache gate as the inner loop before any
   AgentWorldBench rollout or OpenRouter judge spend.

## Linear-Attention Projection Ternary QAT Branch

Added an activation-local QAT harness for the specific ternary/binary blocker:
`linear_attention_projection`. This avoids full 35B backprop on one A100. The
script collects real BF16 activations at selected projection modules, trains
hard-forward ternary or binary weights to preserve each module's linear map,
then writes a safetensors patch for full-model evaluation.

New files:

```text
experiments/qwen_agentworld/train_linear_projection_qat.py
experiments/qwen_agentworld/weight_patch.py
```

Also added `--weight-patch` support to:

```text
experiments/qwen_agentworld/eval_teacher_behavior_cache.py
experiments/qwen_agentworld/serve_transformers_openai.py
experiments/qwen_agentworld/serve_transformers_openai.sh
```

Smoke QAT pass:

```text
artifact:
  /workspace/agentworld_runs/qwen_projection_qat_smoke/linear_projection_ternary_patch.safetensors

scope:
  3 projection modules
  32 activation tokens/module
  2 QAT steps/module

corrected teacher-cache eval:
  /workspace/agentworld_runs/qwen_projection_qat_smoke/linear_projection_ternary_eval.json

KL full mean:        0.12008
KL top-k mean:       0.11998
teacher top-k mass:  0.99990
top1 agreement:      0.97143
```

Full bucket QAT pass v1:

```text
artifact:
  /workspace/agentworld_runs/qwen_projection_qat_full/linear_projection_ternary_patch.safetensors

scope:
  148 projection modules
  256 activation tokens/module
  20 QAT steps/module

local module fit:
  mean output relative RMSE: 0.41909 -> 0.24850
  improved modules: 148 / 148

corrected teacher-cache eval:
  /workspace/agentworld_runs/qwen_projection_qat_full/linear_projection_ternary_eval.json

KL full mean:        2.19874
KL top-k mean:       1.88345
teacher top-k mass:  0.78463
top1 agreement:      0.50714
```

This was a real improvement over raw ternary `linear_attention_projection`:

```text
raw ternary linear_attention_projection:
  KL full mean:        5.41420
  teacher top-k mass:  0.48715
  top1 agreement:      0.33214

QAT v1:
  KL full mean:        2.19874
  teacher top-k mass:  0.78463
  top1 agreement:      0.50714
```

Full bucket QAT pass v2:

```text
artifact:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/linear_projection_ternary_patch.safetensors

scope:
  148 projection modules
  512 activation tokens/module
  100 QAT steps/module
  lr: 2e-4
  codebook_weight: 5e-4

corrected teacher-cache eval:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/linear_projection_ternary_eval.json

KL full mean:        0.22455
KL top-k mean:       0.21115
teacher top-k mass:  0.99430
top1 agreement:      0.91071
```

Conclusion: ternary QAT largely rescues the `linear_attention_projection`
bucket. It is still not BF16, but it moves the bucket from catastrophic PTQ to
near-protected behavior. Binary remains a bad choice for this bucket unless a
separate binary-specific QAT branch proves otherwise.

Mixed-policy gates with QAT linears:

```text
QAT linears + measured policy without reapplying linear_attention_projection:
  eval:
    /workspace/agentworld_runs/qwen_projection_qat_full_v2/mixed_policy_without_linear_eval.json
  KL full mean:        2.06134
  KL top-k mean:       1.81926
  teacher top-k mass:  0.84428
  top1 agreement:      0.63214

QAT linears + protected full_attention + measured MoE/shared/FP8 policy:
  eval:
    /workspace/agentworld_runs/qwen_projection_qat_full_v2/mixed_policy_no_attention_eval.json
  KL full mean:        0.65839
  KL top-k mean:       0.61943
  teacher top-k mass:  0.96724
  top1 agreement:      0.81071
```

Derived strict-8x QAT-linear policy:

```text
artifact:
  /workspace/agentworld_runs/policy_qwen_agentworld_qatlinear_protect_fullattn_rebinary8x.json

policy:
  linear_attention_projection: ternary QAT patch, counted as ternary bits
  full_attention: BF16 protected
  extra binary backfill: 4,160 MoE/shared groups
  final compression: 8.00005x
  chosen_gib: 8.0700

by_rate:
  bf16 groups:          594
  fp8 groups:           2
  ternary40_log8 groups:14355
  binary_log8 groups:   6142
```

Corrected teacher-cache eval for strict-8x candidate:

```text
eval:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/strict8x_eval.json

KL full mean:        0.62492
KL top-k mean:       0.58739
teacher top-k mass:  0.96840
top1 agreement:      0.83929
```

Official AgentWorld 1x7 inference smoke, no judge:

```text
server:
  weight_patch applied: 148 / 148
  fake_quant policy applied: 20351 / 20351

predictions:
  /workspace/agentworld_runs/smoke_1x7_qatlinear_strict8x_no_think_64/predictions.jsonl

degeneracy:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/degeneracy_strict8x_no_think_64.json
  collapsed: 1 / 7
  failed_gate: false

BF16 comparison:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/compare_bf16_no_think_64_vs_strict8x_qatlinear.json
  avg_sequence_ratio: 0.21118
  exact_matches: 0 / 7

BF16 degeneracy reference:
  /workspace/agentworld_runs/qwen_projection_qat_full_v2/degeneracy_bf16_no_think_64_for_compare.json
  collapsed: 0 / 7
```

Verdict:

```text
QAT-linear strict 8x is the first strict binary/ternary candidate that does not
instantly fail the corrected teacher-cache gate, but it is not judge-ready.
The android smoke row still collapses into repeated <think>/ellipsis text, and
the cheap BF16 text-diff remains low.
```

Next actions:

1. Keep the v2 ternary `linear_attention_projection` patch as the current best
   projection artifact.
2. Do not run paid OpenRouter judge scoring on the strict-8x candidate yet.
3. Improve binary backfill selection over MoE/shared using corrected-cache
   group or block damage measured after applying the QAT-linear patch.
4. Consider QAT for the worst MoE/shared binary-backfill blocks, not broad
   mass binary conversion.
5. Cache or pack the patched/policy-applied tensors before more official
   inference; fake-quantizing 20k groups during every server load is slow.

## 2026-07-02 vLLM Dense Top-64 Cache And Smoke Handoff

Active A100 run:

```text
repo:
  /workspace/clark-quantize

cache run:
  /workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019

collector:
  experiments/qwen_agentworld/collect_text_teacher_cache_vllm.py

launcher:
  experiments/qwen_agentworld/build_dense_top64_tranche1_cache_vllm_onepass.sh

target:
  10,000 dense text/reasoning rows
  max_length=8192
  top_k=64
  teacher values stored as normalized logprobs in teacher_top_logprobs

source mix requested:
  Aya:        2,000 rows
  OpenR1:     4,000 rows
  FineWeb-2:  450 rows per selected language config
```

Key implementation details:

```text
vLLM path:
  uses prompt_logprobs with one appended probe token
  alignment: prompt_logprobs positions 1..L supervise original positions 0..L-1
  stores top_indices and teacher_top_logprobs with shape [tokens, 64]

safe batching:
  BATCH_SIZE=8
  MAX_BATCH_TOKENS=4096
  GPU_MEMORY_UTILIZATION=0.75
  SAVE_EVERY_ROWS=256

reason:
  earlier vLLM run OOMed in full-vocab prompt-logprob softmax when
  gpu_memory_utilization was 0.90 and batches were row-count only.
```

Current recovery tools:

```text
status:
  experiments/qwen_agentworld/status_vllm_cache_run.py

pipeline status:
  experiments/qwen_agentworld/status_qwen_vllm_pipeline.py

merge:
  experiments/qwen_agentworld/merge_teacher_cache_shards.py

continuation planning:
  experiments/qwen_agentworld/plan_vllm_cache_continuation.py

queued smoke:
  experiments/qwen_agentworld/run_vllm_cache_train_smoke.sh

handoff:
  experiments/qwen_agentworld/wait_vllm_cache_then_train_smoke.sh
```

Handoff contract:

```text
1. Wait for the cache-builder PID to exit.
2. Wait for collect_text_teacher_cache_vllm and VLLM::EngineCore to exit.
3. Wait for GPU memory to fall below 5000 MiB.
4. Refuse to train unless the final 10k .pt cache exists.
5. Validate cache format, row count, and top_indices/teacher_top_logprobs shape.
6. Start run_vllm_cache_train_smoke.sh only after validation passes.
7. If the builder exits without a final .pt cache, log continuation offsets
   from the latest .pt.partial.json and exit instead of training from partial.
```

Queued smoke intent:

```text
goal:
  validate that train_all2d_ternary_qat.py can consume the vLLM logprob cache
  through chunked top-k+other KD.

important trainer flags:
  --chunked-head-kd
  --topk-other-bucket
  --anti-weight 0
  --optimizer factored_adafactor
  --post-accumulate-step
  --explicit-layer-checkpointing
  --master-dtype float32
  --alpha-start 1
  --alpha-end 1
```

As of the last live check before this note:

```text
rows:          2365 / 10000
tokens:        2,528,466
partial saves: 9
skips:         0
errors:        0
task mix:      2000 Aya, 365 OpenR1

final cache:
  not present yet

training smoke:
  not started yet
```

Decision:

```text
Do not manually start training while vLLM owns the GPU.
Do not train from partial cache.
Let the handoff validate and launch the smoke automatically.
```

## 2026-07-02 Post-Smoke LR Sweep Handoff

Added a second, separate handoff layer after the final-cache smoke. This does
not modify the active cache builder or the active cache-to-smoke watcher.

New scripts:

```text
experiments/qwen_agentworld/run_vllm_cache_lr_sweep.sh
experiments/qwen_agentworld/wait_smoke_then_lr_sweep.sh
experiments/qwen_agentworld/status_qwen_vllm_pipeline.py
experiments/qwen_agentworld/summarize_lr_sweep.py
```

LR sweep defaults:

```text
cache:
  /workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt

lrs:
  2e-5 1e-4 5e-4

steps per LR:
  2000

training:
  hard alpha=1 from step 0
  all 2D ternary g128
  train embed/head/norms
  FP32 masters
  factored Adafactor
  post-accumulate per-parameter stepping
  explicit layer checkpointing
  chunked head KD
  top-k + other bucket
  no global grad clip

forensics:
  code_forensics_groups_per_module=64
  master_delta_sample_size=65536
  code_persistence_short_step=500
  code_persistence_long_step=1500
```

Remote watcher:

```text
pid:
  /workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/post_smoke_lr_sweep.pid

log:
  /workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/post_smoke_lr_sweep_latest.log

nohup:
  /workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/post_smoke_lr_sweep_nohup.log

out root:
  /workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_after_smoke_20260702_084438
```

Watcher contract:

```text
1. Poll the existing smoke handoff log at a 10-minute cadence.
2. Wait for "train_smoke_done out_dir=..." and require smoke report.json.
3. Wait for any active train_all2d_ternary_qat.py process to exit.
4. Wait for GPU memory below 5000 MiB.
5. Refuse to run if the final 10k cache is missing.
6. Run the LR sweep through run_vllm_cache_lr_sweep.sh.
```

Live launch receipt:

```text
post_smoke_watcher_started pid=262364
2026-07-02T08:44:38Z smoke_not_done sleeping_seconds=600
```

Use this for the next scheduled monitor tick:

```bash
ssh a100 'cd /workspace/clark-quantize && python3 experiments/qwen_agentworld/status_qwen_vllm_pipeline.py'
```

Use this after the LR sweep starts or finishes:

```bash
ssh a100 'cd /workspace/clark-quantize && python3 experiments/qwen_agentworld/summarize_lr_sweep.py /workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_after_smoke_20260702_084438'
```
