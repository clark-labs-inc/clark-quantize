# qwen35_global_t2_mirror_10k_v1

Date: 2026-06-30

## Purpose

This is the first real global all-2D QAT run for Qwen3.5-9B after stopping
micro-probes and local patch experiments.

Target:

- Qwen3.5-9B
- all 2D tensors ternary, group size 128
- embeddings and lm_head included
- attention and MLP included
- 1D norms continuous/trainable
- no binary
- no mixed-rate policy
- no BF16 rescue islands
- no LoRA/adapters
- target format: Q2_0-style ternary first, about 7.5x physical compression

The hypothesis is that behavior-preserving low-bit compression requires global
codebook-native co-adaptation of the residual stream, not local tensor repair.

## Preflight State

Teacher cache:

- path: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630`
- latest preflight rows: `14896`
- bad shards: `0`
- fixed10 gate rows loaded directly from
  `/workspace/AgentWorldBench_round8_fixed10_20260629`
- long-gate preflight no longer imports AgentWorld `eval.py`

The previous 100k-row threshold was intentionally lowered to `MIN_ROWS=10000`
because the user decided 10k rows is sufficient for Phase 1.

## Code Changes Made For Launch

Files:

- `experiments/qwen_agentworld/run_long_generation_gate.py`
- `experiments/qwen_agentworld/launch_real_mm_all2d_qat_after_cache.sh`
- `experiments/qwen_agentworld/train_all2d_ternary_qat.py`

Changes:

- long-generation gate now reads `*_test.jsonl` directly instead of importing
  the official AgentWorld eval module, avoiding the accidental `openai`
  dependency during preflight.
- launch script now defaults to:
  - `RUN_ROOT=/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1`
  - `MIN_ROWS=10000`
- trainer now uses bounded sampled codebook regularization:
  - `--code-modules-per-step` default `8`
  - `--code-groups-per-module` default `4096`
- trainer uses chunked prox for phase 2, avoiding full quantized copies of
  giant embedding/head tensors during optimizer steps.

Reason for the trainer change:

- first launch with the full all-module codebook loss OOMed after model load.
- with the stale cache builder stopped, the trainer still OOMed by allocating
  a full-model codebook regularizer.
- bounded sampling fixed the issue without dropping the global all-2D objective.

## Launch

Run root:

```text
/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1
```

Launcher:

```text
/workspace/clark-quantize/experiments/qwen_agentworld/launch_real_mm_all2d_qat_after_cache.sh
```

Phase 1 command parameters:

```text
--mode ternary
--group-size 128
--master-dtype bfloat16
--train-layers all
--train-embed
--train-head
--train-norms
--max-train-tokens 1200
--optimizer mirror_sgd
--lr 2e-6
--hidden-weight 0.0
--anti-weight 0.1
--code-weight 0.001
--grad-clip 0.5
--max-update-rms 1.0
--steps 5000
--alpha-start 0.0
--alpha-end 1.0
--prox-rho 0.0
```

Phase 2, if Phase 1 finishes:

```text
--steps 50000
--alpha-start 1.0
--alpha-end 1.0
--prox-rho 0.001
```

## Current Health Snapshot

Observed at remote time `2026-06-30T08:01:07+00:00`:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU memory: about `43.3GB / 80GB`
- GPU utilization: `100%`
- phase: Phase 1 soft landing
- latest parsed step at that time: `1630`
- alpha at step 1630: `0.325865`
- checkpoint saved at step `1000`

The run is therefore past the previous OOM point and actively training.

## Early Metrics

Parsed JSON training rows through step 1630:

```text
parsed log rows: 855
latest step:     1630
latest task:     cauldron/docvqa
latest loss:     4.1775
latest KL:       3.9326
latest top1:     0.3958
latest topk mass:0.4996
latest code:     0.2020
```

Last-100-row means:

```text
loss mean:        1.3787
KL mean:          1.3169
weighted KL mean: 1.3785
top1 mean:        0.7469
topk mass mean:   0.8482
code mean:        0.2032
```

Task-family means so far:

```text
aya:        n=282, KL mean=0.0521, mass mean=0.9870
openr1:     n=288, KL mean=0.0119, mass mean=0.99997
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966
cauldron:   n=168, KL mean=3.2400, mass mean=0.6449
```

Interpretation:

- text and reasoning rows are healthy under the soft landing.
- multimodal OpenR1 is harder but still has high teacher top-k mass.
- Cauldron chart/doc/image QA rows are the first real stressor.
- Cauldron high KL is partly explained by low teacher top-256 mass, so it is
  not by itself evidence of trainer collapse.

## Monitor Commands

Run status:

```bash
ssh a100 'RUN_ROOT=/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1; \
  date -Is; \
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
  ps -p $(cat "$RUN_ROOT/launcher.pid") -o pid,ppid,stat,etime,pcpu,pmem,cmd; \
  pgrep -af train_all2d_ternary_qat; \
  tail -60 "$RUN_ROOT/phase1_soft_landing/train.log"'
```

Metric summary:

```bash
ssh a100 'python3 - <<'"'"'PY'"'"'
import json, pathlib
p=pathlib.Path("/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/train.log")
rows=[]
for line in p.read_text(errors="ignore").splitlines():
    line=line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj=json.loads(line)
    except Exception:
        continue
    if "step" in obj and "kl" in obj:
        rows.append(obj)
print("rows", len(rows), "latest_step", rows[-1]["step"] if rows else None)
if rows:
    print("latest", json.dumps(rows[-1], indent=2))
    by={}
    for r in rows:
        by.setdefault(r.get("task","?"), []).append(r)
    for task, rs in sorted(by.items()):
        kl=[float(r["kl"]) for r in rs]
        mass=[float(r["mass"]) for r in rs]
        print(task, len(rs), sum(kl)/len(kl), sum(mass)/len(mass), rs[-1]["step"])
PY'
```

## Stop / Continue Rules

Continue while:

- trainer process is alive
- GPU utilization remains nonzero
- log steps advance
- no NaN/OOM
- codebook term remains finite

Do not judge fixed10 during alpha ramp as final model quality. The model is
still blended until alpha reaches 1.

At Phase 1 completion:

- verify full patch and latest patch exist
- inspect `report_phase1.json`
- confirm phase 2 starts
- if phase 2 OOMs in prox, reduce prox chunk size or defer prox; do not fall
  back to local patch training

After meaningful hard-alpha progress:

- run fixed10 generation gate
- inspect think-loop/newline/empty-generation taxonomy
- only then decide whether to introduce on-policy survival cache

## Current Conclusion

This run is the correct global experiment and is currently alive.

The key early finding is practical: full all-2D QAT fits on one A100 only if
auxiliary losses and prox are memory-bounded. The global hard-forward ternary
model itself is trainable under the state-light mirror optimizer.

## Monitor Update: 2026-06-30T08:04Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `44.2GB / 80GB`
- GPU utilization: `100%`
- phase: Phase 1 soft landing
- latest parsed step: `1700`
- latest alpha: about `0.34`
- phase 2 has not started yet

Latest summary through step 1700:

```text
parsed log rows: 862
latest step:     1700
last-100 KL:     mean 1.0917, min 0.0066, max 5.1032
last-100 top1:   mean 0.7683
last-100 mass:   mean 0.8763
last-100 code:   mean 0.2030
```

Task-family summary:

```text
aya:        n=286, KL mean=0.0551, mass mean=0.9868, top1 mean=0.9216
openr1:     n=288, KL mean=0.0119, mass mean=0.99997, top1 mean=0.9700
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Interpretation:

- the run is still advancing and not stalled.
- the high-loss region remains concentrated in Cauldron chart/doc/image QA.
- the cache has rotated back into lower-KL Aya rows after the Cauldron shard,
  so the earlier KL spike is not a monotonic runaway.
- continue Phase 1 unless NaN, OOM, or log-step advancement stops.

## Monitor Update: 2026-06-30T08:08Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `45.0GB / 80GB`
- GPU utilization: `100%`
- phase: Phase 1 soft landing
- latest parsed step: `1760`
- latest alpha: `0.3519`
- phase 2 has not started yet
- latest Phase 1 patch exists:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/latest_phase1_patch.safetensors`

Latest row:

```text
step:        1760
alpha:       0.3519
task:        aya
tokens:      389
loss:        0.1625
kl:          0.1615
weighted_kl: 0.1623
top1:        0.8125
mass:        0.9919
code:        0.2047
```

Window summaries:

```text
last100:
  loss mean 0.8442, min 0.0068, max 5.0014
  kl   mean 0.8128, min 0.0066, max 4.7651
  top1 mean 0.8010
  mass mean 0.9121
  code mean 0.2032

last250:
  loss mean 2.3520
  kl   mean 2.2360
  top1 mean 0.6381
  mass mean 0.7548
  code mean 0.2031

last500:
  loss mean 1.2547
  kl   mean 1.1870
  top1 mean 0.7898
  mass mean 0.8761
  code mean 0.2031
```

Task-family summary:

```text
aya:        n=292, KL mean=0.0583, mass mean=0.9868, top1 mean=0.9198
openr1:     n=288, KL mean=0.0119, mass mean=1.0000, top1 mean=0.9700
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Interpretation:

- the run is alive, using the GPU, and still advancing through Phase 1.
- the sampled codebook term remains finite and stable near `0.20`.
- OpenR1 and Aya remain low-KL under the current alpha ramp.
- multimodal OpenR1 remains acceptable.
- Cauldron chart/doc/image QA remains the current hard distribution and is
  responsible for the elevated rolling KL windows.
- continue Phase 1; do not make generation-quality decisions until alpha
  reaches `1.0` and the hard-ternary phase has meaningful steps.

Known deviation from the memo:

- the active trainer implements state-light group-normalized mirror SGD plus
  `max_update_rms` clipping, but it does not currently implement the memo's
  batch loss-adaptive LR formula.
- this is worth adding before a restart or next run, especially for the
  high-KL Cauldron rows, but it is not enough reason to stop a healthy Phase 1
  alpha ramp. The memo's stop rules say to stop during ramp only for NaN, OOM,
  tokenizer/template bugs, or an unevaluable generation path.

## Monitor Update: 2026-06-30T08:12Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `54.3GB / 80GB`
- GPU utilization: `100%`
- phase: Phase 1 soft landing
- latest parsed step: `1830`
- latest alpha: `0.3659`
- phase 2 has not started yet
- latest artifact timestamps still show only Phase 1 outputs:
  - `latest_phase1_patch.safetensors`
  - `latest_phase1_report.json`
  - `phase1_soft_landing/train.log`

Latest row:

```text
step:        1830
alpha:       0.3659
task:        aya
tokens:      79
loss:        0.3086
kl:          0.2991
weighted_kl: 0.3084
top1:        0.8101
mass:        0.9664
code:        0.2019
anti:        0.0000005
```

Window summaries:

```text
last100:
  loss mean 0.8522, min 0.0068, max 5.0014
  kl   mean 0.8209, min 0.0066, max 4.7651
  top1 mean 0.7944
  mass mean 0.9117
  code mean 0.2033

last250:
  loss mean 2.2385
  kl   mean 2.1366
  top1 mean 0.6456
  mass mean 0.7616
  code mean 0.2030

last500:
  loss mean 1.2573
  kl   mean 1.1894
  top1 mean 0.7875
  mass mean 0.8758
  code mean 0.2030
```

All parsed rows so far:

```text
rows:        875
loss mean:   0.7376
kl mean:     0.6945
top1 mean:   0.8542
mass mean:   0.9252
code mean:   0.2031
anti mean:   0.0006
```

Task-family summary:

```text
aya:        n=299, KL mean=0.0613, mass mean=0.9867, top1 mean=0.9172
openr1:     n=288, KL mean=0.0119, mass mean=1.0000, top1 mean=0.9700
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Recent last-250 task split:

```text
aya:        n=43,  KL mean=0.1452, mass mean=0.9846, top1 mean=0.8503
openr1:     n=32,  KL mean=0.0304, mass mean=1.0000, top1 mean=0.9479
mm_openr1:  n=13,  KL mean=0.1298, mass mean=0.9977, top1 mean=0.9071
cauldron:   n=162, KL mean=3.2422, mass mean=0.6364, top1 mean=0.5105
```

Interpretation:

- the run remains healthy and GPU-bound.
- Phase 1 is progressing normally; alpha is still only about `0.37`.
- the elevated rolling loss remains explained by the Cauldron chart/doc/image
  segment. The lower-KL text/reasoning/multimodal-reasoning families remain
  intact after the Cauldron block.
- codebook loss is stable around `0.203`, with no evidence of prox/code
  instability.
- continue Phase 1. The next meaningful gate is Phase 1 completion and Phase 2
  startup, not fixed10 generation quality during the alpha ramp.

## Monitor Update: 2026-06-30T08:15Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `44.4GB / 80GB`
- GPU utilization: `98%`
- phase: Phase 1 soft landing
- latest parsed step: `1860`
- latest alpha: `0.3719`
- phase 2 has not started yet

Latest row:

```text
step:        1860
alpha:       0.3719
task:        aya
tokens:      637
loss:        0.2840
kl:          0.2837
weighted_kl: 0.2838
top1:        0.8125
mass:        0.9883
code:        0.2020
anti:        0.000118
```

Window summaries:

```text
last100:
  loss mean 0.8574, min 0.0068, max 5.0014
  kl   mean 0.8264, min 0.0066, max 4.7651
  top1 mean 0.7929
  mass mean 0.9114
  code mean 0.2032

last250:
  loss mean 2.1902
  kl   mean 2.0921
  top1 mean 0.6491
  mass mean 0.7652
  code mean 0.2031

last500:
  loss mean 1.2590
  kl   mean 1.1911
  top1 mean 0.7863
  mass mean 0.8757
  code mean 0.2031
```

Task-family summary:

```text
aya:        n=302, KL mean=0.0635, mass mean=0.9865, top1 mean=0.9159
openr1:     n=288, KL mean=0.0119, mass mean=1.0000, top1 mean=0.9700
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Interpretation:

- the run remains alive and GPU-bound.
- Phase 1 is still in the alpha ramp; alpha is only about `0.37`.
- the high rolling KL remains attributable to the recent Cauldron segment, not
  a global collapse: Aya/OpenR1/mm_OpenR1 task-family metrics remain healthy.
- codebook loss remains stable around `0.203`.
- continue Phase 1. No stop condition is present.

## Monitor Update: 2026-06-30T08:17Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `38.4GB / 80GB`
- GPU utilization: `91%`
- phase: Phase 1 soft landing
- latest parsed step: `1900`
- latest alpha: `0.3799`
- phase 2 has not started yet

Latest row:

```text
step:        1900
alpha:       0.3799
task:        aya
tokens:      107
loss:        0.2587
kl:          0.2535
weighted_kl: 0.2584
top1:        0.8854
mass:        0.9840
code:        0.2066
anti:        0.000017
```

Window summaries:

```text
last100:
  loss mean 0.8603, min 0.0068, max 5.0014
  kl   mean 0.8295, min 0.0066, max 4.7651
  top1 mean 0.7914
  mass mean 0.9110
  code mean 0.2032

last250:
  loss mean 2.1419
  kl   mean 2.0478
  top1 mean 0.6532
  mass mean 0.7695
  code mean 0.2032

last500:
  loss mean 1.2606
  kl   mean 1.1928
  top1 mean 0.7850
  mass mean 0.8754
  code mean 0.2031
```

Task-family summary:

```text
aya:        n=306, KL mean=0.0657, mass mean=0.9864, top1 mean=0.9145
openr1:     n=288, KL mean=0.0119, mass mean=1.0000, top1 mean=0.9700
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Recent last-250 task split:

```text
aya:        n=50,  KL mean=0.1606, mass mean=0.9830, top1 mean=0.8433
openr1:     n=32,  KL mean=0.0304, mass mean=1.0000, top1 mean=0.9479
mm_openr1:  n=13,  KL mean=0.1298, mass mean=0.9977, top1 mean=0.9071
cauldron:   n=155, KL mean=3.2340, mass mean=0.6339, top1 mean=0.5098
```

Interpretation:

- the trainer is still alive and advancing; GPU utilization remains high.
- rolling loss is still dominated by the Cauldron segment, but the recent
  last-250 KL has drifted down from the prior sample as the run moves back into
  lower-KL Aya rows.
- no evidence of global collapse: Aya, OpenR1, and multimodal OpenR1 remain in
  acceptable ranges for an alpha-ramp run.
- codebook loss remains stable around `0.203`.
- continue Phase 1; no stop condition is present.

## Monitor Update: 2026-06-30T08:20Z

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `68.2GB / 80GB`
- GPU utilization: `100%`
- phase: Phase 1 soft landing
- latest parsed step: `1950`
- latest alpha: `0.3899`
- phase 2 has not started yet
- latest save artifacts are still the step-1000 Phase 1 latest patch/report;
  the next save checkpoint is expected around step `2000`.

Latest row:

```text
step:        1950
alpha:       0.3899
task:        openr1
tokens:      1024
loss:        0.0843
kl:          0.0841
weighted_kl: 0.0841
top1:        0.9375
mass:        0.9999
code:        0.2059
anti:        0.000000
```

Window summaries:

```text
last100:
  loss mean 0.8627, min 0.0068, max 5.0014
  kl   mean 0.8323, min 0.0066, max 4.7651
  top1 mean 0.7916
  mass mean 0.9113
  code mean 0.2032

last250:
  loss mean 2.0711
  kl   mean 1.9821
  top1 mean 0.6601
  mass mean 0.7752
  code mean 0.2031

last500:
  loss mean 1.2623
  kl   mean 1.1944
  top1 mean 0.7837
  mass mean 0.8754
  code mean 0.2031
```

Task-family summary:

```text
aya:        n=308, KL mean=0.0671, mass mean=0.9863, top1 mean=0.9138
openr1:     n=291, KL mean=0.0128, mass mean=1.0000, top1 mean=0.9693
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Recent last-250 task split:

```text
aya:        n=52,  KL mean=0.1650, mass mean=0.9829, top1 mean=0.8420
openr1:     n=35,  KL mean=0.0359, mass mean=0.9999, top1 mean=0.9440
mm_openr1:  n=13,  KL mean=0.1298, mass mean=0.9977, top1 mean=0.9071
cauldron:   n=150, KL mean=3.2266, mass mean=0.6315, top1 mean=0.5093
```

Interpretation:

- the run is still alive and GPU-bound.
- it has rotated into an OpenR1 shard; latest KL is low with near-perfect
  teacher top-k mass.
- the last-250 KL continues to decrease as Cauldron rows age out of the
  rolling window.
- no global collapse signal is present; codebook loss remains stable near
  `0.203`.
- continue Phase 1. Watch for the expected step-2000 latest patch/report save,
  then keep monitoring toward the alpha=1 handoff.

## Monitor Update: 2026-06-30T08:24Z

Step-2000 save gate:

- passed.
- `checkpoint_saved` is present in `phase1_soft_landing/train.log`.
- latest patch refreshed:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/latest_phase1_patch.safetensors`
- latest report refreshed:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/latest_phase1_report.json`
- refreshed artifact timestamp: `2026-06-30 08:23`
- refreshed patch size: about `17G`
- refreshed report size: about `386K`
- `latest_phase1_report.json` contains history through step `2000`.

Remote state:

- trainer PID: `234570`
- launcher PID: `234553`
- GPU memory: about `76.3GB / 80GB`
- GPU utilization: `96%`
- phase: Phase 1 soft landing
- latest parsed step after the checkpoint: `2010`
- latest alpha: `0.4019`
- phase 2 has not started yet

Checkpoint row at step 2000:

```text
step:        2000
alpha:       0.3999
task:        openr1
tokens:      1024
loss:        0.0979
kl:          0.0977
weighted_kl: 0.0977
top1:        0.8333
mass:        0.9999
code:        0.2090
anti:        0.000000
```

Latest parsed row after save:

```text
step:        2010
alpha:       0.4019
task:        openr1
tokens:      1024
loss:        0.1226
kl:          0.1224
weighted_kl: 0.1224
top1:        0.8542
mass:        0.9992
code:        0.2048
anti:        0.000000
```

Window summaries:

```text
last100:
  loss mean 0.8621, min 0.0068, max 5.0014
  kl   mean 0.8320, min 0.0066, max 4.7651
  top1 mean 0.7920
  mass mean 0.9116
  code mean 0.2032

last250:
  loss mean 1.9593
  kl   mean 1.8743
  top1 mean 0.6718
  mass mean 0.7862
  code mean 0.2032

last500:
  loss mean 1.2631
  kl   mean 1.1952
  top1 mean 0.7828
  mass mean 0.8754
  code mean 0.2031
```

Task-family summary:

```text
aya:        n=308, KL mean=0.0671, mass mean=0.9863, top1 mean=0.9138
openr1:     n=297, KL mean=0.0140, mass mean=1.0000, top1 mean=0.9680
mm_openr1:  n=117, KL mean=0.2481, mass mean=0.9966, top1 mean=0.9063
cauldron:   n=171, KL mean=3.2569, mass mean=0.6431, top1 mean=0.5134
```

Interpretation:

- the checkpoint path works under the live memory load; step-2000 latest patch
  and report refreshed successfully.
- the trainer continued after saving and reached at least step `2010`.
- OpenR1 remains very low-KL with near-perfect teacher top-k mass after the
  save gate.
- the last-250 KL continues falling as Cauldron rows age out of the rolling
  window.
- codebook loss remains stable near `0.203`.
- continue Phase 1 toward alpha `1.0`; next expected latest save is step
  `3000`.

## Generation Smoke: 2026-06-30T08:31Z

Artifact:

- `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/latest_phase1_patch.safetensors`
- step-2000 latest patch from Phase 1 soft landing.
- This is the hard alpha=1 projection saved from the current masters, not the
  live alpha-blended training state. The live training alpha at step 2000 was
  about `0.40`.

Method:

- CPU-only generation with `CUDA_VISIBLE_DEVICES=` so the live A100 trainer was
  not disturbed.
- `AutoModelForCausalLM`, class `Qwen3_5ForCausalLM`.
- Patch application succeeded: `354` tensors applied, `0` missing.
- Prompt: `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- `max_new_tokens=16`, deterministic generation.
- Output JSON:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_step2000_cpu_short/smoke_generation.json`

Generated text:

```text
 how how our.



 how you are are
```

Interpretation:

- The saved hard-ternary patch is not generation-coherent at step 2000.
- This does not necessarily refute the live Phase 1 training state, because the
  trainer is still using an alpha-blended forward pass while the saved patch is
  hard alpha=1.
- Treat this as an early warning that the hard projection is still broken during
  the ramp, not as a stop signal.
- Re-test after alpha reaches `1.0`, and again after meaningful Phase 2 hard
  ternary steps.

## Monitor Update: 2026-06-30T08:37Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `62.9GB / 81.9GB`, `96%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2210`

Latest row:

```text
step:   2210
alpha:  0.4419
task:   mm_openr1
tokens: 410
loss:   0.1545
kl:     0.1436
top1:   0.9063
mass:   0.9947
code:   0.2072
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 0.1075, kl 0.1030, top1 0.9177, mass 0.9975, code 0.2029

last100:
  loss 0.8763, kl 0.8456, top1 0.7854, mass 0.9113, code 0.2031

last250:
  loss 1.5521, kl 1.4800, top1 0.7162, mass 0.8314, code 0.2031

last500:
  loss 1.2669, kl 1.1989, top1 0.7807, mass 0.8753, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=52,  kl=0.1650, top1=0.8420, mass=0.9829
openr1:     n=156, kl=0.0244, top1=0.9568, mass=0.9999
mm_openr1:  n=121, kl=0.2491, top1=0.9049, mass=0.9963
cauldron:   n=171, kl=3.2569, top1=0.5134, mass=0.6431
```

Interpretation:

- The run is still progressing normally through Phase 1.
- OpenR1 and mm_OpenR1 remain healthy at the current alpha.
- Cauldron rows still dominate the long-window loss and are the hard shard.
- No new checkpoint boundary has appeared since step `2000`; next expected save
  is step `3000`.
- Continue training. Do not run another hard-artifact generation smoke until the
  step-3000 checkpoint or alpha `1.0`.

## Monitor Update: 2026-06-30T08:39Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `68.8GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2240`
- latest loaded shard: `mm_reason_shard_017_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   2240
alpha:  0.4479
task:   mm_openr1
tokens: 738
loss:   0.0411
kl:     0.0279
top1:   0.9583
mass:   0.9983
code:   0.2022
anti:   0.000403
```

Rolling windows:

```text
last20:
  loss 0.1369, kl 0.1297, top1 0.9073, mass 0.9964, code 0.2023

last100:
  loss 0.8837, kl 0.8525, top1 0.7829, mass 0.9111, code 0.2031

last250:
  loss 1.4910, kl 1.4206, top1 0.7222, mass 0.8384, code 0.2031

last500:
  loss 1.2685, kl 1.2004, top1 0.7798, mass 0.8752, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=52,  kl=0.1650, top1=0.8420, mass=0.9829
openr1:     n=153, kl=0.0247, top1=0.9566, mass=0.9999
mm_openr1:  n=124, kl=0.2495, top1=0.9031, mass=0.9962
cauldron:   n=171, kl=3.2569, top1=0.5134, mass=0.6431
```

Interpretation:

- No new checkpoint was available, so no generation smoke was run.
- The current mm_OpenR1 rows look strong at alpha `0.45`.
- Codebook loss remains stable near `0.203`.
- The run should continue toward step `3000`; evaluate the new hard artifact
  only after that checkpoint is written.

## Monitor Update: 2026-06-30T08:51Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `46.0GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2440`
- recent shard transition:
  - `mm_shard_000_text512_image512_len1024_last96_top256.pt` at step `2305`
  - `mm_reason_shard_001_text256_reason256_mm128_image384_len1024_last96_top256.pt` at step `2433`

Latest row:

```text
step:   2440
alpha:  0.4879
task:   mm_openr1
tokens: 558
loss:   0.2682
kl:     0.2457
top1:   0.8542
mass:   0.9968
code:   0.1950
anti:   0.000172
```

Rolling windows:

```text
last20:
  loss 1.2160, kl 1.0861, top1 0.7453, mass 0.8983, code 0.2031

last100:
  loss 1.1072, kl 1.0514, top1 0.7474, mass 0.8910, code 0.2031

last250:
  loss 1.3133, kl 1.2428, top1 0.7393, mass 0.8601, code 0.2030

last500:
  loss 1.3167, kl 1.2434, top1 0.7707, mass 0.8711, code 0.2030

last1000:
  loss 0.7221, kl 0.6788, top1 0.8531, mass 0.9276, code 0.2030
```

Task-family summary over the last 500 rows:

```text
aya:        n=60,  kl=0.1983, top1=0.8329, mass=0.9802
openr1:     n=133, kl=0.0267, top1=0.9540, mass=0.9999
mm_openr1:  n=131, kl=0.2456, top1=0.9021, mass=0.9962
cauldron:   n=176, kl=3.2619, top1=0.5131, mass=0.6435
```

Interpretation:

- The run remains healthy and GPU-bound.
- No new checkpoint was available, so no generation smoke was run.
- The row mix shifted back into multimodal/reasoning shards after the earlier
  OpenR1-heavy section, so the rolling loss rose. This is expected from the
  harder shard mix rather than an obvious instability.
- OpenR1 remains excellent, mm_OpenR1 remains usable, and Cauldron continues to
  be the main hard shard.
- Continue to step `3000`; next monitor should not run before roughly
  `09:01Z` unless a concrete issue appears.

## Monitor Update: 2026-06-30T09:01Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `42.2GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2610`
- latest loaded shard: `mm_reason_shard_002_text256_reason256_mm128_image384_len1024_last96_top256.pt`

Latest row:

```text
step:   2610
alpha:  0.5219
task:   cauldron
tokens: 787
loss:   5.5771
kl:     5.3747
top1:   0.3229
mass:   0.4837
code:   0.2018
anti:   0.005253
```

Rolling windows:

```text
last20:
  loss 2.2298, kl 2.1238, top1 0.6375, mass 0.7794, code 0.2037

last100:
  loss 0.9276, kl 0.8708, top1 0.7759, mass 0.9128, code 0.2032

last250:
  loss 1.4068, kl 1.3330, top1 0.7287, mass 0.8498, code 0.2032

last500:
  loss 1.3901, kl 1.3135, top1 0.7597, mass 0.8638, code 0.2031

last1000:
  loss 0.7480, kl 0.7036, top1 0.8494, mass 0.9250, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=60,  kl=0.1983, top1=0.8329, mass=0.9802
openr1:     n=116, kl=0.0288, top1=0.9522, mass=0.9999
mm_openr1:  n=140, kl=0.2615, top1=0.8975, mass=0.9940
cauldron:   n=184, kl=3.2874, top1=0.5096, mass=0.6409
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was available, so no generation smoke was run.
- The current latest row is Cauldron and is bad, but this matches the known hard
  shard pattern rather than a global collapse: OpenR1 remains excellent,
  mm_OpenR1 is still acceptable, and codebook loss is stable near `0.203`.
- Continue to step `3000`; next monitor should not run before roughly `09:11Z`
  unless a checkpoint lands or a concrete issue appears.

## Monitor Update: 2026-06-30T09:11Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `46.4GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2770`
- latest loaded shard: `mm_reason_shard_003_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   2770
alpha:  0.5539
task:   openr1
tokens: 1024
loss:   0.1256
kl:     0.1254
top1:   0.9167
mass:   0.9993
code:   0.1992
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 2.3116, kl 2.2398, top1 0.6391, mass 0.7674, code 0.2023

last100:
  loss 0.9689, kl 0.9172, top1 0.7810, mass 0.9104, code 0.2032

last250:
  loss 1.4137, kl 1.3422, top1 0.7308, mass 0.8500, code 0.2030

last500:
  loss 1.4461, kl 1.3684, top1 0.7512, mass 0.8581, code 0.2030

last1000:
  loss 0.7647, kl 0.7206, top1 0.8470, mass 0.9234, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=60,  kl=0.1983, top1=0.8329, mass=0.9802
openr1:     n=109, kl=0.0382, top1=0.9451, mass=0.9999
mm_openr1:  n=140, kl=0.2615, top1=0.8975, mass=0.9940
cauldron:   n=191, kl=3.3065, top1=0.5075, mass=0.6393
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was available, so no generation smoke was run.
- Alpha has crossed `0.55`; the hard projection is still not expected to be
  judged until the next saved artifact.
- OpenR1 and mm_OpenR1 are still healthy enough; Cauldron remains the hard shard
  and dominates recent rolling loss.
- Continue to step `3000`; next monitor should not run before roughly `09:21Z`
  unless the step-3000 checkpoint lands or a concrete issue appears.

## Monitor Update: 2026-06-30T09:21Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `54.7GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `2000`
- latest parsed training row: step `2930`
- latest loaded shard: `mm_reason_shard_004_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   2930
alpha:  0.5859
task:   mm_openr1
tokens: 504
loss:   0.0818
kl:     0.0777
top1:   0.9583
mass:   0.9985
code:   0.1945
anti:   0.000083
```

Rolling windows:

```text
last20:
  loss 0.1899, kl 0.1746, top1 0.8708, mass 0.9971, code 0.2038

last100:
  loss 0.9648, kl 0.9116, top1 0.7898, mass 0.9134, code 0.2031

last250:
  loss 1.3482, kl 1.2787, top1 0.7391, mass 0.8592, code 0.2031

last500:
  loss 1.4525, kl 1.3742, top1 0.7476, mass 0.8580, code 0.2031

last1000:
  loss 0.7557, kl 0.7120, top1 0.8473, mass 0.9246, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=60,  kl=0.1983, top1=0.8329, mass=0.9802
openr1:     n=97,  kl=0.0485, top1=0.9375, mass=0.9999
mm_openr1:  n=152, kl=0.2562, top1=0.8946, mass=0.9941
cauldron:   n=191, kl=3.3065, top1=0.5075, mass=0.6393
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was visible at this monitor point, so no generation smoke
  was run.
- The latest row is strong mm_OpenR1 at alpha `0.586`.
- The run is close to the next checkpoint boundary; evaluate the new hard
  artifact after the step-3000 save appears.
- Continue; next monitor should not run before roughly `09:31Z` unless a
  checkpoint-specific check is requested.

## Monitor + Generation Smoke: 2026-06-30T09:31Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `66.5GB / 81.9GB`, `72%` utilization at the sampling
  instant
- latest checkpoint: step `3000`
- latest parsed training row: step `3090`
- latest loaded shard: `mm_reason_shard_006_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3090
alpha:  0.6179
task:   aya
tokens: 133
loss:   0.4901
kl:     0.4888
top1:   0.6354
mass:   0.9585
code:   0.2061
anti:   0.002619
```

Rolling windows:

```text
last20:
  loss 1.4722, kl 1.4279, top1 0.7010, mass 0.8588, code 0.2029

last100:
  loss 1.2422, kl 1.1806, top1 0.7485, mass 0.8853, code 0.2032

last250:
  loss 1.2096, kl 1.1525, top1 0.7533, mass 0.8767, code 0.2031

last500:
  loss 1.5100, kl 1.4300, top1 0.7377, mass 0.8524, code 0.2031

last1000:
  loss 0.7733, kl 0.7293, top1 0.8440, mass 0.9229, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=62,  kl=0.2146, top1=0.8248, mass=0.9800
openr1:     n=81,  kl=0.0557, top1=0.9324, mass=0.9999
mm_openr1:  n=161, kl=0.2568, top1=0.8909, mass=0.9942
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Generation smoke:

- Artifact:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/latest_phase1_patch.safetensors`
- Report step: `3000`
- Report alpha: `0.5999`
- Output JSON:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_step3000_cpu_short_valid/smoke_generation.json`
- CPU-only generation with `CUDA_VISIBLE_DEVICES=`.
- Prompt: `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- `max_new_tokens=16`, deterministic.
- Patch key mapping required: `model.language_model.* -> model.*`
- Valid patch application: `354` tensors applied, `0` missing, `0` shape mismatches.

Generated text:

```text
 can you have our of what are still. have the what still still our have
```

Interpretation:

- The step-3000 hard-ternary artifact is still not generation-coherent.
- This is stronger evidence than the step-2000 smoke because all patch tensors
  applied correctly and the checkpoint alpha is now `0.60`.
- It is still a Phase 1 alpha-ramp checkpoint; live training has not reached
  alpha `1.0`, so this is an early hard-projection warning rather than a stop
  signal.
- Keep training to at least alpha `1.0`; retest the hard artifact after the
  step-5000/Phase-1 completion checkpoint and then again after Phase 2 hard
  ternary steps.
- Loader note: one invalid smoke attempt applied only `1/354` tensors because
  the in-memory model wrapper uses `model.*` keys while the saved patch uses
  `model.language_model.*`; ignore that invalid output.

## Monitor Update: 2026-06-30T09:41Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `76.2GB / 81.9GB`, `97%` utilization
- latest checkpoint still: step `3000`
- latest parsed training row: step `3260`
- latest loaded shard: `mm_reason_shard_007_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3260
alpha:  0.6519
task:   openr1
tokens: 1024
loss:   0.1666
kl:     0.1664
top1:   0.9271
mass:   0.9994
code:   0.2047
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 1.0422, kl 1.0250, top1 0.6193, mass 0.9202, code 0.2021

last100:
  loss 1.3490, kl 1.2864, top1 0.7052, mass 0.8767, code 0.2031

last250:
  loss 0.9096, kl 0.8704, top1 0.7769, mass 0.9137, code 0.2031

last500:
  loss 1.5342, kl 1.4546, top1 0.7277, mass 0.8506, code 0.2031

last1000:
  loss 0.7862, kl 0.7420, top1 0.8387, mass 0.9221, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=76,  kl=0.3448, top1=0.7802, mass=0.9713
openr1:     n=73,  kl=0.0662, top1=0.9272, mass=0.9998
mm_openr1:  n=155, kl=0.2608, top1=0.8910, mass=0.9941
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was available beyond step `3000`, so no generation smoke was
  run.
- Alpha is now `0.65`; hard-projection quality remains deferred until the next
  saved artifact or Phase-1 completion at alpha `1.0`.
- OpenR1 and mm_OpenR1 remain stable; Cauldron is still the dominant hard shard.
- Continue; next monitor should not run before roughly `09:51Z` unless a
  checkpoint-specific check is requested.

## Monitor Update: 2026-06-30T09:51Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `42.3GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `3000`
- latest parsed training row: step `3430`
- latest loaded shard: `mm_reason_shard_008_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3430
alpha:  0.6859
task:   openr1
tokens: 1024
loss:   0.1096
kl:     0.1094
top1:   0.9063
mass:   0.9996
code:   0.1971
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 0.4576, kl 0.4444, top1 0.7849, mass 0.9731, code 0.2035

last100:
  loss 1.2008, kl 1.1606, top1 0.7117, mass 0.8915, code 0.2031

last250:
  loss 0.9362, kl 0.8970, top1 0.7697, mass 0.9127, code 0.2031

last500:
  loss 1.5454, kl 1.4662, top1 0.7229, mass 0.8496, code 0.2031

last1000:
  loss 0.7943, kl 0.7499, top1 0.8357, mass 0.9218, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=85,  kl=0.3879, top1=0.7691, mass=0.9682
openr1:     n=81,  kl=0.0801, top1=0.9208, mass=0.9997
mm_openr1:  n=138, kl=0.2743, top1=0.8896, mass=0.9937
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was available beyond step `3000`, so no generation smoke was
  run.
- Alpha is now `0.686`; hard projection is still known-broken at step `3000`,
  so the next meaningful smoke remains the next saved artifact or Phase-1
  completion at alpha `1.0`.
- OpenR1 remains very strong, mm_OpenR1 remains stable, and Cauldron remains the
  main hard shard.
- Continue; next monitor should not run before roughly `10:01Z` unless a
  checkpoint-specific check is requested.

## Monitor Update: 2026-06-30T10:01Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `76.3GB / 81.9GB`, `99%` utilization
- latest checkpoint still: step `3000`
- latest parsed training row: step `3580`
- latest loaded shard: `mm_reason_shard_010_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3580
alpha:  0.7159
task:   openr1
tokens: 1024
loss:   0.1445
kl:     0.1428
top1:   0.9063
mass:   0.9983
code:   0.1997
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 0.3777, kl 0.3725, top1 0.8044, mass 0.9876, code 0.2035

last100:
  loss 1.0187, kl 0.9883, top1 0.7302, mass 0.9130, code 0.2031

last250:
  loss 0.9491, kl 0.9099, top1 0.7653, mass 0.9126, code 0.2032

last500:
  loss 1.5171, kl 1.4420, top1 0.7218, mass 0.8497, code 0.2032

last1000:
  loss 0.7978, kl 0.7536, top1 0.8344, mass 0.9219, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=86,  kl=0.3909, top1=0.7685, mass=0.9677
openr1:     n=95,  kl=0.1064, top1=0.9098, mass=0.9995
mm_openr1:  n=123, kl=0.1746, top1=0.8914, mass=0.9941
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Interpretation:

- The run remains alive and GPU-bound.
- No new checkpoint was available beyond step `3000`, so no generation smoke was
  run.
- Alpha is now `0.716`. The step-3000 hard artifact is known-broken, so the next
  meaningful smoke remains the next saved artifact or Phase-1 completion.
- OpenR1 remains strong, mm_OpenR1 improved in the latest 500-row task summary,
  and Cauldron remains the dominant hard shard.
- Continue; next monitor should not run before roughly `10:11Z` unless a
  checkpoint-specific check is requested.

## Monitor Update: 2026-06-30T10:12Z

Monitor note:

- The scheduled `10:11Z` SSH command initially failed during banner exchange
  with `Operation timed out`.
- One short retry succeeded immediately, so this was treated as a transient SSH
  handshake issue rather than an instance failure.

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `76.3GB / 81.9GB`, `99%` utilization on retry
- latest checkpoint still: step `3000`
- latest parsed training row: step `3760`
- latest loaded shard: `mm_reason_shard_011_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3760
alpha:  0.7520
task:   openr1
tokens: 1024
loss:   0.0827
kl:     0.0825
top1:   0.9583
mass:   0.9987
code:   0.2044
anti:   0.000000
```

Rolling windows:

```text
last20:
  loss 0.2626, kl 0.2619, top1 0.8495, mass 0.9982, code 0.2028

last100:
  loss 0.6400, kl 0.6218, top1 0.7678, mass 0.9553, code 0.2032

last250:
  loss 0.9666, kl 0.9274, top1 0.7575, mass 0.9124, code 0.2032

last500:
  loss 1.5219, kl 1.4476, top1 0.7197, mass 0.8497, code 0.2032

last1000:
  loss 0.8015, kl 0.7576, top1 0.8327, mass 0.9221, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=86,  kl=0.3909, top1=0.7685, mass=0.9677
openr1:     n=113, kl=0.1325, top1=0.8994, mass=0.9993
mm_openr1:  n=105, kl=0.1846, top1=0.8894, mass=0.9935
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Interpretation:

- The run remains alive and GPU-bound after the transient SSH timeout.
- No new checkpoint was available beyond step `3000`, so no generation smoke was
  run.
- Alpha is now `0.752`; the next hard-artifact smoke should wait for the next
  saved checkpoint or Phase-1 completion.
- Recent short-window metrics improved; OpenR1 remains strong, mm_OpenR1 remains
  acceptable, and Cauldron remains the hard shard.
- Continue; next monitor should not run before roughly `10:22Z` unless a
  checkpoint-specific check is requested.

## Monitor Update: 2026-06-30T10:16Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `45.9GB / 81.9GB`, `100%` utilization
- latest checkpoint still: step `3000`
- latest parsed training row: step `3820`
- latest loaded shard:
  `mm_reason_shard_011_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   3820
alpha:  0.7640
task:   openr1
id:     openr1_math:3051
tokens: 1024
loss:   0.4032
kl:     0.4030
top1:   0.7396
mass:   0.9963
code:   0.2019
anti:   0.000059
```

Rolling windows:

```text
last20:
  loss 0.2473, kl 0.2470, top1 0.8646, mass 0.9985, code 0.2033

last100:
  loss 0.6352, kl 0.6182, top1 0.7706, mass 0.9555, code 0.2032

last250:
  loss 0.9707, kl 0.9315, top1 0.7562, mass 0.9124, code 0.2032

last500:
  loss 1.5228, kl 1.4487, top1 0.7195, mass 0.8497, code 0.2032

last1000:
  loss 0.8025, kl 0.7587, top1 0.8324, mass 0.9222, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=86,  kl=0.3909, top1=0.7685, mass=0.9677
openr1:     n=119, kl=0.1362, top1=0.8988, mass=0.9993
mm_openr1:  n=99,  kl=0.1893, top1=0.8883, mass=0.9932
cauldron:   n=196, kl=3.3460, top1=0.5039, mass=0.6347
```

Interpretation:

- Training remains alive and GPU-bound.
- No new checkpoint was available beyond step `3000`, so no generation smoke was
  run.
- Alpha is now `0.764`; continue Phase 1 and defer the next smoke until a new
  checkpoint, with step `5000` / alpha `1.0` still the key Phase-1 gate.
- Cauldron remains the outlier hard shard; OpenR1 and mm_OpenR1 remain much
  healthier.
- User requested fewer checks; after this note, use a wider `20-30` minute
  cadence and avoid filler status messages.

## Monitor Update: 2026-06-30T10:40Z

Live state:

- launcher PID: `234553`
- trainer PID: `234570`
- GPU: A100 80GB, about `60.3GB / 81.9GB`, `100%` utilization
- latest checkpoint: step `4000`
- latest parsed training row: step `4220`
- latest loaded shard:
  `mm_reason_shard_014_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest row:

```text
step:   4220
alpha:  0.8440
task:   aya
id:     aya:3643
tokens: 46
loss:   1.5579
kl:     1.5415
top1:   0.5217
mass:   0.7540
code:   0.1903
anti:   0.008048
```

Rolling windows:

```text
last20:
  loss 1.2746, kl 1.1783, top1 0.7115, mass 0.8887, code 0.2023

last100:
  loss 0.7804, kl 0.7414, top1 0.7804, mass 0.9377, code 0.2031

last250:
  loss 0.8827, kl 0.8406, top1 0.7688, mass 0.9240, code 0.2032

last500:
  loss 1.6274, kl 1.5478, top1 0.7030, mass 0.8390, code 0.2031

last1000:
  loss 0.8582, kl 0.8111, top1 0.8234, mass 0.9173, code 0.2031
```

Task-family summary over the last 500 rows:

```text
aya:        n=92,  kl=0.4580, top1=0.7579, mass=0.9579
openr1:     n=121, kl=0.1377, top1=0.8985, mass=0.9993
mm_openr1:  n=71,  kl=0.2484, top1=0.8621, mass=0.9907
cauldron:   n=216, kl=3.2290, top1=0.5178, mass=0.6486
```

Valid CPU smoke for step `4000`:

- output path:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_step4000_cpu_short_valid/smoke_generation.json`
- patch report step: `4000`
- patch report alpha: `0.8000`
- applied tensors: `354 / 354`
- missing tensors: `0`
- shape mismatches: `0`
- prompt:
  `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- generated tokens: `1`
- generation:

```text
<|im_end|>
```

Interpretation:

- The checkpoint load is valid, unlike the earlier invalid step-3000 smoke that
  only applied one tensor.
- The step-4000 hard artifact is still not generatively useful: it terminates
  immediately instead of producing coherent text.
- This is still inside Phase 1 (`alpha ~= 0.8`), so the next decisive gate is
  Phase-1 completion at step `5000` / alpha `1.0`.
- Continue training; do not over-poll. Next useful check is around the wider
  cadence window or at the step-5000 checkpoint.

## Monitor Update: 2026-06-30T11:28Z

Phase-1 completion state:

- launcher PID: `234553`
- Phase-1 checkpoint reached: step `5000`
- Phase-1 alpha: `1.0`
- Phase-1 final patch:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/qwen35_all2d_t2_phase1_patch.safetensors`
- Phase-1 final full patch:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/qwen35_all2d_t2_phase1_full_patch.safetensors`
- Phase-1 report:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase1_soft_landing/report_phase1.json`

Latest Phase-1 row:

```text
step:   5000
alpha:  1.0
task:   openr1
id:     openr1_math:957
tokens: 1024
loss:   1.3122
kl:     1.2534
top1:   0.5938
mass:   0.9658
code:   0.2109
anti:   0.000022
```

Phase-1 final rolling windows:

```text
last20:
  loss 2.3650, kl 2.2656, top1 0.4202, mass 0.8114, code 0.2029

last100:
  loss 2.1144, kl 2.0225, top1 0.5214, mass 0.8106, code 0.2033

last250:
  loss 1.4641, kl 1.4066, top1 0.6545, mass 0.8704, code 0.2032

last500:
  loss 1.3785, kl 1.3150, top1 0.6979, mass 0.8663, code 0.2032

last1000:
  loss 1.0329, kl 0.9798, top1 0.7882, mass 0.9023, code 0.2031
```

Phase-1 final task-family summary over the last 500 rows:

```text
aya:        n=133, kl=0.9959, top1=0.6440, mass=0.9046
openr1:     n=147, kl=0.2742, top1=0.8617, mass=0.9953
mm_openr1:  n=69,  kl=0.2547, top1=0.8595, mass=0.9905
cauldron:   n=151, kl=3.0938, top1=0.5121, mass=0.6504
```

Valid CPU smoke for step `5000`:

- output path:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_step5000_cpu_short_valid/smoke_generation.json`
- patch report step: `5000`
- patch report alpha: `1.0`
- applied tensors: `354 / 354`
- missing tensors: `0`
- shape mismatches: `0`
- prompt:
  `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- generated tokens: `16`
- generation:

```text






 Qat is a core function that determines the best matching between two
```

Interpretation:

- Phase 1 completed cleanly and produced both the compact patch and full patch.
- The valid hard-alpha smoke improved relative to step `4000` in the narrow
  sense that it no longer immediately emits `<|im_end|>`.
- It is still not a coherent generation checkpoint. The output has leading
  blank-line collapse and a generic/incomplete sentence.
- This confirms that Phase 1 alone is insufficient. The important test now is
  whether Phase 2 hard-ternary training can recover generation quality after
  alpha is fully hardened.

Phase-2 handoff:

- Phase-2 directory was created:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/phase2_hard_ternary`
- Phase-2 trainer PID: `238781`
- Phase-2 command uses:
  `--steps 50000`, `--alpha-start 1.0`, `--alpha-end 1.0`,
  `--prox-rho 0.001`,
  `--init-weight-patch .../phase1_soft_landing/qwen35_all2d_t2_phase1_full_patch.safetensors`
- GPU utilization was temporarily `0%` immediately after handoff, consistent
  with model load/startup; the Phase-2 process was alive.

Next action:

- Continue with reduced polling cadence.
- First meaningful Phase-2 check should verify that GPU utilization resumes and
  Phase-2 rows start appearing.
- The next generation smoke should wait for a real Phase-2 checkpoint rather
  than running repeatedly on the Phase-1 artifact.

## Monitor Update: 2026-06-30T12:00Z

Phase-2 live state:

- launcher PID: `234553`
- Phase-2 trainer PID: `238781`
- GPU: A100 80GB, about `74.3GB / 81.9GB`, `100%` utilization
- Phase-2 checkpoint: none yet
- Phase-2 latest parsed training row: step `313`
- latest loaded Phase-2 shard:
  `mm_reason_shard_002_text256_reason256_mm128_image384_len1024_last96_top256.pt`

Latest Phase-2 row:

```text
step:   313
alpha:  1.0
task:   openr1
id:     openr1_math:568
tokens: 1024
loss:   1.5445
kl:     1.5445
top1:   0.4896
mass:   0.9026
code:   0.000008
anti:   0.000564
```

Phase-2 rolling windows:

```text
last20:
  loss 1.0736, kl 1.0674, top1 0.6750, mass 0.9576, code 0.0000

last100:
  loss 1.9195, kl 1.8635, top1 0.5127, mass 0.8316, code 0.0000

last250:
  loss 2.6322, kl 2.5475, top1 0.3892, mass 0.7311, code 0.0000

last500/current-all:
  loss 2.6538, kl 2.5700, top1 0.3815, mass 0.7255, code 0.0000
```

Phase-2 task-family summary over current rows:

```text
aya:    n=256, kl=2.8816, top1=0.3228, mass=0.6743
openr1: n=57,  kl=1.1707, top1=0.6455, mass=0.9554
```

Interpretation:

- Phase 2 is now genuinely running and GPU-bound; the earlier post-handoff idle
  period was only startup/model-load.
- No Phase-2 checkpoint is available yet, so no Phase-2 generation smoke was
  run.
- Early Phase-2 metrics are worse than the Phase-1 final average, which is not
  surprising immediately after hard-alpha handoff and prox activation, but it
  needs to improve over the first few checkpoints.
- Code distance is near zero because Phase 2 initializes from the hardened
  Phase-1 full patch and uses hard ternary throughout.
- OpenR1 is much healthier than Aya in the first Phase-2 sample; Cauldron has
  not re-entered the Phase-2 window yet.

Next action:

- Continue reduced cadence.
- Next meaningful monitor is the first Phase-2 checkpoint, expected at step
  `1000`, or a wider status window if checkpointing takes longer.

## Monitor Update: 2026-06-30T13:12Z

Phase-2 live state:

- launcher PID: `234553`
- Phase-2 trainer PID: `238781`
- GPU: A100 80GB, about `78.1GB / 81.9GB`, `99%` utilization
- Phase-2 checkpoint: step `1000`
- Phase-2 latest parsed training row: step `1020`
- latest loaded Phase-2 shard:
  `mm_reason_shard_007_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest Phase-2 row:

```text
step:   1020
alpha:  1.0
task:   openr1
id:     openr1_math:1810
tokens: 1024
loss:   0.9310
kl:     0.9128
top1:   0.7500
mass:   0.9671
code:   0.000103
anti:   0.000045
```

Phase-2 rolling windows:

```text
last20:
  loss 2.7764, kl 2.6812, top1 0.3857, mass 0.7172, code 0.0001

last100:
  loss 3.7657, kl 3.5503, top1 0.3807, mass 0.6363, code 0.0001

last250:
  loss 3.5136, kl 3.2657, top1 0.4342, mass 0.6916, code 0.0000

last500:
  loss 2.4596, kl 2.3044, top1 0.5222, mass 0.8197, code 0.0000

last1000:
  loss 2.5695, kl 2.4389, top1 0.4630, mass 0.7792, code 0.0000
```

Phase-2 task-family summary over the last 500 rows:

```text
aya:        n=24,  kl=2.7785, top1=0.3735, mass=0.6949
openr1:     n=220, kl=1.3388, top1=0.6117, mass=0.9523
mm_openr1:  n=104, kl=1.3198, top1=0.6104, mass=0.9267
cauldron:   n=152, kl=4.3009, top1=0.3559, mass=0.5742
```

Valid CPU smoke for Phase-2 step `1000`:

- output path:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_phase2_step1000_cpu_short_valid/smoke_generation.json`
- patch report step: `1000`
- patch report alpha: `1.0`
- applied tensors: `354 / 354`
- missing tensors: `0`
- shape mismatches: `0`
- prompt:
  `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- generated tokens: `4`
- generation:

```text






<|im_end|>
```

Interpretation:

- The Phase-2 step-1000 patch loads correctly.
- Generation is still negative: the model emits blank lines and then stops.
- This is worse than the Phase-1 step-5000 smoke, which at least emitted a
  partial sentence fragment.
- Metrics remain uneven: OpenR1/mm_OpenR1 are better than Cauldron, while
  Cauldron is still the hardest shard and dominates the bad rolling windows.
- This is still early in a `50000`-step hard-ternary run, so do not conclude
  final failure yet. The important question is whether later hard-ternary steps
  reverse the blank-line/EOS behavior.

Next action:

- Continue the run.
- Avoid frequent polling.
- Next useful smoke/checkpoint gate is Phase-2 step `2000`, unless the run
  crashes or metrics become obviously unstable before then.

## Monitor Update: 2026-06-30T14:55Z

Phase-2 live state:

- launcher PID: `234553`
- Phase-2 trainer PID: `238781`
- GPU: A100 80GB, about `65.6GB / 81.9GB`, `100%` utilization
- Phase-2 checkpoint: step `2000`
- Phase-2 latest parsed training row: step `2030`
- latest loaded Phase-2 shard:
  `mm_reason_shard_015_text256_reason256_mm128_image192_len1024_last96_top256.pt`

Latest Phase-2 row:

```text
step:   2030
alpha:  1.0
task:   openr1
id:     openr1_math:3949
tokens: 1024
loss:   0.9918
kl:     0.9918
top1:   0.6354
mass:   0.9792
code:   0.000154
anti:   0.000000
```

Phase-2 rolling windows:

```text
last20:
  loss 1.9035, kl 1.8828, top1 0.5092, mass 0.8517, code 0.0002

last100:
  loss 2.3258, kl 2.2633, top1 0.5075, mass 0.8151, code 0.0001

last250:
  loss 3.4138, kl 3.2674, top1 0.4249, mass 0.6879, code 0.0001

last500:
  loss 2.6519, kl 2.4885, top1 0.5009, mass 0.7920, code 0.0000

last1000:
  loss 2.5421, kl 2.4191, top1 0.4680, mass 0.7832, code 0.0000
```

Phase-2 task-family summary over the last 500 rows:

```text
aya:        n=52,  kl=2.6233, top1=0.3866, mass=0.7237
openr1:     n=160, kl=1.3187, top1=0.6108, mass=0.9542
mm_openr1:  n=117, kl=1.3003, top1=0.6160, mass=0.9291
cauldron:   n=171, kl=4.3550, top1=0.3542, mass=0.5672
```

Valid CPU smoke for Phase-2 step `2000`:

- output path:
  `/workspace/agentworld_runs/qwen35_global_t2_mirror_10k_v1/generation_smoke_phase2_step2000_cpu_short_valid/smoke_generation.json`
- patch report step: `2000`
- patch report alpha: `1.0`
- applied tensors: `354 / 354`
- missing tensors: `0`
- shape mismatches: `0`
- prompt:
  `Write one coherent sentence explaining why global ternary QAT can beat local MLP patching.`
- generated tokens: `16`
- generation:

```text




<think>

</think>

**No**

**No**

**No
```

Interpretation:

- The Phase-2 step-2000 patch loads correctly.
- Generation remains pathological: blank lines, explicit think tags, and
  repeated `No` tokens.
- Compared with step `1000`, the model no longer immediately emits EOS, but it
  has not recovered coherent language. The failure has shifted from
  blank-line/EOS to blank-line/format-loop/repetition.
- Metrics have not shown a breakthrough. OpenR1/mm_OpenR1 remain much healthier
  than Cauldron, while Cauldron is still the dominant hard shard.
- This is still early relative to a `50000`-step hard-ternary run, but the first
  two Phase-2 smoke gates are negative.

Next action:

- Continue the run but do not poll frequently.
- Next useful smoke/checkpoint gate is Phase-2 step `3000` or later.
- If step `3000` remains a format/repetition collapse, consider whether to let
  the long-run/grokking hypothesis continue to at least `5000` hard steps before
  making any stop/restart call.

## Stop Update: 2026-06-30T15:58Z

User requested:

```text
stop the train
```

Actions:

- Cancelled the local scheduled step-3000 monitor/smoke timer.
- Sent termination to the remote launcher/trainer processes.
- Verified no remaining matching processes:
  `train_all2d_ternary_qat` / `launch_real_mm_all2d`.
- Verified GPU is idle:
  `0 MiB / 81920 MiB`, `0%` utilization.

Final remote process state:

```text
PROCS: none
GPU:   0, NVIDIA A100-SXM4-80GB, 0 MiB, 81920 MiB, 0 %
```

Final Phase-2 training state:

- final parsed row: step `2660`
- latest saved Phase-2 checkpoint: step `2000`
- no step-3000 checkpoint was reached
- latest patch/report timestamps remained at the step-2000 save
- final loaded shard:
  `mm_reason_shard_002_text256_reason256_mm128_image384_len1024_last96_top256.pt`

Final latest row:

```text
step:   2660
alpha:  1.0
task:   cauldron
id:     cauldron:ocrvqa:142
tokens: 196
loss:   5.4878
kl:     5.4607
top1:   0.3021
mass:   0.4325
code:   0.000227
anti:   0.001389
```

Final rolling windows:

```text
last20:
  loss 3.7247, kl 3.6102, top1 0.4719, mass 0.6499, code 0.0002

last100:
  loss 2.3369, kl 2.2427, top1 0.5192, mass 0.8031, code 0.0002

last250:
  loss 2.9063, kl 2.7740, top1 0.4706, mass 0.7404, code 0.0001

last500:
  loss 2.7836, kl 2.6106, top1 0.4936, mass 0.7730, code 0.0001

last1000/current:
  loss 2.5366, kl 2.4135, top1 0.4735, mass 0.7841, code 0.0000
```

Final task-family summary over the last 500 rows:

```text
aya:        n=60,  kl=2.6594, top1=0.3813, mass=0.7161
openr1:     n=111, kl=1.2634, top1=0.6186, mass=0.9594
mm_openr1:  n=140, kl=1.2627, top1=0.6270, mass=0.9316
cauldron:   n=189, kl=4.3848, top1=0.3569, mass=0.5640
```

Valid smoke records available:

```text
Phase-2 step 1000:
  applied 354 / 354, missing 0, mismatches 0
  generation: "\n\n\n\n\n\n<|im_end|>"

Phase-2 step 2000:
  applied 354 / 354, missing 0, mismatches 0
  generation: "\n\n\n\n<think>\n\n</think>\n\n**No**\n\n**No**\n\n**No"
```

Stopped-run conclusion:

- The experiment was stopped before the planned step-3000 and step-5000 hard
  gates.
- The run proved the global all-2D ternary QAT path can train on one A100 with
  the state-light mirror optimizer and memory-bounded code/prox machinery.
- It did not prove behavior recovery. Every valid hard-generation smoke through
  Phase-2 step `2000` was negative.
- The latest metrics before stop were still dominated by high-loss Cauldron
  rows, with OpenR1/mm_OpenR1 materially healthier but not sufficient to recover
  free generation.
- The next run should treat this as evidence that global participation is
  necessary but not sufficient; the objective/curriculum likely needs earlier
  on-policy survival KL, hidden/residual-stream anchoring, and shard-aware or
  staged data scheduling.
