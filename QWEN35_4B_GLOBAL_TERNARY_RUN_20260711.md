# Qwen3.5-4B Global Ternary QAT Run

## Experiment identity

- Run: `qwen35_4b_NORMCOMP_fullKL_anchorDecay_5B_20260711`
- Model: `Qwen/Qwen3.5-4B`
- Machine: one NVIDIA A100 80 GB
- Started: 2026-07-11
- Stopped deliberately: 2026-07-19
- Stop point: step 19,800, 1,946,419,200 supervised tokens (38.93% of the planned 5B-token budget)
- Latest durable checkpoint: `ckpt_19000.pt`

The process was terminated cleanly at the user's request. The checkpoint is at
step 19,000, while the final logged probe is at step 19,800. Do not attribute
the step-19,800 metrics to the checkpoint without evaluating the checkpoint
directly.

## Objective

Test whether a Qwen3.5 transformer can recover teacher behavior while all
trainable 2D trunk matrices execute a hard ternary forward pass. The run adds
two compensation mechanisms that were absent from the earlier control:

1. continuous, trainable text RMSNorm parameters;
2. hidden-state anchoring at full-attention boundaries, annealed completely to
   zero over the first 20% of the token budget.

The intended packed endpoint remains group-128 ternary codes plus FP16 scales,
with continuous 1D norms.

## Quantization and optimization

For each group of 128 master weights, the forward weight is hard ternary:

```text
Q(u)_i = s_g * q_i,  q_i in {-1, 0, +1}
```

The backward path uses a straight-through estimator into continuous masters.
The ternary masters are updated by the batched `muon-fast` path, while the
small continuous norm set uses AdamW-role updates. The run used a churn-based
learning-rate servo for ternary masters only.

Exact launch configuration:

```bash
python scripts/train_qwen35_ternary_kd.py \
  --model Qwen/Qwen3.5-4B \
  --data-root /workspace/kd_arms \
  --out /workspace/agentworld_runs/qwen35_4b_NORMCOMP_fullKL_anchorDecay_5B_20260711 \
  --budget-tokens 5e9 \
  --seq-len 1536 \
  --micro-batch 1 \
  --grad-accum 64 \
  --master-lr 3e-3 \
  --train-norms \
  --norm-lr 1e-4 \
  --optimizer muon-fast \
  --grad-clip 20 \
  --hidden-layers full \
  --hidden-weight 1.0 \
  --hidden-anneal-floor 0.0 \
  --anchor-frac 0.2 \
  --kd-temperature 2.0 \
  --kd-topk 128 \
  --probe-every 150 \
  --ckpt-every 1000
```

## Distillation loss

The behavior loss is an exact coarsened full-vocabulary KL: teacher top-128
tokens are represented explicitly and all remaining vocabulary probability is
aggregated into one tail bucket. This prevents the student from moving mass
outside the cached top-k set without penalty. Training uses temperature 2 and
the conventional temperature-squared multiplier.

```text
L = T^2 * KL_top128+tail(p_teacher || p_student)
    + 0.05 * L_CE
    + lambda_hidden(t) * L_anchor
```

The hidden anchor schedule was:

```text
lambda_hidden(t) = max(0, 1 - tokens / (0.2 * 5B))
```

It reached exactly zero after 1B tokens. Behavior continued improving after
the anchor disappeared, so the gain is not dependent on a permanently active
teacher-coordinate constraint.

## Data and adaptive mixture

Training consumed prebuilt teacher-cache arms for:

- knowledge;
- mathematics/reasoning;
- code;
- instruction following;
- tool use;
- multilingual text.

A KL-bandit adjusted sampling fractions toward lagging families. At the final
probe, multilingual remained the hardest family and received about 27.0% of
sampling, while the other arms each received about 11-16%.

## Final observed probe

Final logged probe at step 19,800:

| Metric | Value |
| --- | ---: |
| Tokens | 1,946,419,200 |
| Mean full KL | 0.4246758 |
| Mean top-1 agreement | 75.1248% |
| Mean student top-128 mass | 93.3082% |
| Ternary code churn vs initialization | 4.4472% |
| Sign inheritance | 99.8896% |
| Zero fraction | 31.6214% |
| Movement statistic | 0.5212599 |
| Effective master LR | 0.0063304 |
| Norm LR | 0.00007034 |
| Hidden anchor weight | 0 |
| Throughput | 2,904 tokens/s |
| Peak allocated GPU memory | 59.8 GB |

Per-family KL at the final probe:

| Family | KL |
| --- | ---: |
| Knowledge | 0.4175 |
| Math | 0.4441 |
| Code | 0.3899 |
| Instruction | 0.3713 |
| Tools | 0.2267 |
| Multilingual | 0.6985 |

## Control comparison

The prior full-KL control checkpoint reported:

- mean KL: 0.58687;
- mean top-1 agreement: 71.15%.

At the final probe, the norm-compensation/anchor-decay run achieved:

- 27.6% lower mean KL;
- 3.98 percentage points higher top-1 agreement.

This is strong evidence that continuous norm compensation and temporary
boundary anchoring materially improve hard-ternary global QAT. It is not yet
evidence of behavior-lossless compression: no post-stop generation suite or
downstream benchmark was run, and the durable checkpoint must be evaluated
separately.

## Important scientific interpretation

1. The model improved after the hidden anchor reached zero. Temporary
   residual-stream guidance helped optimization without becoming a permanent
   crutch.
2. Trainable 1D norms are a high-value continuous compensation channel.
3. KL and top-1 improved substantially while churn reached only 4.45%, far
   below the Prism-inspired 20-30% endpoint signature. The required amount of
   code migration remains unproven; forcing a forensic target is not a valid
   substitute for behavioral evaluation.
4. Sign inheritance remained above 99.88%, so almost all observed code change
   was support activation/deactivation rather than active-sign reversal.
5. Multilingual behavior remained the dominant error source despite adaptive
   oversampling.

## Archived artifacts

The network-storage archive should contain:

- `ckpt_19000.pt` (latest durable training state);
- `train_log.jsonl` (structured probe history);
- `train.log` (stdout/stderr training log);
- an artifact manifest with byte sizes and SHA-256 checksums;
- this report and the source commit containing the trainer implementation.

The older checkpoints were intentionally removed under a two-checkpoint local
retention policy to control A100 disk use.

SHA-256 at archival time:

```text
47c1e132d1ba92975e25d1211f883b496f009b2ca50b38dd0009853e352a5822  ckpt_19000.pt
56e8c4c38f6204b69a7612b7a7b59999f0067e9d170dec7fb542fda56b98d05f  train_log.jsonl
5e4fed6092c8172a87dcf0383ea0e855f35b3fe572c9363981af7f5468ecc416  train.log
```
