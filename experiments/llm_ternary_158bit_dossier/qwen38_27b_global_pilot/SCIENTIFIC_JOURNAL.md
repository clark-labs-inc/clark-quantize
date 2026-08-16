# Scientific Journal

## 2026-08-16 — preregistration

The prior global base-3 PTQ artifact is structurally complete but behaviorally
catastrophic (perplexity ratio 117.533 and teacher KL 5.3109). The previous
one-layer QAT rescue was only a mechanism check: ten updates over eight short
calibration rows changed too little of the model to test the global-training
hypothesis.

The most transferable successful pattern in this repository is hard-forward
training with continuous masters, frequent projection/scale refresh, compact
teacher supervision, and direct downstream evaluation. This pilot adapts the
existing state-light all-2D trainer rather than inventing another optimizer.

The local Clark headless worker passed its self-test at version 0.1.155. No paid
or external Clark provider run is authorized for this experiment, so the frozen
protocol and implementation/test receipts serve as the deterministic scientist
and engineer handoff. This is not represented as an external specialist result.

No training result exists yet. BF16 and current base-3 PTQ capability baselines
must be persisted before the first retained trained checkpoint.

## 2026-08-16 — terminal pilot result

The one-A100 mechanism fit after fixing a transient source-module retention bug
in the inherited replacement routine. The final run completed 224/224 updates,
quantized 496 text matrices, trained 208 norm tensors, and peaked at 52.83 GiB
allocated / 53.65 GiB reserved. Sampled code churn reached 1.5827%; 77.10% of
the codes changed at step 56 remained changed at the terminal step.

Cached-teacher diagnostics improved substantially on sampled training rows, but
the frozen downstream gate disproved the recovery hypothesis. The deployed hard
patch generated exactly `Based` in two tokens for all 32 MATH-500 and all 32
GSM8K prompts. It scored 0/32 on both tasks and 15/64 on MMLU, versus BF16
scores of 10/32, 14/32, and 53/64. It did not improve over the base-3 PTQ arm;
MMLU was eight questions worse.

Interpretation: 42,140 input tokens / 13,381 supervised positions and 224
state-light updates are enough to create persistent code movement and reduce
cached top-k KL, but nowhere near enough to preserve open-loop generation in a
27B globally hard-ternary model. Cached teacher agreement is therefore not a
promotion surrogate. A credible successor needs orders of magnitude more
training tokens, broader code/instruction/multilingual coverage, and periodic
frozen downstream probes; the failed checkpoint must not be packed or released.

The GPU power cap used for thermal safety was restored from 250 W to the 300 W
device default after evaluation. Raw caches, logs, model artifacts, predictions,
and the single new checkpoint remain on the A100; unrelated client
infrastructure was not used.
