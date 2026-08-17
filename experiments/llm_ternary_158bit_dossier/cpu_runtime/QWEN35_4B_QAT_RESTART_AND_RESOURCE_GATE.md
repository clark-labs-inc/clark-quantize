# Qwen3.5-4B QAT restart and CPU resource gate

Status: the current CCX33 CPU node is **rejected for global 4B QAT before
model load**. The launch package is restart-safe, but no training result is
claimed.

## Why the old trainer was not safe to relaunch

The July trainer wrote `ckpt_*.pt` directly. Those files included model and
optimizer tensors but omitted Python, NumPy, Torch, CUDA, per-arm data RNG,
KL-bandit, movement-servo, and KL-history state. A resumed process therefore
was not the same stochastic experiment. The save itself was non-atomic and
had no content hash or automatic retention.

The repaired path now provides:

- temporary-directory save followed by atomic directory promotion;
- a SHA-256 manifest and master-tensor inventory;
- complete model state split into masters and retained tensors without
  duplicating master storage;
- optimizer, every RNG, arm sampler, bandit, servo, and KL-history replay;
- strict run identity across model revision, sealed data, training arguments,
  source hashes, and framework versions;
- BF16 Muon momentum restoration instead of PyTorch's implicit FP32 upcast;
- disk admission plus two-checkpoint retention;
- optimizer-step-boundary SIGTERM/SIGINT checkpointing;
- rejection of legacy non-exact resume unless explicitly marked
  non-promotable.

The independent tiny-model replay smoke checkpointed after step 3, rebuilt a
fresh model and optimizer, restored the checkpoint, and executed step 4 again.
The sampled arm, NumPy arm values, Python random value, Torch random tensor,
loss, full model state, full optimizer state, bandit, servo, and KL history all
matched exactly. Its systemd run used 213.8 MB peak memory and exited zero.
The authoritative rerun also pins the checkpoint and optimizer implementation
hashes; it used 213.2 MB peak memory and exited zero. Receipt:
`receipts/qwen-global-resume-smoke-v1-source-complete.json`. The earlier
source-incomplete receipt is retained as development evidence.

## Corrected resource receipt

The gate inspected all 738 tensors in Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and hashed both weight shards:

- shard 1: `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`;
- shard 2: `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`.

It found 3,565,158,400 hard-ternary master parameters out of 4,659,865,088
total parameters. The persistent lower bounds are:

| Mode | Raw lower bound | With only 10% headroom | Node available | Result |
|---|---:|---:|---:|---|
| Full teacher + anchors | 54.29 GB | 59.72 GB | 31.92 GB | fail |
| Offline teacher cache, no anchors | 44.97 GB | 49.47 GB | 31.92 GB | fail |

These bounds already include FP32 masters and gradients, BF16 hard-forward
cache and Muon momentum, retained model tensors, norm state, and—in the full
recipe—the BF16 teacher. They explicitly exclude activations, autograd saved
tensors, allocator fragmentation, temporary surgery overlap, kernel
workspaces, and framework overhead. The previous observed A100 run peaked at
59.8 GB, consistent with the bound.

One atomic checkpoint is estimated at 23.58 GB. Two retained generations plus
one in-progress save and a 10 GiB reserve require 81.48 GB. The node had
141.61 GB free, so disk passed while memory failed.

The first receipt used `MemFree` (10.66 GB) instead of Linux `MemAvailable`.
It is retained as invalid measurement evidence and was not overwritten. The
corrected receipt uses `MemAvailable`; both lead to the same rejection, but
only the corrected receipt is authoritative.

## Decision

- Do not start global 4B QAT on the 32 GB CCX33.
- Use at least an 80 GB A100/H100-class training GPU for the faithful anchored
  recipe already observed at 59.8 GB peak, or a CPU host with materially more
  than the 59.72 GB lower bound; 96-128 GB RAM is the practical CPU admission
  tier before throughput is considered.
- The existing CPU node remains useful for frozen BF16/candidate evaluation,
  packing, correctness tests, and kernel work.
- The first new model-changing checkpoint must run the quick WikiText,
  MATH-500, and CUTE/MMLU capability tripwire before any longer continuation.

Receipts:

- authoritative: `receipts/qwen35-4b-qat-resource-v1-corrected.json`;
- invalid but preserved: `receipts/qwen35-4b-qat-resource-v1-invalid-memfree.json`.
