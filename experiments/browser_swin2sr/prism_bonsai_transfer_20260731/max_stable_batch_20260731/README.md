# Swin2SR maximum-stable-batch study

## Question

What is the largest useful physical batch for the exact BackdropKit Swin2SR
retained-precision training path on this 64 GiB Apple M5?

This study distinguishes three different quantities:

1. **Physical batch**: examples resident in one forward/backward update.
2. **Effective batch**: physical batch multiplied by gradient accumulation.
3. **Quality-qualified batch**: a physical batch that improves throughput and
   retains development quality at equal examples processed.

Only the first quantity is selected here. `protocol.json` was sealed before
any capacity output. The follow-up matched study selects the third quantity.

## Capacity gate

- exact hard-ternary student and full-precision teacher;
- exact 128 px crop and retained-precision loss path;
- one fresh MPS process per batch candidate;
- geometric search from 8 through 1,024;
- binary refinement in multiples of eight after the first failure;
- two warm-up and five measured optimizer updates;
- finite losses and gradients on every measured update; and
- peak driver allocation at or below 75% of MPS recommended memory.

The explicit 25% headroom rejects batches that technically finish only by
operating at the allocator or swap cliff.

## Run order

Do not run beside another MPS job. From the repository root:

```sh
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
  experiments/browser_swin2sr/prism_bonsai_transfer_20260731/max_stable_batch_20260731/run_batch_study.py \
  experiments/browser_swin2sr/prism_bonsai_transfer_20260731/max_stable_batch_20260731/protocol.json
```

The runner verifies every sealed source and input hash before starting. It
resumes completed probe receipts and never retries a failed MPS process inside
the same process.

## Required follow-up

Capacity alone cannot change the training default. Compare batch 8 and the
capacity winner using exactly the same examples and seeds under:

- the fixed `4e-6` learning rate;
- square-root learning-rate scaling; and
- linear learning-rate scaling.

Choose the largest batch that materially improves images per second while
remaining non-inferior on ground-truth, edge, and teacher metrics. If the
capacity winner loses quality, move downward along the measured throughput
curve rather than hiding the regression with more training data.
