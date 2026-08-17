# Mandatory Quick Capability Preflight

Status: active policy
Date: 2026-08-17

Loss, KL, teacher-forced token alignment, kernel parity, and throughput cannot
authorize a longer quantization run by themselves. Every candidate must run a
cheap generation-level capability tripwire against its exact BF16 parent before
we extend training, expand quantized depth, or move from 4B to 27B.

## Default quick suite

1. Four fixed 512-token WikiText-2 perplexity chunks for a broad likelihood
   collapse signal.
2. Eight pinned level-1 MATH-500 examples with exact final-answer scoring.
3. `CUTE-Quick-20`: four fixed official rows from each of `contains_char`,
   `ins_char`, `del_char`, `swap_char`, and `sub_char`.
4. Six sealed execution-free code traces as a secondary syntax/reasoning
   signal. These are not HumanEval and are never reported as HumanEval.

An equivalently cheap, pinned, subject-balanced MMLU slice may replace CUTE
when character generation cannot be evaluated through the candidate runtime.
MATH-500 may serve as the minimum emergency tripwire, but the default CPU lane
runs both MATH-500 and CUTE because they fail in different ways.

`CUTE-Quick-20` uses official CUTE data and task prompts pinned at repository
revision `4238eaa301abb698d8850f42297564362d4e9032`. It is a development subset,
not a full CUTE run and not comparable to the CUTE leaderboard.

## Stop rule

The quick gate stops a candidate if it retains less than 60% of the BF16
control's correct items on either MATH-500 or CUTE-Quick-20. Fractions are
converted to integer requirements with a ceiling. A BF16 score of zero makes
that slice invalid rather than making a candidate pass. Infrastructure errors
are invalid runs, not zero scores.

This threshold detects catastrophic forgetting; it does not authorize model
promotion. A candidate that passes proceeds to the larger frozen evaluation,
where the program's stricter five-absolute-percentage-point quality boundary,
full denominators, repeated performance receipts, and artifact accounting
still apply.

## Required receipt chain

- Exact model SHA-256 for BF16 and candidate artifacts.
- Frozen manifest content hash and evaluator source hash.
- One record per item, including raw output, normalized prediction, gold label,
  token usage, finish reason, and latency.
- Exact correct/total denominator per task.
- Atomic comparison receipt from `compare_cpu_capability_tripwire.py`.
- A nonzero evaluator or comparison exit is preserved as evidence; it must not
  be overwritten by a later successful command.

No long QAT continuation, layer-depth expansion, or 27B scale-up begins without
this receipt chain for the candidate artifact being advanced.

For llama.cpp-compatible candidates, run the full mandatory generation gate as
one failure-preserving job:

```bash
run_llama_cpu_quick_gate.sh \
  /path/to/candidate.gguf candidate-label \
  /path/to/frozen-bf16-math.json /path/to/frozen-bf16-cute.json \
  /path/to/output-directory
```

The wrapper hashes the exact candidate, launches and cleans up its isolated
server, writes item-level CUTE and MATH/code reports atomically, and exits `2`
when the frozen retention comparison rejects the candidate. A missing report
or any other nonzero exit remains an infrastructure-invalid run, never a zero.
