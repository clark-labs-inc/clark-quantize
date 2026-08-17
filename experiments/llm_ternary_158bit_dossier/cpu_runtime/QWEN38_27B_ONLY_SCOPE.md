# Qwen3.8-27B-only operating scope

Effective 2026-08-17, all new compute, implementation, export, evaluation, and
promotion work in this research lane is restricted to `Qwen/Qwen3.8-27B`.
Earlier smaller-model receipts may be cited as historical engineering evidence,
but they cannot launch new work or satisfy a Qwen3.8-27B gate.

The active sequence is:

1. finish the frozen Q4_K_M capability baseline;
2. measure the baseline at 1, 2, and 4 physical CPU cores;
3. convert only the preregistered layer-0 gate/up/down tensors to CQ3_0 and prove
   that every non-target GGUF payload remains byte-identical;
4. run CUTE, MATH/code, MMLU, and perplexity parity on that candidate;
5. compare paired baseline/candidate token throughput only after quality passes;
6. promote CPU kernels only through separately frozen correctness and
   performance gates; and
7. expand ternary depth only after the preceding 27B gate is valid.

Every model or graph integration must include a quick CUTE check and MMLU-type
capability evidence. Infrastructure-invalid work is neither a zero nor a
completed result. Loss, KL, reconstruction error, a running process, or a
microkernel speedup alone cannot support a capability or breakthrough claim.

Maximizing correct Qwen3.8-27B CPU prefill and decode tokens per second is a
continuous secondary objective. Storage compression, synthetic kernel timing,
and end-to-end token throughput are reported separately.
