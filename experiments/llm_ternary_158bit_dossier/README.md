# LLM 1.58-bit Ternary Research Dossier

Status: active CPU-first research program
Last reviewed: 2026-08-17

This folder is the self-contained navigation and interpretation surface for
Clark's LLM ternary-quantization experiments. The historical GPU training
records remain evidence, while the active lane is now a CPU-native ternary
runtime and kernel-aware training program. GPU training is retained as a means
to produce CPU-deployable weights, not as the deployment target.

## Folder contents

- [Scientific journal](SCIENTIFIC_JOURNAL.md) — verified findings, negative
  results, unresolved questions, and the gates required before this lane could
  be resumed.
- [Artifact index](ARTIFACT_INDEX.md) — exact links to authoritative run
  reports, frozen evaluation protocols, implementation surfaces, and
  scientific synthesis.
- [Folder migration record](FOLDER_MIGRATION.md) — files consolidated here,
  historical paths intentionally preserved, and the safe relocation gate.
- [CPU-first research program](cpu_runtime/RESEARCH_PROGRAM.md) — active
  execution order, artifact/runtime architecture, hardware lanes, and gates.
- [CPU kernel contracts](cpu_runtime/KERNEL_CONTRACTS.md) — exact operator and
  numerical contracts for the Qwen hybrid graph.

## Provenance policy

Historical reports, source files, protocols, checkpoints, and logs remain at
their original repository or recorded remote paths. They were not copied or
moved into this folder because doing so would break old commands, imports,
cross-references, or artifact provenance. This dossier is therefore
self-contained as an evidence map, while the linked run-specific receipt is
authoritative for every numerical claim.

Do not infer current process, checkpoint, or remote-storage state from this
archive. Reopen the live source before reporting current status.

## Research boundary

The active goal is text-first CPU inference for `Qwen/Qwen3.8-27B` only. User
direction on 2026-08-17 froze all Qwen3.5-4B work; those receipts remain
historical evidence but receive no new implementation, evaluation, export, or
compute. Historical BackdropKit and GPU-QAT work remains separate and is not
silently reclassified as CPU-runtime evidence.

The current 27B lane first seals a runnable `Q4_K_M` CPU control on CUTE,
MATH/code, MMLU, and WikiText. The next preregistered rung selectively changes
only the layer-0 MLP triplet to `CQ3_0`; its converter must prove all 848
non-target GGUF payloads byte-identical before any candidate score is accepted.
