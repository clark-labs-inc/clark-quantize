# Clark Scientist runtime safety audit

- Source task: `/root/clark_scientists_audit`
- Source session: `019fb948-d269-7902-af3b-564ee3ad604f`
- Captured: 2026-08-01
- Status: archival research lead; verify claims against primary evidence before use


Canonical implementation authority: the separately versioned `clark-scientist` engine; this report is retained here because its scope and recommendations concern this Swin2SR program.

## Report

Read-only audit complete. The safest conclusion is:

**Do not run the existing autoresearch loop against the active `clark-quantize` tree.** Its rollback mechanism overwrites declared writable files from checkpoints, which is unsafe with concurrent dirty work. Its evaluator also has no timeout and buffers all output in memory, making long Swin2SR training runs risky.

Key evidence:

- The Clark monorepo’s `make autoresearch` is designed to generate and grade Clark agent `EvalScenario`s, not quantization experiments:
  - `/Users/stan/Documents/git/clark/Makefile:832-926`
  - `/Users/stan/Documents/git/clark/evals/rust/src/eval_research/catalog.rs:77-234`
  - `/Users/stan/Documents/git/clark/evals/rust/src/eval_research/validation.rs:105-190`
- The actual Clark Scientist runtime is `/Users/stan/Documents/git/clark-scientist`. It has a sound event-sourced scientific kernel:
  - preregistration and protected metrics: `crates/scientist-protocol/src/research.rs:16-49`
  - bounded experiment proposals: `crates/research-kernel/src/command.rs:204-231`
  - authority attenuation: `crates/research-kernel/src/command.rs:279-343`
  - replicated-evidence acceptance: `crates/research-kernel/src/command.rs:505-543`
  - content-addressed artifacts and source snapshots: `crates/scientist-protocol/src/artifact.rs:16-88`
  - evidence provenance: `crates/scientist-protocol/src/evidence.rs:58-104`
  - persist-before-execute effects: `crates/research-runtime/src/runtime.rs:68-120`
- Its Scientist proposal lane is read-only, tool-free, and returns a strict hypothesis:
  - `crates/scientist-pack/src/lib.rs:27-77`
  - `crates/scientist-pack/src/lib.rs:195-279`
  - `crates/code-worker/src/effects.rs:211-286`
- The unsafe legacy autoresearch rollback is explicit:
  - checkpoint capture: `crates/code-worker/src/checkpoint.rs:21-64`
  - restoration by overwriting/removing files: `crates/code-worker/src/checkpoint.rs:70-92`
  - controller invokes rollback after failures/rejections: `crates/code-research/src/controller.rs:150-313`
- The evaluator lacks a subprocess timeout and buffers stdout/stderr through `.output()`:
  - `crates/code-research/src/evaluator.rs:35-97`

Recommended integration:

1. Use `scientist_bootstrap` and `scientist_propose_hypothesis` first, in a new Clark Scientist session registered to `clark-quantize`. Give it a frozen, hashed summary of current Swin2SR findings plus Bonsai/Prisma weight analysis. This produces research proposals only—no mutations or evidence claims.

2. Ask for explicit model/provider confirmation before invoking it. The installed product route defaults to Clark-managed GLM-5.2 at xhigh; direct Qwen is an explicit paid route:
   - `/Users/stan/Documents/git/clark-scientist/README.md:85-135`

3. For actual experiments, build a small Swin2SR domain adapter on the newer `research-kernel`, not the legacy controller. It should:
   - write only new run directories such as `experiments/browser_swin2sr/scientist_runs/<id>`;
   - pin source, dataset, configuration, checkpoint, and evaluator SHA-256s;
   - enforce subprocess timeout and bounded logs;
   - use the existing glyph-safe promotion policy as the deterministic oracle;
   - require at least three seeds;
   - protect Q4 non-inferiority, text/glyph quality, browser artifact size, runtime correctness, CPU time, and memory;
   - let the Scientist propose structured intervention JSON while deterministic code compiles it into exact training commands.

4. If the legacy loop is used at all, give it a disposable isolated copy containing only experiment configuration and generated artifacts—never the live dirty repository.

There is no dedicated model-based “critic” in Clark Scientist. Its stronger scientific critic is the combination of preregistration, deterministic evaluators, protected metrics, held-out data, replication, provenance, and explicit acceptance decisions. That is the right authority model for Swin2SR; a second LLM can challenge hypotheses but should not decide promotion.
