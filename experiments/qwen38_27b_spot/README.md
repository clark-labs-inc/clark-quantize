# Qwen3.8-27B single-GPU spot laboratory

This lane validates quantization ideas on the official dense
`Qwen/Qwen3.8-27B` checkpoint using one high-memory GPU. The supported profiles
are an A100 with 80 GB VRAM and an RTX PRO 6000 Blackwell Server Edition with
96 GB VRAM. It is not a claim that simultaneous global QAT fits on one GPU.

## Fixed boundaries

- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Model payload: 18 safetensors shards, approximately 55.6 GB.
- Node: one compute-capability 8.0 or 12.0 GPU; CUDA 13 PyTorch environment.
- Useful scope: BF16 baseline, PTQ, layer/block reconstruction, cached-teacher
  probes, packed-artifact checks, and held-out behavioral canaries.
- Out of scope: simultaneous online-teacher, all-parameter QAT.

The exact Clark Quantize, canonical in-product Clark Scientist, Clark Desktop,
and model revisions copied to the node are recorded in
`source_provenance.json`. The Scientist source is a Rust workspace component,
not a Python dependency; it remains under its canonical private Clark product
path rather than being copied from the retired sibling repository.

## Spot contract

Verda can discontinue a spot instance at any time without warning. The node
therefore uses resumable Hugging Face downloads, atomic status receipts under
`/workspace/quant-state`, one-job locking, and run-local command/log/exit
receipts under `/workspace/quant-runs`.

The first Prime Intellect / Verda node was evicted on 2026-08-14 at 20:05 PDT,
and Prime showed no retained shared disk. Treat `/workspace` as ephemeral for
this provider path regardless of lower-level Verda defaults. A watchdog on the
same root disk is necessary but not sufficient.

Off-node mirroring is intentionally disabled. An external backup destination
must be explicitly approved for this project before any credential, host, or
remote path is configured. On the current non-spot A100, the immutable model,
atomic run state, and resumable pack artifact remain local under `/workspace`.
The full pack produces one artifact rather than rolling checkpoint copies.
Future QAT launchers must retain at most the newest two local checkpoints.

## Bootstrap and model download

```bash
cd /workspace/clark-quantize
bash scripts/spot/bootstrap_quant_node.sh

scripts/spot/run_spot_job.sh qwen38-download -- \
  bash experiments/qwen38_27b_spot/download_model.sh
```

After the download, run the streamed preflight before constructing a 27B QAT
model:

```bash
scripts/spot/run_spot_job.sh qwen38-preflight -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/preflight_quantization.py
```

The preflight does not instantiate the full model. It inventories every tensor,
reports both ideal base-3 and aligned-2-bit effective-rate ledgers (including
FP16 scales and retained islands), and compares optimal-MSE versus absmean
ternary initialization on representative matrices. Its atomic receipt is
`/workspace/quant-state/qwen38-preflight.json`.

The conservative group-128 lane is 2.178900571 effective bits per original
parameter after its scales and BF16-retained vision/MTP islands are counted; it
must not be described as a complete 1.58-bit model. The separately sealed
`target_158_preregistration.json` defines the next storage rung: all rank-2
parameters plus per-row FP16 scales project to 1.590214265 bpp. That projection
does not promote until all 582 rank-not-2 tensors are verified as BF16 and
language, vision, MTP, convolution, and linear-attention reconstruction evidence
exists. It still requires modality-aware behavioral evaluation before a
model-quality claim.

The bounded BF16 load/generation gate is:

```bash
scripts/spot/run_spot_job.sh qwen38-bf16-smoke -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/bf16_smoke.py
```

It is local-only, deterministic, capped at 16 new tokens, and writes peak VRAM
and timing to `/workspace/quant-state/qwen38-bf16-smoke.json`.

The first ideal base-3 lane uses an exact streaming arithmetic-code reference,
not the common five-trits-per-byte approximation:

```bash
scripts/spot/run_spot_job.sh qwen38-base3-reference -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/base3_packing_research.py
```

Promotion requires an exact CRC-protected round-trip of one million real
optimal-MSE Qwen trits and payload rate no more than 0.001 bits/trit above
`log2(3)`. Reference throughput is reported honestly; a native parallel codec
is a later gate, not implied by mathematical packing success.

The next promotion gate is deliberately run twice. The first invocation must
be terminated after at least one committed chunk; rerunning the same command
must strictly validate and reuse that prefix before completing:

```bash
scripts/spot/run_spot_job.sh qwen38-base3-resume -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/base3_resume_gate.py \
  --commit-delay 2
```

The chunk directory uses complete independently decodable `CQ3A1` frames,
per-chunk raw/frame SHA-256, per-frame CRC32, bounded parallel workers, and an
atomically replaced partial manifest. Resume rejects any model, quantizer,
source-order, shape, or regenerated-code mismatch.

Only after that receipt promotes, the mapped full-checkpoint pack is:

```bash
scripts/spot/run_spot_job.sh qwen38-base3-full -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/full_base3_pack.py
```

It streams one tensor row batch at a time, bounds outstanding CPU encoders,
writes independently verifiable FP16 scale blocks, and publishes a durable
progress receipt. Retained high-precision tensors remain referenced from the
immutable downloaded checkpoint during this packing study; their bytes are
included in the complete-model ledger and are not misreported as ternary.

After the full pack promotes, run the local-only behavioral PTQ control:

First, `export_retained_tensors.py` writes the non-ternary BF16 tensors to a
separate authenticated safetensors file and reconciles their exact parameter
and byte counts with the preflight ledger. Together with the code frames and
FP16 scales, this forms a complete parameter bundle while the pinned source
directory continues to provide configuration and tokenizer files. The reported
complete rate comes from the actual framed pack ledger, not the lower ideal
preflight projection. Final audit reopens and hashes every retained tensor
against its manifest. It also independently walks the finalized code artifact:
the exact chunk and scale file sets, contiguous tensor-row coverage, frame
headers, CRC32, SHA-256, canonical padding, payload bits, frame bytes, scale
counts, retained bytes, and complete-model/transport rates are recomputed from
disk rather than trusted from the pack receipt.

```bash
scripts/spot/run_spot_job.sh qwen38-base3-quality -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/behavioral_quality_gate.py
```

It loads the BF16 parent once, records held-out likelihood and deterministic
generations, replaces only artifact-mapped matrices in bounded row batches,
and then measures perplexity degradation and forward teacher KL. This is a
small diagnostic gate, not a substitute for downstream benchmarks or QAT.
A non-promoting result is retained as terminal science. It may enter the
preregistered rescue rung only when the receipt independently proves finite
metrics, deterministic nonempty controls, exact artifact/model identity, full
parameter application, and an authenticated local BF16 teacher cache.

Repeated reconstruction jobs build and require the optional native arithmetic
decoder on the A100:

```bash
scripts/spot/build_base3_native.sh
```

It implements the identical `CQ3A1` state machine behind the Python fallback.
Frame SHA-256/CRC, decoded-code SHA-256, scale SHA-256, and shape checks remain
mandatory, so the optimization changes throughput but not artifact semantics.

QAT checkpoint writers must finish into a temporary directory, atomically
promote it to `checkpoint-STEP`, write a matching complete `checkpoint.json`,
and then run:

```bash
python scripts/spot/prune_qat_checkpoints.py \
  --root /workspace/quant-runs/qwen38-qat/checkpoints --keep 2
```

The retention tool operates only on immediate complete checkpoint directories,
refuses symlinks and malformed manifests, and leaves the active incomplete save
untouched. Thus disk use is bounded to two complete checkpoints plus one
in-progress atomic save.

The practical packing comparison is deliberately separate from the ideal
arithmetic artifact:

```bash
scripts/spot/run_spot_job.sh qwen38-radix5-performance -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/radix5_performance_gate.py
```

`CQ3R5` is canonical, CRC-protected, and vectorized at five trits per byte. Its
rate is 1.6 bits/trit, not `log2(3)`; the receipt reports the exact roughly
50-MB full-model cost of that throughput tradeoff.

If PTQ quality is insufficient, select the first QAT target by measured
behavioral sensitivity rather than weight error:

The four candidate layers, mapped tensor/parameter counts, tie-break rule, and
all eight calibration texts are sealed in `qat_preregistration.json`. The
sensitivity gate must find positive teacher-KL recovery, and the teacher cache
must match those exact texts, top-k width, token cap, selected layer, and model
revision before training is allowed.

```bash
scripts/spot/run_spot_job.sh qwen38-layer-sensitivity -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/layer_sensitivity_gate.py
```

The gate restores layers 0, 3, 31, and 63 to BF16 one at a time, ranks their
teacher-KL recovery per GiB, and proves exact behavioral reversion after each
layer is returned to the packed ternary state. It writes no model checkpoint.
The subsequent one-layer training rung is fixed in `qat_preregistration.json`;
it may not expand beyond ten optimizer steps until its hard-projected held-out
metrics, exact resume, VRAM, and two-checkpoint retention criteria pass.
Its eight calibration sequences are disjoint from the likelihood/sensitivity
texts and are cached locally by `qat_teacher_cache.py` only after sensitivity
selects a layer with positive teacher-KL recovery.

If and only if the ten-step receipt promotes, continue the exact authenticated
master weights, BonsaiQAT8 state, global step, and RNG state to the sealed
100-step rung:

```bash
scripts/spot/run_spot_job.sh qwen38-one-layer-qat-100 -- \
  /workspace/venvs/quant/bin/python \
  experiments/qwen38_27b_spot/one_layer_qat_100.py
```

It evaluates held-out teacher KL every ten steps, stops after two consecutive
regressions, and retains only checkpoint 10 plus one terminal checkpoint. A
valid non-promotion is a scientific result and is not automatically rerun.

For reboot recovery, install and enable (without starting over the live
transient queue) `scripts/spot/quant-qwen38-pipeline.service`. Its orchestrator
resumes the chunked full pack, accepts schema-valid terminal scientific
receipts without silently promoting them, enforces two-checkpoint retention,
and refuses ambiguous reruns. Infrastructure and layer-selection prerequisites
remain promotion-gated; a valid non-promoting QAT receipt completes its bounded
rung and cannot trigger the 100-step continuation.

`scripts/spot/quant-qwen38-post-pipeline.service` is ordered after that unit.
It conditionally runs the authenticated 100-step continuation only when the
ten-step receipt promotes, accepts a valid non-promoting receipt as terminal,
runs the independent target-1.59 reconstruction gate, reapplies checkpoint
retention, and invokes the fail-closed completion audit. It never launches a
second full checkpoint pack.

Clark Scientist/Engineer completion requires more than a binary self-test.
`source_provenance.json` records each local source archive/bundle path and hash,
the staged source path and revision, and the expected worker version. The final
audit rehashes those transport artifacts, checks live Git heads where present,
matches all four revisions to the install receipt, and reruns the exact worker
protocol/catalog self-test.

Inspect at any time:

```bash
cat /workspace/quant-state/status.json
tail -f /workspace/quant-runs/qwen38-download/job.log
nvidia-smi
df -h /workspace
```

Every scientific probe must add a falsifiable hypothesis, primary metric,
protected metrics, token/time budget, and stop condition before launch. A lower
weight-reconstruction error alone is not a quality result.
