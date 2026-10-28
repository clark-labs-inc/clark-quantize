# Baseline freeze: OmniVoiceGraph U26

The evaluation gate starts from the current best recorded real-data checkpoint,
not from a QAT overlay or a newer but incomplete training run.

| Field | Value |
|---|---|
| Checkpoint | `u26_best_step46k.pt` |
| Local source | `/Users/stan/Library/OmniVoiceGraph/models/u26_best_step46k.pt` |
| Quant path | `/workspace/models/u26_best_step46k.pt` |
| SHA-256 | `2bb0acb40d2bfdc834f92a3925278491147d0810bd8dd80fe953a324c41ff295` |
| Quant service | `ovg-api`, checkpoint loaded successfully |
| Recorded run | `20260816T092000Z-u26-fused-152` |
| Metric implementation | SGLang Omni `ca6aa45513ff0c9728ff122629fde96acc26054d` |
| Benchmark data | `/Volumes/Lexar/moss-transcribe-diarize-public-eval/` |

The recorded U26 real-data history reached held-out WER `0.2252` at step 44k
and `0.2266` at step 46k; step 46k is the deployed/best-file artifact and is
used for a reproducible checkpoint identity. U33 is not substituted here: its
current run has only a 2k-step record and is not yet the evidence-backed best
checkpoint.

The next gate is inference-only. It must use the official SGLang Omni
MOSS-style metric implementation or an independently parity-tested port, and
must report exact sample counts, CER, cpCER, Δcp, latency/RTF, throughput, RSS,
errors, and the checkpoint hash. QAT is out of scope until this baseline is
complete and the results justify it.
