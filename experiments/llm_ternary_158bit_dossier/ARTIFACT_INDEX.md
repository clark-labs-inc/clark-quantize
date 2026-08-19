# LLM 1.58-bit Ternary Artifact Index

Status date: 2026-08-19

This is a link-only index. Existing experiment files remain in their original
locations so historical commands, imports, and archived paths continue to work.

## Active CPU-first program

| Artifact | Role | Evidence boundary |
| --- | --- | --- |
| [CPU-first research program](cpu_runtime/RESEARCH_PROGRAM.md) | Active runtime, training, hardware, and promotion sequence | Design and work order; performance targets are not measured claims |
| [Qwen3.8-27B-only scope](cpu_runtime/QWEN38_27B_ONLY_SCOPE.md) | Current model and compute boundary | Historical smaller-model receipts are reference-only and cannot launch or satisfy an active gate |
| [CPU kernel contracts](cpu_runtime/KERNEL_CONTRACTS.md) | Qwen graph operator and numerical contract | Design v0; optimized kernels are not implemented yet |
| [Hetzner CCX33 bring-up and controls](cpu_runtime/HETZNER_CCX33_RUN_20260816.md) | Reproducible CPU environment, historical controls, and completed 27B arithmetic pack | Packing evidence is separate from model capability and end-to-end runtime evidence |
| [Qwen3.8-27B Q4 quality control](cpu_runtime/receipts/qwen38_27b_cpu_q4_baseline_v1/summary.json) | Frozen CUTE, math/code, MMLU, and WikiText reference | Quick development suite, not leaderboard-comparable or a full model-quality characterization |
| [Qwen3.8-27B Q4 CPU throughput control](cpu_runtime/receipts/qwen38_27b_cpu_throughput_v1/summary.json) | Repeated 1-, 2-, and 4-thread prefill/decode baseline | Throughput only; load latency, TTFT, energy, and candidate comparisons remain separate gates |
| [Qwen3.8-27B selective CQ3 conversion](cpu_runtime/receipts/qwen38_27b_layer0_mlp_direct_cq3_v1/conversion-receipt.json) | Exact three-tensor layer-0 MLP conversion and load smoke | Byte scope and loadability only; capability evaluation is still required |
| [Qwen3.8-27B selective CQ3 quality gate](cpu_runtime/receipts/qwen38_27b_layer0_mlp_direct_cq3_quality_v1/summary.json) | Terminal CUTE, MATH/code, MMLU-64, and WikiText comparison | Frozen quick quality gate passed for one layer-0 MLP triplet; not a whole-model or throughput promotion |
| [Qwen3.8-27B all-layer trained-CQ3 GGUF materializer](cpu_runtime/materialize_qwen38_27b_all_layer_trained_cq3_v46.py) | Fail-closed exact-byte overlay for all 192 trained v44 MLP payloads | Implementation and synthetic tests only; requires a passed v44 receipt plus an authenticated all-MLP CQ3 template, and excludes AFF3, residuals, quality, and performance |
| [Qwen3.8-27B fused rotated-AFF3 kernel](cpu_runtime/kernels/qwen38_aff3_rotated_avx2_v47.cpp) and [development receipt](cpu_runtime/receipts/qwen38_27b_aff3_rotated_avx2_v47/development-receipt.json) | Direct six-stage Givens rotation, group-128 A8 quantization, and AFF3_X16 register-lookup GEMV | Synthetic and authenticated direct-v43 layer-0 parity passed; post-implementation development evidence only, with no trained-v44, llama.cpp graph, quality, or performance result |
| [Qwen3.8-27B authenticated AFF3 GGML node](cpu_runtime/qwen38_aff3_ggml_v48.cpp) and [development receipt](cpu_runtime/receipts/qwen38_27b_aff3_ggml_v48/development-receipt.json) | Read-only mmap validation, load-time rotation coefficients, and four-task decode/eight-token graph execution | Preregistered direct-v43 layer-0 correctness passed bit-exactly at one and eight tokens; no timing, trained-v44, model-graph, quality, or breakthrough evidence |
| [Qwen3.8-27B AFF3 llama.cpp integration](cpu_runtime/llama_cpp/prepare_llama_cpp_qwen38_aff3_v49.py) and [development receipt](cpu_runtime/receipts/qwen38_27b_aff3_llamacpp_v49/development-receipt.json) | Exact Qwen full/linear-attention graph seams, projection tensor skip, authenticated model-owned sidecars, and graph-owned four-task scratch | Clean transformed llama.cpp build plus direct-v43 layer-0 one/eight-token parity passed; actual 27B model load, timing, trained-v44, capability, and breakthrough evidence remain open |
| [Qwen3.8-27B persistent-affinity gate](cpu_runtime/qwen38_27b_persistent_affinity_preregistration_v1.json) | Exact full-token comparison of per-call versus once-per-worker CPU pinning | Frozen protocol queued after graph and capability work; no observed speedup yet |
| [Qwen3.8-27B full-graph profile](cpu_runtime/qwen38_27b_full_graph_profile_preregistration_v1.json) | Exact 557-call projection-time attribution over the 64-layer graph | Frozen diagnostic protocol; it cannot establish a throughput improvement or capability claim |

## Primary run records

| Artifact | Role | Evidence boundary |
| --- | --- | --- |
| [Qwen3.5-4B global ternary run](../../QWEN35_4B_GLOBAL_TERNARY_RUN_20260711.md) | Strongest recorded dense-LLM training result | Final logged probe is step 19,800; durable checkpoint is step 19,000; no downstream post-stop suite |
| [Qwen AgentWorld status](../qwen_agentworld/STATUS.md) | Detailed experiment ledger and code-homotopy map | Contains reduced-context canaries and remote artifact paths, not a full-fidelity release gate |
| [Qwen AgentWorld README](../qwen_agentworld/README.md) | Official-harness boundary, setup, and commands | Full fidelity requires the documented 4-GPU boundary |
| [global mirror 10k run](../qwen_agentworld/GLOBAL_T2_MIRROR_10K_RUN.md) | Early global all-2D mirror-SGD evidence | Run-era snapshot; consult later status before treating it as current |
| [global fulltrain v1](../qwen_agentworld/GLOBAL_T2_FULLTRAIN_V1_RUN.md) | Mirror-SGD/AdaGrad global ramp and OOM history | Establishes fit and failure modes, not quality |
| [global fulltrain v2](../qwen_agentworld/GLOBAL_T2_FULLTRAIN_V2_RUN.md) | Hard-shadow and hidden-anchor follow-up design | Design/execution record, not a completed promotion receipt |

## Scientific synthesis and forensics

| Artifact | Role | Caution |
| --- | --- | --- |
| [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md) | Separates verified mechanisms, historical results, executable candidates, and hypotheses | The large-model lane is archived |
| [archived large-model research program](../large_scale_ternary/RESEARCH_PROGRAM.md) | Historical completion standard, evidence, gates, and protocol status | Superseded as an active plan by the CPU-first program above |
| [Qwen ternary recipe](../../TERNARY_QWEN35_RECIPE.md) | Signature-steered KD hypothesis and trainer design | Some signature targets are hypotheses, not success criteria |
| [A100 runbook](../../A100_RUNBOOK.md) | Reproduction and execution guidance | Revalidate environment and artifact paths before use |
| [Prism-level compression analysis](../../PRISM_LEVEL_COMPRESSION.md) | Separates packing efficiency from quality recovery | Contains historical live-run interpretations |
| [Bonsai ternary forensics](../../bonsai_ternary_deep_forensics.md) | Weight and format forensics | Optimizer identity cannot be inferred from endpoint weights |
| [Prism researcher reconstruction](../../prism_ml_researchers_reconstruction.md) | Public-source research reconstruction | Treat undisclosed training details as hypotheses |

## Frozen evaluation protocol

| Artifact | Role |
| --- | --- |
| [Gate B preregistration](../large_scale_ternary/gate_b_preregistration.json) | Sealed model/control/dataset evaluation contract |
| [Gate B protocol](../large_scale_ternary/gate_b_protocol.py) | Validation and receipt machinery |
| [Gate B model arms](../large_scale_ternary/gate_b_model_arms.py) | Fail-closed arm and checkpoint mapping |
| [Gate B likelihood evaluator](../large_scale_ternary/gate_b_likelihood.py) | Equal-language likelihood/KL implementation |
| [Gate B report comparator](../large_scale_ternary/compare_gate_b_reports.py) | Paired comparison |
| [FineWeb2 test universe](../large_scale_ternary/fineweb2_v210_test_universe.json) | Pinned 1,318 language-script test universe |

## Main implementation surfaces

| Artifact | Role |
| --- | --- |
| [global all-2D trainer](../qwen_agentworld/train_all2d_ternary_qat.py) | Mirror-SGD, factored optimizer, hard-forward global QAT |
| [code-homotopy trainer](../qwen_agentworld/train_code_homotopy_qat.py) | Annealed code-transition experiments |
| [teacher cache collector](../qwen_agentworld/collect_text_teacher_cache.py) | Text teacher-cache construction |
| [multimodal teacher cache collector](../qwen_agentworld/collect_multimodal_teacher_cache.py) | Multimodal cache construction |
| [packed runtime linear](../qwen_agentworld/packed_runtime_linear.py) | Packed-manifest runtime prototype |
| [policy pack smoke](../qwen_agentworld/pack_policy_smoke.py) | Packed-policy smoke verification |
| [ternary KD module](../../clark_quantize/ternary_kd.py) | Hard ternary modules, distillation, signatures, and Muon-fast |
| [Bonsai Muon](../../clark_quantize/bonsai_muon.py) | Original matrix optimizer candidate |
| [BonsaiQAT8](../../clark_quantize/bonsai_qat8.py) | Low-state QAT optimizer candidate |

## Record authority

For a numerical claim, prefer the most specific run report or immutable JSON
receipt over a synthesis document. For current execution state, reopen the
actual process, log, checkpoint, and remote storage; this index is not a live
status surface.
