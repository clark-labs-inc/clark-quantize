# Artifact index

Paths beginning with `../browser_swin2sr/` are canonical. Do not relocate a
sealed artifact merely to make the directory tree prettier: its recorded path,
hash, and manifest are part of the evidence.

## Program state and acceptance

- current live boundary: `CURRENT_STATE.md`
- requirement-by-requirement closing ledger: `ACCEPTANCE_LEDGER.md`
- chronological findings and corrections: `SCIENTIFIC_JOURNAL.md`
- provenance-preserving folder plan: `FOLDER_MIGRATION.md`

## Scientist and research-agent records

- Clark Scientist run registry: `scientist_runs/README.md`
- exact DeepSeek 07/31 structured proposal and provenance receipt:
  `scientist_runs/20260731T180026Z-deepseek-swin-scientist/proposal.json`,
  `scientist_runs/20260731T180026Z-deepseek-swin-scientist/run-receipt.json`
- complete DeepSeek 07/31 Clark Scientist run, including provider responses,
  reasoning details, worker configuration, research events, and trajectory:
  `scientist_runs/20260731T180026Z-deepseek-swin-scientist/raw/`
- bounded research-agent review registry and content ledger:
  `research_agent_reviews/README.md`, `research_agent_reviews/manifest.json`
- Prism/Bonsai precision review:
  `research_agent_reviews/20260731-bonsai-prisma-primary.md`
- Clark Scientist runtime safety review:
  `research_agent_reviews/20260731-clark-scientist-runtime-audit.md`
- exact Swin2SR deployed-precision review:
  `research_agent_reviews/20260731-swin-precision-map.md`
- earlier distilled Scientist receipt and study context:
  `../browser_swin2sr/prism_bonsai_transfer_20260731/SCIENTIST_DEEPSEEK_0731.md`,
  `../browser_swin2sr/prism_bonsai_transfer_20260731/SCIENTIFIC_JOURNAL.md`

Credentials, caches, unrelated machine-private data, and complete Codex session
databases are intentionally excluded. Codex sessions are not Clark Scientist
model artifacts; their captured research reports are indexed above.

## Current selected still-image release: G3

- experiment protocol and chronological evidence:
  `../browser_swin2sr/joint_bonsai_recovery_20260801/PROTOCOL.md`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/SCIENTIFIC_JOURNAL.md`
- selected mixed-precision development metrics and comparisons:
  `../browser_swin2sr/joint_bonsai_recovery_20260801/int7_per_output/development/metrics.json`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/int7_per_output/vs_release.json`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/int7_per_output/vs_ternary.json`
- canonical public-package metadata and exact artifact hashes:
  `../browser_swin2sr/huggingface/clark-swin2sr-lightweight-x2-1.58bit/technical/quantization.json`
- native bundle ledger and portable ONNX ledger:
  `../browser_swin2sr/joint_bonsai_recovery_20260801/int7_per_output/browser_artifact_v4/bundle-ledger.json`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/int7_per_output/browser_artifact_v4/portable_onnx/model.ledger.json`
- public package and non-blocking Space source:
  `../browser_swin2sr/huggingface/clark-swin2sr-lightweight-x2-1.58bit/`,
  `../browser_swin2sr/huggingface/spaces/clark-swin2sr-lightweight-x2-1.58bit/`
- rejected post-G3 fixed-scale and bias runs:
  `../browser_swin2sr/joint_bonsai_recovery_20260801/fixed_int7_scales_mask7_seed20260807/`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/fixed_int7_scales_mask0_seed20260807/`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/fixed_int7_scales_mask7_lr5e5_seed20260807/`,
  `../browser_swin2sr/joint_bonsai_recovery_20260801/rstb_bias_seed20260808/`

## Current realtime-video release and data study

- realtime study journal and source:
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/SCIENTIFIC_JOURNAL.md`,
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/realtime_student.py`
- selected RGB M5 runtime receipt:
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/runtime-receipt-rgb-20260801.json`
- public package ledger and Seedance action evaluation:
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/huggingface/clark-realtime-ui-video-upscaler-x2/technical/package-ledger.json`,
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/huggingface/clark-realtime-ui-video-upscaler-x2/technical/seedance-action-video-evaluation.json`
- rejected 14-channel temporal arm:
  `../browser_swin2sr/realtime_1080p_to_4k_20260731/run_temporal_14ch_seed20260809/`
- rights and source policy, video journal, and pinned index receipt:
  `../browser_swin2sr/video_training_data_20260801/source-manifest.json`,
  `../browser_swin2sr/video_training_data_20260801/SCIENTIFIC_JOURNAL.md`,
  `../browser_swin2sr/video_training_data_20260801/index/markov-screen-recording-index-receipt.json`

## Baselines and sealed evaluation

- q4 full baseline: `../browser_swin2sr/q4-baseline-receipt.json`
- q4 glyph-safe baseline: `../browser_swin2sr/q4-glyphsafe-baseline-receipt.json`
- heldout construction and isolation:
  `../browser_swin2sr/glyphsafe-heldout-receipt.json`
- active promotion policy for future candidates:
  `../browser_swin2sr/promotion_policy_glyphsafe_v10.json`
- policy amendments: `policies/`

## Method and study definitions

- Clark-to-Swin2SR method map: `methods/BACKDROPKIT_METHOD_MAP.md`
- sensitive curriculum preregistration:
  `studies/SENSITIVE_HARDSHADOW_STUDY_20260730.md`
- sensitive input-convolution rescue preregistration:
  `studies/SENSITIVE_IO_RESCUE_STUDY_20260730.md`
- full-surface optimizer-transfer preregistration:
  `studies/FULL_SURFACE_OPTIMIZER_TRANSFER_STUDY_20260730.md`
- optimizer-transfer source and command sealer:
  `../browser_swin2sr/seal_optimizer_transfer_study.py`
- optimizer-transfer all-hard stratified evaluator:
  `../browser_swin2sr/evaluate_optimizer_transfer_stratified.py`
- optimizer-transfer canonical-state development selector:
  `../browser_swin2sr/decide_optimizer_transfer.py`
- canonical dense/runtime state materializer:
  `../browser_swin2sr/materialize_surface_states.py`
- output-convolution group-size preregistration:
  `studies/OUTPUT_CONV_GROUP_PROBE_20260730.md`
- matched output-convolution rescue preregistration:
  `studies/OUTPUT_CONV_RESCUE_STUDY_20260730.md`
- implementation overview: `../browser_swin2sr/README.md`

## Optimizer evidence

- numerical smoke:
  `../browser_swin2sr/optimizer_smoke_20260730/results.json`
- periodic-path smoke:
  `../browser_swin2sr/optimizer_periodic_smoke_20260730/summary-receipt.json`
- sealed 27-arm development tournament:
  `../browser_swin2sr/optimizer_development_20260730/manifest.json`
- ranked tournament receipt:
  `../browser_swin2sr/optimizer_development_20260730/summary-receipt.json`
- completed 1,000-step replications:
  `../browser_swin2sr/optimizer_replicate_seed20260730/`,
  `../browser_swin2sr/optimizer_replicate_seed20260731/`,
  `../browser_swin2sr/optimizer_replicate_seed20260732/`
- paired analyzer:
  `../browser_swin2sr/aggregate_optimizer_replicates.py`
- preregistered low-state decision rule:
  `../browser_swin2sr/decide_optimizer_replicates.py`
- completed paired ground-truth aggregate:
  `../browser_swin2sr/optimizer_replicates_20260730/ground-truth-aggregate.json`
- completed development recipe decision:
  `../browser_swin2sr/optimizer_replicates_20260730/decision.json`
- reusable optimizer implementations:
  `../../clark_quantize/qat_optimizer_variants.py`
- full-surface optimizer and CLI contracts:
  `../browser_swin2sr/full_surface_optimizer.py`,
  `../browser_swin2sr/full_surface_cli.py`

## Objective and surface-recovery evidence

- completed pixel-objective decision:
  `../browser_swin2sr/pixel_objective_probe_20260731/pixel-objective-decision.json`
- selected PSNR-aligned checkpoint:
  `../browser_swin2sr/pixel_objective_probe_20260731/psnr_aligned/run/checkpoint-final.pt`
- sealed matched surface-recovery manifest:
  `../browser_swin2sr/surface_group_recovery_20260731/study-manifest.json`

## Sensitive-I/O rescue

- preregistration: `studies/SENSITIVE_IO_RESCUE_STUDY_20260730.md`
- active study root:
  `../browser_swin2sr/sensitive_io_rescue_20260730/`
- group-size and optimizer-hyperparameter checkpoint fork:
  `../browser_swin2sr/fork_sensitive_io_checkpoint.py`
- source and command sealer:
  `../browser_swin2sr/seal_sensitive_io_rescue.py`
- preregistered checkpoint-gate selector:
  `../browser_swin2sr/decide_sensitive_io_rescue.py`
- fail-closed exact-arm coverage validator:
  `../browser_swin2sr/validate_sensitive_io_rescue_coverage.py`
- sealed-source checkpoint continuation runner:
  `../browser_swin2sr/run_sealed_sensitive_resume.py`
- sealed-source continuation receipt verifier:
  `../browser_swin2sr/verify_sealed_sensitive_resume.py`
- group-2, 0.25x convolution LR:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/`
- group-2, 1x convolution LR:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/`
- group-2, 4x convolution LR:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/`
- matched restarted group-9, 0.25x control:
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/`
- matched step-15 hard-started receipts:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/hard-started-validation-step-15000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/hard-started-validation-step-15000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/hard-started-validation-step-15000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/hard-started-validation-step-15000-mps.json`
- matched step-15 ternary signatures:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/signatures-step-15000.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/signatures-step-15000.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/signatures-step-15000.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/signatures-step-15000.json`
- matched step-16 hard-started receipts and signatures:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/hard-started-validation-step-16000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/hard-started-validation-step-16000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/hard-started-validation-step-16000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/hard-started-validation-step-16000-mps.json`,
  and the corresponding `signatures-step-16000.json` files in each arm.
- matched step-17 hard-started receipts and signatures:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/hard-started-validation-step-17000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/hard-started-validation-step-17000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/hard-started-validation-step-17000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/hard-started-validation-step-17000-mps.json`,
  and the corresponding `signatures-step-17000.json` files in each arm.
- sealed full-development gate with complete local source closure:
  `../browser_swin2sr/sensitive_io_rescue_20260730/development-gate-manifest-v3.json`
- superseded provenance-incomplete v2 gate:
  `../browser_swin2sr/sensitive_io_rescue_20260730/development-gate-manifest-v2.json`
- pre-outcome v1 invalidation receipt:
  `../browser_swin2sr/sensitive_io_rescue_20260730/development-gate-manifest-v1-invalidated.json`
- staged full-bundle evaluator and frozen selector:
  `../browser_swin2sr/evaluate_hard_started_stratified.py`,
  `../browser_swin2sr/decide_sensitive_io_development.py`
- completed exact step-19 receipts:
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/hard-started-validation-step-19000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/resume-step17000-to19000-verification.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/hard-started-validation-step-19000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr1/resume-step17000-to19000-verification.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/hard-started-validation-step-19000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr025/resume-step17000-to19000-verification.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/hard-started-validation-step-19000-mps.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/resume-step17000-to19000-verification.json`
- exact step-19 coverage and checkpoint decision:
  `../browser_swin2sr/sensitive_io_rescue_20260730/step19-exact-arm-coverage.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/step19-checkpoint-decision.json`
- full-development input reports and selected recipe:
  `../browser_swin2sr/sensitive_io_rescue_20260730/restart-g9-lr025/full-development/metrics.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/io-g2-lr4/full-development/metrics.json`,
  `../browser_swin2sr/sensitive_io_rescue_20260730/sensitive-io-development-decision.json`
- sealed output-convolution study root and development gate:
  `../browser_swin2sr/output_conv_rescue_20260730/`,
  `../browser_swin2sr/output_conv_rescue_20260730/development-gate-manifest.json`
- completed all-hard output-g9 control checkpoint and full-development report:
  `../browser_swin2sr/output_conv_rescue_20260730/output-g9/run/checkpoint-surface-24000.pt`,
  `../browser_swin2sr/output_conv_rescue_20260730/output-g9/full-development/metrics.json`
- selected all-hard output-g2 checkpoint, full-development report, and frozen
  output decision:
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/run/checkpoint-final.pt`,
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/full-development/metrics.json`,
  `../browser_swin2sr/output_conv_rescue_20260730/output-conv-development-decision.json`
- active sealed QAT8-versus-AdamW optimizer transfer:
  `../browser_swin2sr/optimizer_transfer_20260730/study-manifest.json`,
  `../browser_swin2sr/optimizer_transfer_20260730/`
- confirmed three-surface intervention and disjoint joint evidence:
  `../browser_swin2sr/surface_group_confirmation_20260731/decision.json`,
  `../browser_swin2sr/surface_group_confirmation_20260731/joint-disjoint-256.json`
- frozen three-surface matched recovery protocol and queued study root:
  `../browser_swin2sr/surface_group_recovery_protocol_20260731/protocol.json`,
  `../browser_swin2sr/surface_group_recovery_20260731/`
- broader 14-surface diagnostic, full-development report, and rejection:
  `../browser_swin2sr/surface_group_broad_confirmation_20260731/expanded-surface-map.json`,
  `../browser_swin2sr/surface_group_broad_confirmation_20260731/full-development/development/metrics.json`,
  `../browser_swin2sr/surface_group_broad_confirmation_20260731/full-development/decision.json`
- zero-parameter residual-gain protocol and rejected result:
  `../browser_swin2sr/residual_gain_probe_20260731/protocol.json`,
  `../browser_swin2sr/residual_gain_probe_20260731/result.json`
- packed mixed-group runtime infrastructure receipt:
  `../browser_swin2sr/mixed-group-runtime-smoke-receipt-20260730.json`
- dense portable-ONNX WebGPU/WASM infrastructure receipt:
  `../browser_swin2sr/full-surface-portable-onnx-chromium-runtime.json`

## Full-surface hard-shadow control

- run folder: `../browser_swin2sr/sensitive_hardshadow_20260730/`
- sealed manifest:
  `../browser_swin2sr/sensitive_hardshadow_20260730/study-manifest.json`
- command:
  `../browser_swin2sr/sensitive_hardshadow_20260730/command.json`
- hard-start diagnostics through 8,000:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-through-8000.json`
- step-10,000 diagnostic:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-10000.json`
- step-12,000 diagnostic:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-12000.json`
- step-12,000 ternary signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-12000.json`
- step-14,000 MPS hard-started validation:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-14000-mps.json`
- step-14,000 CPU reprojection diagnostic:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-14000.json`
- step-14,000 ternary signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-14000.json`
- sealed source snapshot:
  `../browser_swin2sr/sensitive_hardshadow_20260730/sealed_sources/`
- exact-hash sealed-source reconstruction receipt:
  `../browser_swin2sr/sensitive_hardshadow_20260730/sealed_sources/reconstruction-receipt.json`
- fail-closed reconstruction utility:
  `../browser_swin2sr/reconstruct_sealed_control_sources.py`
- step-18,000 MPS hard-started validation:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-18000-mps.json`
- step-18,000 ternary signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-18000.json`
- step-19,000 MPS hard-started validation and signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-19000-mps.json`,
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-19000.json`
- canonical step-14,000 materialization:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/materialization.json`
- canonical dense hard state:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/dense-hard.pt`
- canonical runtime hard state:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/runtime-hard.pt`
- packed browser artifact and verification:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/packed/`,
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/packed-verification.json`
- portable ONNX artifact and WASM receipt:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/portable/`,
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/portable/wasm-receipt.json`
- canonical Chromium WebGPU/WASM receipt:
  `../browser_swin2sr/sensitive_hardshadow_20260730/canonical-step-14000-v1/portable/chromium-browser-receipt.json`
- step-15,000 MPS hard-started validation:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-15000-mps.json`
- step-15,000 ternary signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-15000.json`
- step-16,000 MPS hard-started validation and signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-16000-mps.json`,
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-16000.json`
- step-17,000 MPS hard-started validation and signatures:
  `../browser_swin2sr/sensitive_hardshadow_20260730/hard-started-validation-step-17000-mps.json`,
  `../browser_swin2sr/sensitive_hardshadow_20260730/signatures-step-17000.json`
- sensitive I/O scale-group probe and refined receipt:
  `../browser_swin2sr/probe_sensitive_io_group_sizes.py`,
  `../browser_swin2sr/sensitive_io_group_probe_20260730/step16000-first-convolution-refined.json`,
  `../browser_swin2sr/sensitive_io_group_probe_20260730/step17000-first-convolution.json`
- sensitive I/O visual comparison:
  `../browser_swin2sr/render_sensitive_io_comparison.py`,
  `../browser_swin2sr/sensitive_io_group_probe_20260730/visual-step16000/receipt.json`
- step-20,000 output-convolution external probe:
  `../browser_swin2sr/sensitive_io_group_probe_20260730/step20000-output-convolution.json`
- step-20,000 output g9/g3/g2 visual comparison:
  `../browser_swin2sr/sensitive_io_group_probe_20260730/visual-step20000-output-optimal/receipt.json`
- preregistered output full-development selector:
  `../browser_swin2sr/decide_output_conv_development.py`
- output-specific immutable stratified evaluator:
  `../browser_swin2sr/evaluate_output_conv_stratified.py`
- pre-outcome output development-gate sealer:
  `../browser_swin2sr/seal_output_conv_development_gate.py`
- output-specific matched-arm source and command sealer:
  `../browser_swin2sr/seal_output_conv_rescue.py`
- pre-outcome periodic hard-start/signature diagnostics plan:
  `studies/OUTPUT_CONV_PERIODIC_DIAGNOSTICS_PLAN_20260730.json`
- proposed post-selection data-scale and realistic-degradation ablation:
  `studies/DATA_SCALE_AND_DEGRADATION_STUDY_20260730.md`

## Browser-runtime infrastructure

- preregistered g2 scale-codebook protocol and rejected result:
  `../browser_swin2sr/scale_codec_probe_20260731/protocol.json`,
  `../browser_swin2sr/scale_codec_probe_20260731/result.json`
- deterministic scale codec implementation and focused tests:
  `../browser_swin2sr/scale_codebook.py`,
  `../browser_swin2sr/test_scale_codebook.py`,
  `../browser_swin2sr/evaluate_scale_codebooks.py`,
  `../browser_swin2sr/test_evaluate_scale_codebooks.py`
- selected g2 independent packed FP16-scale artifact and 16x16 runtime fixture:
  `../browser_swin2sr/scale_codec_probe_20260731/baseline-fp16/model.json`,
  `../browser_swin2sr/scale_codec_probe_20260731/baseline-fp16/model.bin`,
  `../browser_swin2sr/scale_codec_probe_20260731/runtime-fixture-16.json`
- adaptive SIMD-to-scalar WASM loader and performance receipt:
  `../browser_swin2sr/runtime/packedRuntime.js`,
  `../browser_swin2sr/runtime/full_runtime_auto.html`,
  `../browser_swin2sr/scale_codec_probe_20260731/packed-runtime-performance-receipt.json`

- selected g2 product canonical materialization:
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/product-release-canonical/materialization.json`
- selected g2 portable product package and manifest inputs:
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/product-release/model.onnx`,
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/product-release/model.onnx.gz`
- selected g2 128x128 canonical runtime fixture:
  `../browser_swin2sr/output_conv_rescue_20260730/output-g2/product-release/runtime-fixture-128.json`
- BackdropKit production commits and deployment:
  `18dbfc065`, `9a996e648`,
  Cloudflare version `9e77edd4-74f2-4d07-b67d-fdb7ee8ca3a3`
- product package identity:
  683,547 bytes,
  SHA-256 `d58ba816c4ac5cd3db7edd3c61068619695d233f7c4d5dd2984da52b7bde2de9`

- full-surface WebGPU/WASM receipt:
  `../browser_swin2sr/full-surface-portable-onnx-smoke-receipt.json`
- compressed-package fallback receipt:
  `../browser_swin2sr/compressed-portable-onnx-infrastructure-chromium-receipt-20260730.json`
- promotion-shape Chromium receipt:
  `../browser_swin2sr/mixed-group-portable-onnx-promotion-shape-chromium-receipt-20260730.json`
- materialized MPS-to-CPU parity diagnostic:
  `../browser_swin2sr/sensitive_hardshadow_20260730/materialized-device-parity-step-14000.json`
- reusable materialized-state parity verifier:
  `../browser_swin2sr/verify_materialized_device_parity.py`
- canonical materializer:
  `../browser_swin2sr/materialize_surface_states.py`
- packed exporter:
  `../browser_swin2sr/export_browser_artifact.py`
- portable ONNX exporter:
  `../browser_swin2sr/build_portable_onnx_artifact.py`
- optimizer-selected canonical artifact preparer:
  `../browser_swin2sr/prepare_optimizer_final_candidate.py`
- shared canonical artifact identity validator:
  `../browser_swin2sr/final_artifact_contract.py`
- optimizer-selected sealed final evaluator:
  `../browser_swin2sr/evaluate_optimizer_final_candidate.py`
- canonical hard-state runtime-fixture generator:
  `../browser_swin2sr/prepare_full_onnx_smoke.py`
- cross-browser receipt merger:
  `../browser_swin2sr/merge_runtime_receipts.py`
- final paired, stratified, size, and runtime promotion gate:
  `../browser_swin2sr/evaluate_promotion.py`
- optimizer-selected final receipt and promotion authorizer:
  `../browser_swin2sr/decide_optimizer_final_promotion.py`
- runtime implementation: `../browser_swin2sr/runtime/`
- gradual scale-codebook QAT:
  `../browser_swin2sr/scale_codebook_qat.py`,
  `../browser_swin2sr/train_scale_codebook_recovery.py`
- queued matched scale-QAT sealer and fail-closed decision:
  `../browser_swin2sr/seal_scale_codebook_qat_study.py`,
  `../browser_swin2sr/decide_scale_codebook_qat.py`
- scale-codebook checkpoint/materialization contract:
  `../browser_swin2sr/scale_codebook_state.py`,
  `../browser_swin2sr/materialize_scale_codebook.py`
- exact chained bridge-to-q1 implementation and state contract:
  `../browser_swin2sr/staged_scale_codebook.py`,
  `../browser_swin2sr/train_staged_scale_codebook_recovery.py`,
  `../browser_swin2sr/staged_scale_codebook_state.py`,
  `../browser_swin2sr/materialize_staged_scale_codebook.py`
- prepared fail-closed staged-collapse sealer and decision:
  `../browser_swin2sr/seal_staged_scale_codebook_study.py`,
  `../browser_swin2sr/decide_staged_scale_codebook.py`
- bit-packed scale exporter and v3 runtime coverage:
  `../browser_swin2sr/export_browser_artifact.py`,
  `../browser_swin2sr/runtime/src/scale_codec.rs`,
  `../browser_swin2sr/runtime/tests/artifact_v3.rs`
- deterministic single-file gzip transport and browser decoder:
  `../browser_swin2sr/pack_browser_v3_bundle.py`,
  `../browser_swin2sr/runtime/packedBundle.js`,
  `../browser_swin2sr/runtime/packedRuntime.js`
- packed-q1 final candidate, locked evaluator, fixture, and promotion gate:
  `../browser_swin2sr/prepare_scale_final_candidate.py`,
  `../browser_swin2sr/evaluate_scale_final_candidate.py`,
  `../browser_swin2sr/prepare_scale_runtime_fixture.py`,
  `../browser_swin2sr/decide_scale_final_promotion.py`
- preregistered category-scale sensitivity diagnostic:
  `../browser_swin2sr/mixed_scale_map_probe_20260731/protocol.json`,
  `../browser_swin2sr/mixed_scale_map_probe_20260731/result.json`
- mixed-category scale reconstruction and evaluator:
  `../browser_swin2sr/mixed_scale_codebook.py`,
  `../browser_swin2sr/evaluate_mixed_scale_maps.py`
- runtime surface contract and tests:
  `../browser_swin2sr/runtime_surface.mjs`,
  `../browser_swin2sr/runtime_surface.test.mjs`

All runtime artifacts in this section are infrastructure evidence unless their
receipt explicitly names the eventual heldout quality winner.

## Core implementation

- core QAT trainer: `../browser_swin2sr/train_swin2sr_qat.py`
- full-surface trainer:
  `../browser_swin2sr/train_swin2sr_full_surface.py`
- optimizer controls: `../browser_swin2sr/optimizer_controls.py`
- surface quantization:
  `../browser_swin2sr/surface_quantization.py`
- sensitive schedule: `../browser_swin2sr/surface_schedule.py`
- hard-shadow curriculum:
  `../browser_swin2sr/surface_curriculum.py`
- objectives: `../browser_swin2sr/qat_objective.py`
- run-state and provenance:
  `../browser_swin2sr/run_state.py`,
  `../browser_swin2sr/surface_run_state.py`,
  `../browser_swin2sr/full_surface_provenance.py`
- tournament runner:
  `../browser_swin2sr/run_optimizer_tournament.py`
- tournament summarizer:
  `../browser_swin2sr/summarize_optimizer_tournament.py`

## Historical and exploratory runs

Older `qat_*`, factorial, recovery, binary, and convolution-grouping folders
remain under `../browser_swin2sr/`. They are retained because negative results
and resume provenance are scientifically useful. They should be classified in
the journal before any archival move or deletion; absence from the promotion
path does not make them disposable.

The organization and deferred-move rationale are recorded in
`FOLDER_MIGRATION.md`.
