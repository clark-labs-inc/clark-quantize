# Qwen AgentWorld Clark Air Experiment

Goal: validate Clark Air compression on `Qwen/Qwen-AgentWorld-35B-A3B` using
`Qwen/AgentWorldBench` as the system-performance metric.

This experiment intentionally uses the official Qwen AgentWorld evaluation
harness for inference, judging, and score aggregation. Clark Air code should
change the model artifact or serving endpoint, not the benchmark.

## Hardware Reality

The official Qwen README serves the model at 262k context with tensor
parallelism 4. The current `a100` node is a single A100 80GB. That is enough
for local staging and reduced-context smoke runs, but not a faithful full
262k-context official serving setup.

Use this node for:

- dataset/model staging
- reduced-context baseline smoke
- Clark Air PTQ/QAT prototyping
- AgentWorldBench subset inference
- score aggregation after a confirmed judge is available

Use a larger 4x GPU node for:

- full-context official baseline
- full AgentWorldBench inference at Qwen's recommended deployment settings
- any release-quality compressed-vs-BF16 system-performance claim

Current bounded calibration, packing, and fake-quant endpoint smokes fit on the
single A100. These runs are canaries only. They are not final validation because
they use reduced context and short generations. For faithful full-context Qwen
serving, plan for 4 GPUs rather than 2.

The first packed-runtime smoke also fits on the single A100. It replaces one
whole-tensor linear module with a packed-manifest-backed runtime module and
removes that BF16 weight from the live model parameter graph. It is a memory
accounting proof, not an optimized serving kernel.

## Directory Layout On A100

```text
/workspace/Qwen-AgentWorld                 official eval harness
/workspace/AgentWorldBench                 official dataset snapshot
/workspace/models/Qwen-AgentWorld-35B-A3B  model local-dir
/workspace/agentworld_runs                 outputs
/workspace/clark-quantize                  Clark Air code
```

## Commands

Setup Python dependencies:

```bash
cd /workspace/clark-quantize
bash experiments/qwen_agentworld/setup_a100.sh
```

Setup a full-fidelity 4-GPU validation node:

```bash
cd /workspace/clark-quantize
BACKEND=vllm \
bash experiments/qwen_agentworld/setup_full_fidelity_node.sh
```

This script refuses to run on the current single-A100 node. It expects at
least `4` visible GPUs because the official Qwen-AgentWorld deployment uses
tensor parallelism `4` at the `262144` context boundary.

Download official assets:

```bash
cd /workspace/clark-quantize
bash experiments/qwen_agentworld/download_assets.sh
```

Build a small domain-balanced smoke subset:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/make_subset.py \
  --data-dir /workspace/AgentWorldBench \
  --out-dir /workspace/AgentWorldBench_smoke_1x7 \
  --per-domain 1
```

Serve the model through a local OpenAI-compatible API:

```bash
cd /workspace/clark-quantize
bash experiments/qwen_agentworld/serve_transformers_openai.sh
```

The smoke server disables Qwen's chat-template thinking channel by default.
This better matches the official reasoning-parser deployment boundary for the
benchmark because `message.content` should contain the predicted observation,
not raw reasoning. To reproduce the old behavior explicitly:

```bash
ENABLE_THINKING=1 bash experiments/qwen_agentworld/serve_transformers_openai.sh
```

Run official inference only:

```bash
cd /workspace/clark-quantize
bash experiments/qwen_agentworld/run_official_eval.sh infer smoke_1x7
```

Run the official full-fidelity boundary:

```bash
cd /workspace/clark-quantize
bash experiments/qwen_agentworld/download_assets.sh

JUDGE_API_KEY="$OPENROUTER_API_KEY" \
JUDGE_BASE_URL=https://openrouter.ai/api/v1 \
JUDGE_MODEL=qwen/qwen3.5-flash-02-23 \
bash experiments/qwen_agentworld/run_full_fidelity_eval.sh \
  bf16 full_bf16_262k_32768
```

This uses the official model boundary:

- `MAX_MODEL_LEN=262144`
- `MAX_TOKENS=32768`
- `TP=4`
- full `/workspace/AgentWorldBench` by default

The script refuses to run BF16 full-fidelity mode on the current single-A100
node. The official README documents tensor parallelism 4 for both vLLM and
SGLang at the 256K context boundary. Use the reduced-context smoke runner only
for fast canaries and local allocator/debug iterations.

To evaluate a compressed endpoint at full fidelity, start a full-context
OpenAI-compatible endpoint separately, then use endpoint mode:

```bash
MODEL_BASE_URL=http://127.0.0.1:8000/v1 \
JUDGE_API_KEY="$OPENROUTER_API_KEY" \
JUDGE_BASE_URL=https://openrouter.ai/api/v1 \
JUDGE_MODEL=qwen/qwen3.5-flash-02-23 \
bash experiments/qwen_agentworld/run_full_fidelity_eval.sh \
  endpoint full_compressed_candidate_262k_32768
```

Run a gated binary/ternary smoke candidate:

```bash
cd /workspace/clark-quantize
RUN_JUDGE=0 \
bash experiments/qwen_agentworld/run_binary_ternary_smoke.sh \
  smoke_1x7_moe_ternary_stage256_no_think_64 \
  /workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage256.json
```

Run a BF16 reference through the same runner:

```bash
cd /workspace/clark-quantize
BASELINE_PREDICTIONS= \
RUN_JUDGE=1 \
bash experiments/qwen_agentworld/run_binary_ternary_smoke.sh \
  smoke_3x7_bf16_8192_no_think_64 \
  none
```

The gated runner starts a reduced-context local endpoint, applies the selected
binary/ternary fake-quant policy, runs official inference, then performs cheap
pre-judge checks:

- generation-collapse detection;
- BF16 sequence comparison;
- optional OpenRouter judging only when `RUN_JUDGE=1` and the cheap gates pass.

By default the runner applies only `ternary40_log8` groups. This matches the
current evidence: ternary MoE groups are the quality-preserving surface, while
binary groups are an explicit rate-pressure experiment. To test binary or true
mixed-rate MoE policies, override:

```bash
FAKE_QUANT_RATES=ternary40_log8,binary_log8 \
bash experiments/qwen_agentworld/run_binary_ternary_smoke.sh \
  smoke_1x7_moe_mixed_candidate_no_think_64 \
  /workspace/agentworld_runs/policy_qwen_agentworld_candidate.json
```

For the current AgentWorld judge lane, the user-confirmed judge is OpenRouter
`qwen/qwen3.5-flash-02-23`. Pass the key explicitly:

```bash
JUDGE_API_KEY="$OPENROUTER_API_KEY" \
RUN_JUDGE=1 \
bash experiments/qwen_agentworld/run_binary_ternary_smoke.sh \
  smoke_1x7_moe_ternary_stage256_no_think_64 \
  /workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage256.json
```

Compare two prediction files without a paid judge:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_fake_quant_32groups/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_fake_quant_32groups.json
```

Measure bounded teacher-behavior damage for policy groups without a paid judge:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/measure_group_damage.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --data-dir /workspace/AgentWorldBench_smoke_1x7 \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --out /workspace/agentworld_runs/group_damage_smoke_12groups.json \
  --max-length 256 \
  --limit-jobs 3 \
  --last-n 1 \
  --include-experts \
  --buckets linear_attention_projection,moe_expert \
  --rates ternary40_log8,binary_log8 \
  --max-groups 12 \
  --max-params 2200000
```

Measure selected binary expert-slice damage:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/measure_group_damage.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --data-dir /workspace/AgentWorldBench_smoke_1x7 \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --out /workspace/agentworld_runs/group_damage_smoke_binary_12groups.json \
  --max-length 256 \
  --limit-jobs 3 \
  --last-n 1 \
  --include-experts \
  --buckets moe_expert \
  --rates binary_log8 \
  --max-groups 12 \
  --max-params 2200000
```

Collect router-utilization calibration on the smoke subset:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/collect_router_calibration.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --data-dir /workspace/AgentWorldBench_smoke_1x7 \
  --out /workspace/agentworld_runs/router_calibration_smoke_512.json \
  --max-length 512
```

Build a projected 8x mixed-rate policy:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/model_inventory.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --out /workspace/agentworld_runs/model_inventory.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/build_rate_policy.py \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --router-calibration /workspace/agentworld_runs/router_calibration_smoke_512.json \
  --out /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json
```

Build and compare a corrected measured-damage policy:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/build_rate_policy.py \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --router-calibration /workspace/agentworld_runs/router_calibration_smoke_512.json \
  --damage-measurements \
    /workspace/agentworld_runs/group_damage_smoke_12groups.json \
    /workspace/agentworld_runs/group_damage_smoke_binary_12groups.json \
  --damage-metric kl_mean \
  --out /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_policies.py \
  --base /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --candidate /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out /workspace/agentworld_runs/policy_compare_router_vs_measured_smoke24_monotonic.json
```

The policy builder enforces a measured-damage floor: an unmeasured lower-bit
choice cannot undercut a measured higher-bit choice for the same group.

Run a bounded packer smoke test:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_policy_smoke.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --out-dir /workspace/agentworld_runs/pack_smoke_qwen_agentworld \
  --max-tensors 4 \
  --max-params 12000000
```

Run expert-slice packer smoke tests and verify packed artifacts:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_policy_smoke.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --out-dir /workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts \
  --include-experts \
  --max-tensors 6 \
  --max-params 2200000

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_policy_smoke.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json \
  --out-dir /workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts \
  --include-experts \
  --rate binary_log8 \
  --max-tensors 4 \
  --max-params 2200000

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_smoke_qwen_agentworld_experts/manifest.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_smoke_qwen_agentworld_binary_experts/manifest.json
```

Pack and verify expert slices from the corrected measured policy:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_policy_smoke.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_smoke_qwen_agentworld_measured24_monotonic_experts \
  --include-experts \
  --max-tensors 6 \
  --max-params 2200000

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_smoke_qwen_agentworld_measured24_monotonic_experts/manifest.json
```

Pack and verify parent-tensor mixed-rate artifacts from the corrected measured
policy:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_parent_policy.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_smoke \
  --include-experts \
  --buckets moe_expert \
  --max-parent-tensors 1 \
  --max-groups-per-parent 24 \
  --max-params-per-group 2200000

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_parent_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_smoke/manifest.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_parent_policy.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_binary_smoke \
  --include-experts \
  --buckets moe_expert \
  --rates binary_log8 \
  --max-parent-tensors 1 \
  --max-groups-per-parent 12 \
  --max-params-per-group 2200000

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_parent_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_binary_smoke/manifest.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_parent_policy.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_fp8_smoke \
  --rates fp8 \
  --max-parent-tensors 1 \
  --max-groups-per-parent 0 \
  --max-params-per-group 0

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_parent_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_fp8_smoke/manifest.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_parent_policy.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full_dryrun \
  --include-experts \
  --max-parent-tensors 0 \
  --max-groups-per-parent 0 \
  --max-params-per-group 0 \
  --dry-run
```

`--max-parent-tensors 0`, `--max-groups-per-parent 0`, and
`--max-params-per-group 0` mean "no limit". The full dry run does not load or
write tensor payloads; it verifies that the selected policy surface covers all
parent tensors and groups before launching a full pack.

Run and verify the full corrected measured-policy parent pack:

```bash
cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/pack_parent_policy.py \
  --model-dir /workspace/models/Qwen-AgentWorld-35B-A3B \
  --inventory /workspace/agentworld_runs/model_inventory.json \
  --policy /workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
  --out-dir /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full \
  --include-experts \
  --max-parent-tensors 0 \
  --max-groups-per-parent 0 \
  --max-params-per-group 0 \
  --resume

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/verify_parent_pack_manifest.py \
  --manifest /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
  --out /workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/verify.json
```

Run corrected measured-policy fake-quant inference and compare without a judge:

```bash
cd /workspace/clark-quantize
FAKE_QUANT_POLICY=/workspace/agentworld_runs/policy_qwen_agentworld_8x_measured_smoke24_monotonic.json \
FAKE_QUANT_MAX_GROUPS=32 \
FAKE_QUANT_INCLUDE_EXPERTS=1 \
FAKE_QUANT_RATES=ternary40_log8,binary_log8 \
FAKE_QUANT_BUCKETS=linear_attention_projection,moe_expert \
FAKE_QUANT_REPORT=/workspace/agentworld_runs/fake_quant_report_measured24_monotonic_32groups.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=2048 \
TEMPERATURE=0.6 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_fake_quant_measured24_monotonic_32groups

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_fake_quant_measured24_monotonic_32groups.json
```

Run packed-manifest-loaded inference and compare without a judge:

```bash
cd /workspace/clark-quantize
PACKED_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_MAX_GROUPS=32 \
PACKED_INCLUDE_EXPERTS=1 \
PACKED_RATES=ternary40_log8,binary_log8 \
PACKED_BUCKETS=linear_attention_projection,moe_expert \
PACKED_REPORT=/workspace/agentworld_runs/packed_manifest_report_32groups.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=2048 \
TEMPERATURE=0.6 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_packed_manifest_32groups

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_packed_manifest_32groups.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_fake_quant_measured24_monotonic_32groups/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_fake_quant_vs_packed_manifest_32groups.json
```

This path proves the packed artifact can source endpoint weights. It still
dequantizes into ordinary model parameters, so it is not yet a memory-saving
packed runtime.

Run a first memory-saving packed-runtime linear smoke:

```bash
cd /workspace/clark-quantize
PACKED_RUNTIME_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_RUNTIME_MAX_GROUPS=1 \
PACKED_RUNTIME_RATES=ternary40_log8 \
PACKED_RUNTIME_BUCKETS=linear_attention_projection \
PACKED_RUNTIME_CACHE=cpu \
PACKED_RUNTIME_REPORT=/workspace/agentworld_runs/packed_runtime_report_1linear_cpu.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=2048 \
TEMPERATURE=0.6 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_packed_runtime_1linear_cpu

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_1linear_cpu/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1linear_cpu.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_packed_manifest_32groups/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_1linear_cpu/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_packed_manifest_vs_packed_runtime_1linear_cpu.json
```

The `cache=cpu` mode is deliberately conservative: it proves the module no
longer owns a BF16 weight parameter, but it copies the dequantized CPU cache to
GPU during forward. Use it to validate replacement and memory accounting. Use
`cache=gpu` only for speed experiments, because that restores persistent GPU
weight storage for the replaced module.

Scale the same direct liveness smoke to eight whole-tensor linears:

```bash
cd /workspace/clark-quantize
PACKED_RUNTIME_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_RUNTIME_MAX_GROUPS=8 \
PACKED_RUNTIME_RATES=ternary40_log8 \
PACKED_RUNTIME_BUCKETS=linear_attention_projection \
PACKED_RUNTIME_CACHE=cpu \
PACKED_RUNTIME_REPORT=/workspace/agentworld_runs/packed_runtime_report_8linear_cpu.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-AgentWorld-35B-A3B",
    "messages": [{"role": "user", "content": "Reply with exactly: packed runtime ok"}],
    "max_tokens": 32,
    "temperature": 0
  }' > /workspace/agentworld_runs/packed_runtime_8linear_cpu_direct_smoke.json
```

The 8-linear CPU-cache smoke replaced 8 linears, skipped 0, removed
134,479,872 BF16 parameter bytes from the live module graph, and fit on the
single A100. This is still a runtime-development smoke; full quality validation
must use the official AgentWorld inference/judge/score path.

Run the first fused MoE expert packed-runtime smoke:

```bash
cd /workspace/clark-quantize
PACKED_RUNTIME_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_RUNTIME_MAX_GROUPS=512 \
PACKED_RUNTIME_INCLUDE_EXPERTS=1 \
PACKED_RUNTIME_RATES=bf16,ternary40_log8,binary_log8 \
PACKED_RUNTIME_BUCKETS=moe_expert \
PACKED_RUNTIME_CACHE=cpu \
PACKED_RUNTIME_REPORT=/workspace/agentworld_runs/packed_runtime_report_1expert_layer_cpu.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-AgentWorld-35B-A3B",
    "messages": [{"role": "user", "content": "Reply with exactly: expert packed runtime ok"}],
    "max_tokens": 32,
    "temperature": 0
  }' > /workspace/agentworld_runs/packed_runtime_1expert_layer_cpu_direct_smoke.json

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=256 \
TEMPERATURE=0 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_packed_runtime_1expert_layer_cpu_256

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_1expert_layer_cpu_256/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_1expert_layer_cpu_256.json
```

The first fused expert smoke replaced one `Qwen3_5MoeExperts` module, skipped
0, removed 1,610,612,736 BF16 parameter bytes from the live module graph, and
completed the official 7-domain inference smoke at `MAX_TOKENS=256`.
`PACKED_RUNTIME_RATES` must include `bf16` for this path because the runtime
needs every expert slice in the fused module, including protected BF16 slices.

Scale the fused expert smoke to four MoE layers:

```bash
cd /workspace/clark-quantize
PACKED_RUNTIME_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_RUNTIME_MAX_GROUPS=2048 \
PACKED_RUNTIME_INCLUDE_EXPERTS=1 \
PACKED_RUNTIME_RATES=bf16,ternary40_log8,binary_log8 \
PACKED_RUNTIME_BUCKETS=moe_expert \
PACKED_RUNTIME_CACHE=cpu \
PACKED_RUNTIME_REPORT=/workspace/agentworld_runs/packed_runtime_report_4expert_layers_cpu.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=256 \
TEMPERATURE=0 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_packed_runtime_4expert_layers_cpu_256

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_cpu_256/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_vs_packed_runtime_4expert_layers_cpu_256.json
```

The four-layer CPU-cache smoke replaced 4 fused expert modules, skipped 0,
removed 6,442,450,944 BF16 parameter bytes, and completed the official
7-domain smoke at `MAX_TOKENS=256` in about 3 minutes. This confirms runtime
replacement scales beyond one expert layer on a single A100. The remaining
runtime problem is throughput, not replacement correctness.

Run the same four-layer fused expert smoke with active expert GPU caching:

```bash
cd /workspace/clark-quantize
PACKED_RUNTIME_MANIFEST=/workspace/agentworld_runs/pack_parent_qwen_agentworld_measured24_monotonic_full/manifest.json \
PACKED_RUNTIME_MAX_GROUPS=2048 \
PACKED_RUNTIME_INCLUDE_EXPERTS=1 \
PACKED_RUNTIME_RATES=bf16,ternary40_log8,binary_log8 \
PACKED_RUNTIME_BUCKETS=moe_expert \
PACKED_RUNTIME_CACHE=active \
PACKED_RUNTIME_ACTIVE_CACHE_LIMIT=64 \
PACKED_RUNTIME_REPORT=/workspace/agentworld_runs/packed_runtime_report_4expert_layers_active64.json \
MAX_MODEL_LEN=8192 \
bash experiments/qwen_agentworld/serve_transformers_openai.sh

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 \
MAX_TOKENS=256 \
TEMPERATURE=0 \
bash experiments/qwen_agentworld/run_official_eval.sh infer \
  smoke_1x7_packed_runtime_4expert_layers_active64_256

curl -s http://127.0.0.1:8000/v1/models \
  > /workspace/agentworld_runs/packed_runtime_stats_4expert_layers_active64_after_smoke.json

/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_cpu_256/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_256/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_cpu_vs_active64_packed_runtime_4expert_layers_256.json
```

The active-cache smoke produced byte-identical predictions to the CPU-cache
4-layer smoke, finished the 7-domain bounded inference in about 2.5 minutes,
and used 1,610,612,736 bytes of active expert GPU cache after the run.

For corrected no-thinking runtime validation, use separate run names and compare
against the no-thinking BF16 256-token baseline:

```bash
cd /workspace/clark-quantize/experiments/qwen_agentworld
DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 MAX_TOKENS=256 TEMPERATURE=0 \
  ./run_official_eval.sh infer smoke_1x7_bf16_8192_no_think_256

DATA_DIR=/workspace/AgentWorldBench_smoke_1x7 MAX_TOKENS=256 TEMPERATURE=0 \
  ./run_official_eval.sh infer smoke_1x7_packed_runtime_4expert_layers_active64_no_think_256

cd /workspace/clark-quantize
/workspace/agentworld-venv/bin/python experiments/qwen_agentworld/compare_predictions.py \
  --baseline /workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think_256/predictions.jsonl \
  --candidate /workspace/agentworld_runs/smoke_1x7_packed_runtime_4expert_layers_active64_no_think_256/predictions.jsonl \
  --out /workspace/agentworld_runs/compare_bf16_no_think_256_vs_packed_runtime_4expert_layers_active64_no_think_256.json
```

Run the confirmed OpenRouter Qwen 3.5 Flash judge smoke. Keep the API key in
the environment; do not pass it as a command-line argument.

```bash
cd /workspace/clark-quantize
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=qwen/qwen3.5-flash-02-23
export JUDGE_API_KEY=<openrouter-key-from-env>
export JUDGE_MAX_TOKENS=4096
export JUDGE_TEMPERATURE=0

for run in \
  smoke_1x7_bf16_8192 \
  smoke_1x7_packed_manifest_32groups \
  smoke_1x7_packed_runtime_4expert_layers_cpu_256
do
  bash experiments/qwen_agentworld/run_official_eval.sh judge "$run" \
    > "/workspace/agentworld_runs/logs/judge_${run}_openrouter_qwen35_flash.log" 2>&1
  bash experiments/qwen_agentworld/run_official_eval.sh score "$run" \
    > "/workspace/agentworld_runs/${run}/score_openrouter_qwen35_flash.txt" 2>&1
done
```

## Validation Gates

1. Baseline endpoint answers the 7-domain smoke subset.
2. Official `eval.py infer` produces `predictions.jsonl`.
3. Confirmed judge scores the same predictions.
4. Clark Air compressed endpoint repeats the same subset.
5. Compare:
   - AgentWorldBench overall
   - per-domain scores
   - valid/failed count
   - exact packed bytes
   - compression ratio versus BF16 model size

## Clark Air Compression Plan

Phase 0: PTQ inventory and bit ledger.

- inspect safetensors index
- classify dense attention, MoE expert, router, norm, embedding, output tensors
- compute BF16 bytes and target 8x bytes

Phase 1: Fisher/rate allocation.

- estimate diagonal Fisher on teacher KL from AgentWorldBench calibration
- weight MoE expert groups by router utilization when available
- choose binary/ternary/FP8 groups by water-filling
- current smoke policy uses router utilization plus proxy damage; replace the
  proxy damage with empirical Fisher as the next measurement upgrade

Phase 2: runnable compressed endpoint.

- replace target linear tensors with Clark Air hard-forward modules
- validate generation parity on smoke prompts
- run QAT/polish only after baseline and PTQ policy are measurable
- full MoE expert-slice packing is still needed; current pack smoke verifies
  individual whole-tensor and expert-slice artifacts

Phase 3: full benchmark.

- run official inference on full AgentWorldBench
- judge with a confirmed judge model/provider
- aggregate and compare to BF16 baseline
