#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/workspace/agentworld-venv}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen-AgentWorld-35B-A3B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

args=(
  experiments/qwen_agentworld/serve_transformers_openai.py
  --model-path "$MODEL_DIR"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ -n "${ENABLE_THINKING:-}" ]; then
  args+=(--enable-thinking)
fi
if [ -n "${WEIGHT_PATCH:-}" ]; then
  args+=(--weight-patch "$WEIGHT_PATCH")
fi
if [ -n "${WEIGHT_PATCH_REPORT:-}" ]; then
  args+=(--weight-patch-report "$WEIGHT_PATCH_REPORT")
fi
if [ -n "${FAKE_QUANT_POLICY:-}" ]; then
  args+=(--fake-quant-policy "$FAKE_QUANT_POLICY")
fi
if [ -n "${FAKE_QUANT_MAX_GROUPS:-}" ]; then
  args+=(--fake-quant-max-groups "$FAKE_QUANT_MAX_GROUPS")
fi
if [ -n "${FAKE_QUANT_INCLUDE_EXPERTS:-}" ]; then
  args+=(--fake-quant-include-experts)
fi
if [ -n "${FAKE_QUANT_RATES:-}" ]; then
  args+=(--fake-quant-rates "$FAKE_QUANT_RATES")
fi
if [ -n "${FAKE_QUANT_BUCKETS:-}" ]; then
  args+=(--fake-quant-buckets "$FAKE_QUANT_BUCKETS")
fi
if [ -n "${FAKE_QUANT_REPORT:-}" ]; then
  args+=(--fake-quant-report "$FAKE_QUANT_REPORT")
fi
if [ -n "${PACKED_MANIFEST:-}" ]; then
  args+=(--packed-manifest "$PACKED_MANIFEST")
fi
if [ -n "${PACKED_MAX_GROUPS:-}" ]; then
  args+=(--packed-max-groups "$PACKED_MAX_GROUPS")
fi
if [ -n "${PACKED_INCLUDE_EXPERTS:-}" ]; then
  args+=(--packed-include-experts)
fi
if [ -n "${PACKED_RATES:-}" ]; then
  args+=(--packed-rates "$PACKED_RATES")
fi
if [ -n "${PACKED_BUCKETS:-}" ]; then
  args+=(--packed-buckets "$PACKED_BUCKETS")
fi
if [ -n "${PACKED_REPORT:-}" ]; then
  args+=(--packed-report "$PACKED_REPORT")
fi
if [ -n "${PACKED_RUNTIME_MANIFEST:-}" ]; then
  args+=(--packed-runtime-manifest "$PACKED_RUNTIME_MANIFEST")
fi
if [ -n "${PACKED_RUNTIME_MAX_GROUPS:-}" ]; then
  args+=(--packed-runtime-max-groups "$PACKED_RUNTIME_MAX_GROUPS")
fi
if [ -n "${PACKED_RUNTIME_INCLUDE_EXPERTS:-}" ]; then
  args+=(--packed-runtime-include-experts)
fi
if [ -n "${PACKED_RUNTIME_RATES:-}" ]; then
  args+=(--packed-runtime-rates "$PACKED_RUNTIME_RATES")
fi
if [ -n "${PACKED_RUNTIME_BUCKETS:-}" ]; then
  args+=(--packed-runtime-buckets "$PACKED_RUNTIME_BUCKETS")
fi
if [ -n "${PACKED_RUNTIME_CACHE:-}" ]; then
  args+=(--packed-runtime-cache "$PACKED_RUNTIME_CACHE")
fi
if [ -n "${PACKED_RUNTIME_ACTIVE_CACHE_LIMIT:-}" ]; then
  args+=(--packed-runtime-active-cache-limit "$PACKED_RUNTIME_ACTIVE_CACHE_LIMIT")
fi
if [ -n "${PACKED_RUNTIME_ACTIVE_CPU_CACHE_LIMIT:-}" ]; then
  args+=(--packed-runtime-active-cpu-cache-limit "$PACKED_RUNTIME_ACTIVE_CPU_CACHE_LIMIT")
fi
if [ -n "${PACKED_RUNTIME_REPORT:-}" ]; then
  args+=(--packed-runtime-report "$PACKED_RUNTIME_REPORT")
fi

exec "$VENV/bin/python" "${args[@]}"
