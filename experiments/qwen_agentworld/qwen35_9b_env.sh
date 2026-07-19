#!/usr/bin/env bash

# Source this file on a node before running the AgentWorld/Clark Air helpers
# against Qwen3.5-9B instead of the older Qwen-AgentWorld-35B-A3B defaults.

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
export MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
export DATASET_ID="${DATASET_ID:-Qwen/AgentWorldBench}"
export DATA_DIR="${DATA_DIR:-/workspace/AgentWorldBench}"
export RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"

# Qwen3.5-9B's native long-context boundary. Smoke runs can override these,
# but full validation should keep them at the BF16 model boundary.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"

# A single A100 can stage and smoke Qwen3.5-9B. If vLLM/SGLang is used, tensor
# parallelism does not need to mirror the old 35B AgentWorld topology.
export TP="${TP:-1}"
export REQUIRED_GPUS="${REQUIRED_GPUS:-1}"

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-flash-02-23}"
