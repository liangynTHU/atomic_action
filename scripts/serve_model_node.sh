#!/usr/bin/env bash
# Run on EVERY allocated node. One TP8 endpoint uses all eight local GPUs.
set -euo pipefail
SERVE_SCRIPT="${SERVE_SCRIPT:?set the Qwen deployment serve_qwen38_node.sh path}"
MODEL_PATH="${MODEL_PATH:?set the shared Qwen model directory}"
[[ -f "${SERVE_SCRIPT}" && -d "${MODEL_PATH}" ]] || { echo 'Missing model or serving script' >&2; exit 2; }
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "${gpu_count}" -eq 8 ]] || { echo "Expected 8 visible GPUs; got ${gpu_count}" >&2; exit 2; }
export MODEL_PATH
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TP_SIZE=8 EP_SIZE=8 PORT="${PORT:-8007}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
exec bash "${SERVE_SCRIPT}" --disable-custom-all-reduce "$@"
