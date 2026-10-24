#!/usr/bin/env bash
set -euo pipefail
AGIBOT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGIBOT_PYTHON="${AGIBOT_PYTHON:-/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$AGIBOT_PYTHON" ]] || { printf 'Missing AGIBOT_PYTHON=%s\n' "$AGIBOT_PYTHON" >&2; exit 2; }

# Fresh API run; never mix these labels with the legacy Qwen run_v1.
exec "$AGIBOT_PYTHON" "$AGIBOT_REPO/scripts/agibot.py" \
  --source "${AGIBOT_SOURCE:-/apdcephfs_gy2/share_303214312/leoyizhang/datasets/agibot_beta_v3}" \
  --output "${AGIBOT_OUTPUT:-/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot/run_glm53_flash_xhigh_grounded_v7}" \
  --api-base "${AGIBOT_API_BASE:-http://127.0.0.1:15721/tencent/v1}" \
  --api-token-file "${AGIBOT_API_TOKEN_FILE:-$AGIBOT_REPO/.api_token}" \
  --model "${AGIBOT_MODEL:-glm-5.3-flash-ioa}" \
  --reasoning-effort "${AGIBOT_REASONING_EFFORT:-xhigh}" \
  --temperature "${AGIBOT_TEMPERATURE:-0.1}" \
  --max-tokens "${AGIBOT_MAX_TOKENS:-8192}" \
  --concurrency "${AGIBOT_CONCURRENCY:-4}" \
  --batch-size "${AGIBOT_BATCH_SIZE:-4}" \
  "$@"
