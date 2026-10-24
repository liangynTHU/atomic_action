#!/usr/bin/env bash
set -euo pipefail

AGIBOT_PART="${1:?Missing AgiBot partition number}"
shift
[[ "$AGIBOT_PART" == 1 || "$AGIBOT_PART" == 2 ]] || {
  printf 'AgiBot partition must be 1 or 2.\n' >&2
  exit 2
}

AGIBOT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGIBOT_PYTHON="${AGIBOT_PYTHON:-/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python}"
AGIBOT_SOURCE="${AGIBOT_SOURCE:-/apdcephfs_gy2/share_303214312/leoyizhang/datasets/agibot_beta_v3}"
AGIBOT_RUN_ROOT="${AGIBOT_OUTPUT:-/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot/run_glm53_flash_xhigh_grounded_v7_16d_split2}"
export PYTHONDONTWRITEBYTECODE=1

for AGIBOT_ARGUMENT in "$@"; do
  case "$AGIBOT_ARGUMENT" in
    --source|--source=*|--output|--output=*|--selection|--selection=*)
      printf 'Use AGIBOT_SOURCE / AGIBOT_OUTPUT to configure both partitions; --source, --output and --selection are managed by these launchers.\n' >&2
      exit 2
      ;;
  esac
done

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<EOF
Usage: bash agibot_part${AGIBOT_PART}.sh [command] [pipeline options]

Commands:
  split     Prepare or verify both episode selections; no model requests.
  info      Prepare selections and display this partition's paths.
  check     Check paths, dependencies and the authenticated model catalog.
  probe     Make one real GLM image request.
  prepare   Freeze source and prepare this partition's production inventory.
  start     Process this partition (default); rerun to resume.
  status    Show this partition's progress.
  review    Build this partition's browser preview.

Environment:
  AGIBOT_SOURCE       Shared source dataset.
  AGIBOT_OUTPUT       Shared output parent; each launcher adds /part1 or /part2.
  AGIBOT_SPLIT_DIR    Optional shared directory for immutable selections.
  AGIBOT_CONCURRENCY  Workers per partition (default: 2, total: 4).
  AGIBOT_PROGRESS_EVERY  Episodes per progress summary (default: 1000).
  Other AGIBOT_* model/API settings are inherited by agibot.sh.

Use the same environment for both launchers. Selection and output CLI overrides
are reserved to keep the two partitions separate. Model options such as
--object-grounding can be passed through.
EOF
  exit 0
fi

[[ -x "$AGIBOT_PYTHON" ]] || {
  printf 'Missing AGIBOT_PYTHON=%s\n' "$AGIBOT_PYTHON" >&2
  exit 2
}

AGIBOT_SPLIT_DIR="${AGIBOT_SPLIT_DIR:-$("$AGIBOT_PYTHON" "$AGIBOT_REPO/scripts/split_agibot.py" --source "$AGIBOT_SOURCE" --print-default-dir)}"
AGIBOT_COMMAND="start"
case "${1:-}" in
  split|info|check|probe|prepare|start|status|review)
    AGIBOT_COMMAND="$1"
    shift
    ;;
esac

if [[ "$AGIBOT_COMMAND" == "split" || "$AGIBOT_COMMAND" == "info" ]] && (( $# )); then
  printf 'The %s command does not accept pipeline options.\n' "$AGIBOT_COMMAND" >&2
  exit 2
fi

"$AGIBOT_PYTHON" "$AGIBOT_REPO/scripts/split_agibot.py" \
  --source "$AGIBOT_SOURCE" \
  --split-dir "$AGIBOT_SPLIT_DIR"

export AGIBOT_PYTHON AGIBOT_SOURCE
export AGIBOT_OUTPUT="$AGIBOT_RUN_ROOT/part$AGIBOT_PART"
export AGIBOT_CONCURRENCY="${AGIBOT_CONCURRENCY:-2}"
export AGIBOT_PROGRESS_EVERY="${AGIBOT_PROGRESS_EVERY:-1000}"
AGIBOT_SELECTION="$AGIBOT_SPLIT_DIR/selection/part$AGIBOT_PART.json"

printf 'AgiBot part %s: output=%s\nselection=%s\nconcurrency=%s\n' \
  "$AGIBOT_PART" "$AGIBOT_OUTPUT" "$AGIBOT_SELECTION" "$AGIBOT_CONCURRENCY"

if [[ "$AGIBOT_COMMAND" == "split" || "$AGIBOT_COMMAND" == "info" ]]; then
  exit 0
fi

exec bash "$AGIBOT_REPO/agibot.sh" "$AGIBOT_COMMAND" \
  --selection "$AGIBOT_SELECTION" "$@"
