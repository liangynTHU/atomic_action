#!/usr/bin/env bash
# Recheck the same 18 examples with one-second edge trimming and build a new review.
set -euo pipefail
REALOMNI_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REALOMNI_PYTHON="${REALOMNI_PYTHON:-/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python}"
REALOMNI_RUN="${1:-$REALOMNI_REPO/artifacts/realomni_trim1s_run}"
REALOMNI_REVIEW="${2:-$REALOMNI_REPO/artifacts/realomni_trim1s_review}"
export PYTHONPATH="$REALOMNI_REPO/.runtime/pretrain_deps${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
REALOMNI_ARGS=()
while IFS= read -r episode; do
  [[ -z "$episode" ]] || REALOMNI_ARGS+=(--episode "$episode")
done < "$REALOMNI_REPO/configs/realomni_examples.txt"
"$REALOMNI_PYTHON" "$REALOMNI_REPO/scripts/prepare_realomni.py" \
  --quality-policy trim_edges_1s --output-root "$REALOMNI_RUN" "${REALOMNI_ARGS[@]}"
if [[ -f "$REALOMNI_REVIEW/index.html" ]]; then
  printf 'Preview: %s/index.html\n' "$REALOMNI_REVIEW"
else
  "$REALOMNI_PYTHON" "$REALOMNI_REPO/scripts/build_realomni_trim_review.py" \
    --run-root "$REALOMNI_RUN" --output-root "$REALOMNI_REVIEW" \
    --reuse-media-from "$REALOMNI_REPO/artifacts/realomni_examples_review" --limit 6
fi
