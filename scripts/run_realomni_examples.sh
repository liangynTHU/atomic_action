#!/usr/bin/env bash
# Reproduce six real examples with durable checks, then build an offline preview.
set -euo pipefail

REALOMNI_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REALOMNI_PYTHON="${REALOMNI_PYTHON:-/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python}"
REALOMNI_OUTPUT="${2:-$REALOMNI_REPO/artifacts/realomni_examples_run}"
REALOMNI_STAGE="${1:-all}"
export PYTHONPATH="$REALOMNI_REPO/.runtime/pretrain_deps${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [[ "$REALOMNI_STAGE" != all && "$REALOMNI_STAGE" != check && "$REALOMNI_STAGE" != review && "$REALOMNI_STAGE" != status ]]; then
  printf 'Usage: bash scripts/run_realomni_examples.sh [all|check|review|status] [output-directory]\n' >&2
  exit 2
fi

if [[ "$REALOMNI_STAGE" == all || "$REALOMNI_STAGE" == check ]]; then
  "$REALOMNI_PYTHON" "$REALOMNI_REPO/scripts/prepare_realomni.py" \
    --output-root "$REALOMNI_OUTPUT" \
    --episode 'Clutter-Tidy-Up-[Stage2]/00001/01710.mcap' \
    --episode 'Cooking_and_Kitchen_Clean/clean_bowl/00001/00004.mcap' \
    --episode 'Folding_Clothes_and_Zipper_Operations/fold_and_store_clothes/00001/00008.mcap' \
    --episode 'Organize_Clutter/desktop_object_sorting/00001/00001.mcap' \
    --episode 'Shoes_Handling/lace_up_shoes_with_both_hands/00001/00006.mcap' \
    --episode '[STAGE 3]/domestic_services/bedroom/clothing_organization/fold_and_store_clothes/0001/0668fc3fecf0489fa7d4f72bee5d8b41.mcap'
fi

if [[ "$REALOMNI_STAGE" == all || "$REALOMNI_STAGE" == review ]]; then
  if [[ -f "$REALOMNI_OUTPUT/review/index.html" ]]; then
    printf 'Existing preview: %s/review/index.html\n' "$REALOMNI_OUTPUT"
  else
    "$REALOMNI_PYTHON" "$REALOMNI_REPO/scripts/build_realomni_review.py" \
      --run-root "$REALOMNI_OUTPUT" --output-root "$REALOMNI_OUTPUT/review" --limit 6
  fi
fi

if [[ "$REALOMNI_STAGE" == status ]]; then
  "$REALOMNI_PYTHON" "$REALOMNI_REPO/scripts/prepare_realomni.py" \
    --output-root "$REALOMNI_OUTPUT" --report-only
fi
