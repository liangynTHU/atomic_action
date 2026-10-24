#!/usr/bin/env python3
"""Generate full-episode rollout/failure/recovery CoT pilot data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from atomic_seg.recovery_cot.full_episode import generate_full_episode_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Required: outcome/phase pilot is not the native planner release.",
    )
    parser.add_argument(
        "--existing-asset-root",
        type=Path,
        default=None,
        help="Reuse previously extracted assets/ instead of decoding videos.",
    )
    parser.add_argument(
        "--reference-review-root",
        type=Path,
        default=None,
        help="Optional complete-episode review bundle with audited failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.experimental:
        raise ValueError(
            "full-episode diagnostics require --experimental; use recovery_cot.py for formal planner data"
        )
    rows, report = generate_full_episode_dataset(
        args.manifest,
        args.output_dir,
        existing_asset_root=args.existing_asset_root,
        reference_review_root=args.reference_review_root,
    )
    print(
        f"episodes={report['episode_count']} rows={len(rows)} "
        f"actions={report['action_supervision_rows']} "
        f"diagnostics={report['diagnostic_only_rows']} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
