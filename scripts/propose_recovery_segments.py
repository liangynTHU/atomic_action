#!/usr/bin/env python3
"""Propose non-authoritative recovery boundaries for human/VLM review."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

import numpy as np

from atomic_seg.io import load_episode, save_json
from atomic_seg.recovery_cot.io import recovery_state_layout
from atomic_seg.recovery_cot.segmentation import (
    RecoveryProposalConfig,
    propose_recovery_segments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--min-segment-seconds", type=float, default=0.25)
    parser.add_argument("--max-segment-seconds", type=float, default=3.0)
    return parser.parse_args()


def _supervised_range(source: dict[str, object]) -> tuple[int, int]:
    states = np.asarray(source["observation.state"])
    mask = np.ones(len(states), dtype=bool)
    if "action_is_supervised" in source:
        mask &= np.asarray(source["action_is_supervised"], dtype=bool)
    if "observation_only" in source:
        mask &= ~np.asarray(source["observation_only"], dtype=bool)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("episode contains no supervised recovery rows")
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError(
            "supervised recovery rows are not one contiguous range; select "
            "--start-frame/--end-frame explicitly"
        )
    return int(indices[0]), int(indices[-1])


def main() -> None:
    args = parse_args()
    source = load_episode(
        args.data_root,
        args.episode,
        columns=(
            "observation.state",
            "timestamp",
            "action_is_supervised",
            "observation_only",
        ),
    )
    detected_start, detected_end = _supervised_range(source)
    start = detected_start if args.start_frame is None else args.start_frame
    end = detected_end if args.end_frame is None else args.end_frame
    proposal = propose_recovery_segments(
        np.asarray(source["observation.state"], dtype=np.float64),
        np.asarray(source["timestamp"], dtype=np.float64),
        recovery_state_layout(args.data_root),
        start_frame=start,
        end_frame=end,
        config=RecoveryProposalConfig(
            min_segment_seconds=args.min_segment_seconds,
            max_segment_seconds=args.max_segment_seconds,
        ),
    )
    payload = {
        **proposal,
        "data_root": str(args.data_root.resolve()),
        "episode_index": args.episode,
    }
    save_json(args.output, payload)
    print(
        f"episode={args.episode} proposals={len(proposal['segments'])} "
        f"review_required=true -> {args.output}"
    )


if __name__ == "__main__":
    main()
