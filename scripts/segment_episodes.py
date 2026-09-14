#!/usr/bin/env python3
"""Segment curated successful episodes using the final state-only algorithm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_seg import SegmentationConfig, segment_episode
from atomic_seg.io import (
    load_episode,
    load_success_manifest,
    resolve_episode_tasks,
    save_json,
    validate_end_pose_dataset,
)
from atomic_seg.segmentation import serializable_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/segmentation"),
    )
    parser.add_argument("--min-segment-seconds", type=float, default=0.10)
    parser.add_argument("--boundary-merge-seconds", type=float, default=0.10)
    parser.add_argument("--phase-window-seconds", type=float, default=0.18)
    parser.add_argument("--phase-min-seconds", type=float, default=0.10)
    parser.add_argument("--onset-search-seconds", type=float, default=0.18)
    parser.add_argument("--cross-arm-sync-seconds", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = validate_end_pose_dataset(args.data_root)
    episodes = resolve_episode_tasks(root, load_success_manifest(args.manifest))
    config = SegmentationConfig(
        min_segment_seconds=args.min_segment_seconds,
        boundary_merge_seconds=args.boundary_merge_seconds,
        phase_window_seconds=args.phase_window_seconds,
        phase_min_seconds=args.phase_min_seconds,
        onset_search_seconds=args.onset_search_seconds,
        cross_arm_sync_seconds=args.cross_arm_sync_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "segmentation_run_manifest",
        "success_only": True,
        "data_root": str(root.resolve()),
        "episodes": [],
    }
    for episode in episodes:
        source = load_episode(root, episode.episode_index)
        result = segment_episode(
            np.asarray(source["observation.state"], dtype=np.float64),
            np.asarray(source["timestamp"], dtype=np.float64),
            config,
        )
        payload = {
            **serializable_segmentation(result),
            "episode_index": episode.episode_index,
            "episode_id": episode.source_episode_id,
            "task": episode.task,
            "success": True,
            "success_provenance": episode.success_provenance,
        }
        output_path = (
            args.output_dir / f"episode_{episode.episode_index:06d}.json"
        )
        save_json(output_path, payload)
        manifest["episodes"].append(
            {
                "episode_index": episode.episode_index,
                "path": str(output_path),
                "segment_count": len(result["segments"]),
                "boundaries": result["boundaries"],
            }
        )
        print(
            f"episode={episode.episode_index} "
            f"segments={len(result['segments'])} -> {output_path}"
        )
    save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
