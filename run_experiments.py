#!/usr/bin/env python3
"""Run one or all segmentation versions on selected RoboTwin episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from atomic_seg.io import (
    DEFAULT_JOINT_ROOT,
    DEFAULT_REFERENCE_JSON,
    infer_pose_root,
    load_episode,
    load_episode_tasks,
    load_reference,
    parse_episode_ids,
    save_json,
)
from atomic_seg.pipeline import VERSION_NAMES, VERSION_ROLES, run_version
from atomic_seg.plotting import write_diagnostic_svg
from atomic_seg.segmentation import SegmenterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_JOINT_ROOT)
    parser.add_argument("--pose-root", type=Path, default=None)
    parser.add_argument("--episodes", type=str, default="0,550")
    parser.add_argument(
        "--versions",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated list from 1..8",
    )
    parser.add_argument(
        "--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument("--max-segments", type=int, default=20)
    parser.add_argument("--min-segment-frames", type=int, default=5)
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.10,
        help="V6+ minimum primitive duration in physical seconds",
    )
    parser.add_argument(
        "--boundary-merge-seconds",
        type=float,
        default=0.10,
        help="V6+ same-event temporal merge radius in seconds",
    )
    parser.add_argument(
        "--phase-window-seconds",
        type=float,
        default=0.18,
        help="V6+ local state window duration in seconds",
    )
    parser.add_argument(
        "--phase-min-seconds",
        type=float,
        default=0.10,
        help="V6+ minimum persistent phase duration in seconds",
    )
    parser.add_argument(
        "--onset-search-seconds",
        type=float,
        default=0.18,
        help="V7/V8 search radius for successor trend onset",
    )
    parser.add_argument(
        "--cross-arm-sync-seconds",
        type=float,
        default=0.08,
        help="V8 maximum offset for compatible left/right synchronization",
    )
    parser.add_argument(
        "--arm-mode",
        choices=("joint", "active"),
        default="joint",
        help=(
            "joint fuses left/right evidence (final formulation); active "
            "retains the previous global-active-arm simplification as an ablation"
        ),
    )
    parser.add_argument(
        "--vertical-direction",
        type=str,
        default="0,0,1",
        help="Comma-separated vertical unit direction in the trajectory frame",
    )
    return parser.parse_args()


def clean_for_json(record: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    pose_root = args.pose_root or infer_pose_root(args.data_root)
    episodes = parse_episode_ids(args.episodes)
    versions = parse_episode_ids(args.versions)
    tasks = load_episode_tasks(pose_root)
    vertical_direction = tuple(
        float(value) for value in args.vertical_direction.split(",")
    )
    if len(vertical_direction) != 3:
        raise ValueError("--vertical-direction must contain exactly 3 values")
    config = SegmenterConfig(
        min_segment_frames=args.min_segment_frames,
        max_segments=args.max_segments,
        arm_mode=args.arm_mode,
        vertical_direction=vertical_direction,
        min_segment_seconds=args.min_segment_seconds,
        boundary_merge_seconds=args.boundary_merge_seconds,
        phase_window_seconds=args.phase_window_seconds,
        phase_min_seconds=args.phase_min_seconds,
        onset_search_seconds=args.onset_search_seconds,
        cross_arm_sync_seconds=args.cross_arm_sync_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "requested_data_root": str(args.data_root),
        "pose_root": str(pose_root),
        "episodes": episodes,
        "versions": versions,
        "version_roles": {
            str(version): VERSION_ROLES[version] for version in versions
        },
        "arm_mode": args.arm_mode,
        "vertical_direction": list(vertical_direction),
        "state_only_v6_plus": {
            "decision_inputs": ["observation.state", "timestamp"],
            "visual_or_action_config_used_for_boundaries": False,
            "min_segment_seconds": args.min_segment_seconds,
            "boundary_merge_seconds": args.boundary_merge_seconds,
            "phase_window_seconds": args.phase_window_seconds,
            "phase_min_seconds": args.phase_min_seconds,
            "onset_search_seconds": args.onset_search_seconds,
            "cross_arm_sync_seconds": args.cross_arm_sync_seconds,
        },
        "reference_type": "weak_segmentation_reference",
        "results": [],
    }
    for episode_index in episodes:
        episode = load_episode(pose_root, episode_index)
        trajectory = np.asarray(episode["observation.state"], dtype=np.float64)
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        reference = load_reference(args.reference_json, episode_index)
        episode_dir = args.output_dir / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        for version in versions:
            result = run_version(
                trajectory,
                version,
                reference=reference,
                segmenter_config=config,
                timestamps=timestamps,
            )
            result["episode_index"] = episode_index
            result["task"] = tasks.get(episode_index, "")
            result["timestamp_start"] = float(timestamps[0])
            result["timestamp_end"] = float(timestamps[-1])
            result["fps_estimate"] = float(
                1.0 / np.median(np.diff(timestamps))
            )
            json_path = episode_dir / f"v{version}_{VERSION_NAMES[version]}.json"
            save_json(json_path, clean_for_json(result))
            feature_bundle = result["_feature_bundle"]
            plot_bundle = result.get("_state_bundle", feature_bundle)
            active_arm = str(result["active_arm"])
            energy = np.asarray(
                plot_bundle["arms"][active_arm]["energy_smooth"]
            )
            svg_path = episode_dir / f"v{version}_{VERSION_NAMES[version]}.svg"
            write_diagnostic_svg(
                svg_path,
                trajectory,
                result["segments"],
                active_arm,
                energy,
                title=(
                    f"Episode {episode_index} · v{version} "
                    f"{VERSION_NAMES[version]} · {tasks.get(episode_index, '')}"
                ),
                reference_segments=reference.get("segments")
                if reference
                else None,
                boundary_evidence=result["selection"].get(
                    "boundary_evidence"
                ),
                feature_bundle=plot_bundle,
                selection=result["selection"],
                timestamps=timestamps,
            )
            manifest["results"].append(
                {
                    "episode_index": episode_index,
                    "version": version,
                    "json": str(json_path),
                    "svg": str(svg_path),
                    "segment_count": len(result["segments"]),
                    "weak_reference_boundary_f1": (
                        result["boundary_metrics"]["f1"]
                        if result["boundary_metrics"]
                        else None
                    ),
                    "segment_count_error": (
                        result["boundary_metrics"]["segment_count_error"]
                        if result["boundary_metrics"]
                        else None
                    ),
                    "over_segmentation_count": (
                        result["boundary_metrics"][
                            "over_segmentation_count"
                        ]
                        if result["boundary_metrics"]
                        else None
                    ),
                    "under_segmentation_count": (
                        result["boundary_metrics"][
                            "under_segmentation_count"
                        ]
                        if result["boundary_metrics"]
                        else None
                    ),
                }
            )
            print(
                f"episode={episode_index} version={version} "
                f"segments={len(result['segments'])} -> {json_path}"
            )
    save_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
