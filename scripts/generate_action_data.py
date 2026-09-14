#!/usr/bin/env python3
"""Generate action-only supervision from curated successful episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_seg.generation import generate_action_dataset, generation_manifest
from atomic_seg.io import (
    load_success_manifest,
    resolve_episode_tasks,
    save_json,
    validate_end_pose_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/action_data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = validate_end_pose_dataset(args.data_root)
    episodes = resolve_episode_tasks(root, load_success_manifest(args.manifest))
    dataset = generate_action_dataset(root, episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train.json", dataset)
    save_json(
        args.output_dir / "manifest.json",
        generation_manifest(
            data_root=root,
            episodes=episodes,
            action_dataset=dataset,
        ),
    )
    print(
        f"episodes={dataset['episode_count']} "
        f"segments={dataset['segment_count']} "
        f"-> {args.output_dir / 'train.json'}"
    )


if __name__ == "__main__":
    main()
