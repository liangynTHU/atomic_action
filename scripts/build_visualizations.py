#!/usr/bin/env python3
"""Build three segmentation and three complete successful-episode examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_seg.generation import TemplateReasoner, generate_cot_dataset
from atomic_seg.io import (
    load_success_manifest,
    resolve_episode_tasks,
    save_json,
    validate_end_pose_dataset,
    write_jsonl,
)
from atomic_seg.visualization import (
    build_episode_visual_assets,
    package_visualizations,
    write_visualization_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/visualizations"),
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=(
            "Portable ZIP path. Defaults to "
            "<output-dir-parent>/atomic_episode_visualizations.zip. "
            "Generated media should remain outside Git."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = validate_end_pose_dataset(args.data_root)
    episodes = resolve_episode_tasks(root, load_success_manifest(args.manifest))
    if len(episodes) != 3:
        raise ValueError(
            "the visualization bundle must contain exactly 3 episodes"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reasoner = TemplateReasoner()
    action_dataset, cot_rows = generate_cot_dataset(
        root,
        episodes,
        args.output_dir,
        reasoner,
    )
    rows_by_episode = {
        episode.episode_index: [
            row
            for row in cot_rows
            if int(row["episode_index"]) == episode.episode_index
        ]
        for episode in episodes
    }
    action_by_episode = {
        int(record["episode_index"]): record
        for record in action_dataset["episodes"]
    }
    entries = []
    for episode in episodes:
        paths = build_episode_visual_assets(
            data_root=root,
            output_root=args.output_dir,
            episode=episode,
            action_episode=action_by_episode[episode.episode_index],
            cot_rows=rows_by_episode[episode.episode_index],
        )
        entries.append(
            {
                "episode_index": episode.episode_index,
                "task": episode.task,
                **paths,
            }
        )
        print(
            f"episode={episode.episode_index} "
            f"segmentation={paths['segmentation_svg']} "
            f"complete={paths['episode_html']}"
        )
    save_json(args.output_dir / "action_examples.json", action_dataset)
    write_jsonl(args.output_dir / "cot_examples.jsonl", cot_rows)
    save_json(
        args.output_dir / "manifest.json",
        {
            "schema": "visualization_examples",
            "success_only": True,
            "reasoner": reasoner.name,
            "segmentation_visualization_count": len(entries),
            "complete_episode_visualization_count": len(entries),
            "entries": entries,
        },
    )
    write_visualization_index(
        args.output_dir / "index.html",
        entries=entries,
    )
    zip_path = (
        args.zip_path
        if args.zip_path is not None
        else args.output_dir.parent / "atomic_episode_visualizations.zip"
    )
    package = package_visualizations(args.output_dir, zip_path)
    print(
        f"zip={package['path']} files={package['file_count']} "
        f"bytes={package['bytes']}"
    )


if __name__ == "__main__":
    main()
