#!/usr/bin/env python3
"""Merge independently generated full-episode judgment manifests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from atomic_seg.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    episodes = []
    sources = []
    seen = set()
    for path in args.inputs:
        value = load_json(path)
        if value.get("schema") != "full_episode_recovery_cot_manifest":
            raise ValueError(f"{path}: unexpected schema")
        sources.append(str(path.resolve()))
        for episode in value.get("episodes", []):
            episode_id = str(episode["episode_id"])
            if episode_id in seen:
                raise ValueError(f"duplicate episode_id {episode_id}")
            seen.add(episode_id)
            episodes.append(episode)
    save_json(
        args.output,
        {
            "schema": "full_episode_recovery_cot_manifest",
            "judgment_source": {
                "mode": "merged_qwen_blind_multimodal",
                "source_manifests": sources,
            },
            "episodes": episodes,
        },
    )
    print(f"episodes={len(episodes)} -> {args.output}")


if __name__ == "__main__":
    main()
