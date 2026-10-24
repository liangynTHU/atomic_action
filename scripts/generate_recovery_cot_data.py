#!/usr/bin/env python3
"""Generate outcome-aware segment CoT for audited recovery episodes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from dataclasses import replace

from atomic_seg.io import save_json, write_jsonl
from atomic_seg.recovery_cot.generation import (
    OpenAICompatibleRecoveryReasoner,
    RecoveryTemplateReasoner,
    generate_recovery_cot_dataset,
    recovery_generation_manifest,
)
from atomic_seg.recovery_cot.io import load_recovery_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/recovery_cot_data"),
    )
    parser.add_argument(
        "--reasoner",
        choices=("template", "openai-compatible"),
        default="openai-compatible",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Acknowledge this is a non-release arbitrary-layout adapter.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Required for offline template smoke data; not publishable.",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8007")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Pilot safety limit; omit only for an intentional full run.",
    )
    parser.add_argument(
        "--max-segments-per-episode",
        type=int,
        default=None,
        help="Optional pilot limit applied after manifest validation.",
    )
    return parser.parse_args()


def _positive_or_none(value: int | None, name: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    args = parse_args()
    if not args.experimental:
        raise ValueError(
            "Use recovery_cot.py for formal data; this adapter requires --experimental"
        )
    if args.reasoner == "template" and not args.test_only:
        raise ValueError("template recovery generation requires --test-only")
    max_episodes = _positive_or_none(args.max_episodes, "max-episodes")
    max_segments = _positive_or_none(
        args.max_segments_per_episode,
        "max-segments-per-episode",
    )
    episodes = load_recovery_manifest(args.manifest)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if max_segments is not None:
        episodes = [
            replace(episode, segments=episode.segments[:max_segments])
            for episode in episodes
        ]
    reasoner = (
        RecoveryTemplateReasoner()
        if args.reasoner == "template"
        else OpenAICompatibleRecoveryReasoner(
            endpoint=args.endpoint,
            model=args.model,
            api_key=os.environ.get(args.api_key_env),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_dataset, cot_rows = generate_recovery_cot_dataset(
        episodes,
        args.output_dir,
        reasoner,
    )
    save_json(
        args.output_dir / "recovery_action_train.json",
        action_dataset,
    )
    write_jsonl(
        args.output_dir / "recovery_cot_train.jsonl",
        cot_rows,
    )
    save_json(
        args.output_dir / "manifest.json",
        recovery_generation_manifest(
            source_manifest=args.manifest,
            episodes=episodes,
            action_dataset=action_dataset,
            cot_rows=cot_rows,
            reasoner=reasoner,
        ),
    )
    print(
        f"recovery_episodes={len(episodes)} "
        f"segments={action_dataset['segment_count']} "
        f"cot_rows={len(cot_rows)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
