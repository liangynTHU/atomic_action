#!/usr/bin/env python3
"""Generate action-only and CoT data for exactly one successful episode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_seg.generation import (
    OpenAICompatibleReasoner,
    TemplateReasoner,
    generate_cot_dataset,
    generation_manifest,
)
from atomic_seg.io import (
    SuccessEpisode,
    resolve_episode_tasks,
    save_json,
    validate_end_pose_dataset,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--success-provenance",
        default="curated successful demonstration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/episode_cot"),
    )
    parser.add_argument(
        "--reasoner",
        choices=("template", "openai-compatible"),
        default="template",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8007")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = validate_end_pose_dataset(args.data_root)
    episodes = resolve_episode_tasks(
        root,
        [
            SuccessEpisode(
                episode_index=args.episode,
                task=args.task,
                success=True,
                success_provenance=args.success_provenance,
            )
        ],
    )
    reasoner = (
        TemplateReasoner()
        if args.reasoner == "template"
        else OpenAICompatibleReasoner(
            endpoint=args.endpoint,
            model=args.model,
            api_key=os.environ.get(args.api_key_env),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_dataset, cot_rows = generate_cot_dataset(
        root,
        episodes,
        args.output_dir,
        reasoner,
    )
    save_json(args.output_dir / "action.json", action_dataset)
    write_jsonl(args.output_dir / "cot.jsonl", cot_rows)
    save_json(
        args.output_dir / "manifest.json",
        generation_manifest(
            data_root=root,
            episodes=episodes,
            action_dataset=action_dataset,
            cot_rows=cot_rows,
            reasoner=reasoner,
        ),
    )
    print(
        f"episode={args.episode} segments={len(cot_rows)} "
        f"-> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
