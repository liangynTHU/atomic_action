#!/usr/bin/env python3
"""Build the full native RoboTwin recovery CoT release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.recovery_cot.contracts import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE_DRIVER,
    DEFAULT_SOURCE_ROOT,
)
from atomic_seg.recovery_cot.native import create_manifest
from atomic_seg.recovery_cot.production import finalize, run_rank, status


def _endpoints(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    manifest.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    manifest.add_argument("--workers", type=int, default=16)
    manifest.add_argument("--limit-episodes", type=int)

    rank = subparsers.add_parser("run-rank")
    rank.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    rank.add_argument("--rank", type=int, required=True)
    rank.add_argument("--world-size", type=int, default=8)
    rank.add_argument(
        "--endpoints",
        default="http://127.0.0.1:8007",
        help="Comma-separated OpenAI-compatible base URLs.",
    )
    rank.add_argument("--model", default=DEFAULT_MODEL)
    rank.add_argument("--endpoint-file", type=Path)
    rank.add_argument("--prepare-workers", type=int, default=4)
    rank.add_argument("--generation-workers", type=int, default=4)
    rank.add_argument("--image-width", type=int, default=384)
    rank.add_argument("--jpeg-quality", type=int, default=90)
    rank.add_argument("--request-timeout", type=float, default=600.0)
    rank.add_argument("--max-tokens", type=int, default=8192)
    rank.add_argument("--critic-max-tokens", type=int, default=1600)
    rank.add_argument("--max-attempts", type=int, default=9)
    rank.add_argument("--revision-rounds", type=int, default=5)
    rank.add_argument("--generation-rounds", type=int, default=4)
    rank.add_argument("--launch-attempt", type=int, default=1)
    rank.add_argument("--reference-driver", type=Path, default=DEFAULT_REFERENCE_DRIVER)
    rank.add_argument("--limit-episodes", type=int)
    rank.add_argument("--resume", action="store_true")
    rank.add_argument("--template", action="store_true")
    rank.add_argument(
        "--test-only",
        action="store_true",
        help="Required with --template; final release always rejects templates.",
    )

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    finish.add_argument("--world-size", type=int, default=8)
    finish.add_argument(
        "--reference-driver", type=Path, default=DEFAULT_REFERENCE_DRIVER
    )

    check = subparsers.add_parser("status")
    check.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    check.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        result = create_manifest(
            args.source_root,
            args.output_root,
            workers=args.workers,
            limit_episodes=args.limit_episodes,
        )
    elif args.command == "run-rank":
        from atomic_seg.common.locking import writer_lock

        if args.template and not args.test_only:
            raise ValueError(
                "--template requires --test-only and cannot produce a formal release"
            )
        with writer_lock(args.output_root / "control" / f"rank_{args.rank:02d}.lock"):
            result = run_rank(
                output_root=args.output_root,
                rank=args.rank,
                world_size=args.world_size,
                endpoint_values=(
                    [
                        line.strip()
                        for line in args.endpoint_file.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
                    if args.endpoint_file
                    else _endpoints(args.endpoints)
                ),
                model=args.model,
                prepare_workers=args.prepare_workers,
                generation_workers=args.generation_workers,
                image_width=args.image_width,
                jpeg_quality=args.jpeg_quality,
                timeout=args.request_timeout,
                max_tokens=args.max_tokens,
                critic_max_tokens=args.critic_max_tokens,
                max_attempts=args.max_attempts,
                revision_rounds=args.revision_rounds,
                generation_rounds=args.generation_rounds,
                launch_attempt=args.launch_attempt,
                resume=args.resume,
                reference_driver=args.reference_driver,
                limit_episodes=args.limit_episodes,
                template=args.template,
            )
    elif args.command == "finalize":
        from atomic_seg.common.locking import writer_lock

        with writer_lock(args.output_root / "control/finalize.lock"):
            result = finalize(
                args.output_root,
                world_size=args.world_size,
                reference_driver=args.reference_driver,
            )
    else:
        result = status(args.output_root, world_size=args.world_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
