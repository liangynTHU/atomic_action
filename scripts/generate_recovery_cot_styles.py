#!/usr/bin/env python3
"""Experimental text-only recovery style rewrite; not a native formal release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.recovery_cot.styles import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    finalize_dataset,
    generate_annotations,
    refresh_repetitive_openings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    generate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    generate.add_argument("--endpoint-file", type=Path, required=True)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--workers", type=int, default=64)
    generate.add_argument("--batch-size", type=int, default=256)
    generate.add_argument("--timeout", type=float, default=600.0)
    generate.add_argument("--generation-max-tokens", type=int, default=4096)
    generate.add_argument("--max-rounds", type=int, default=4)
    generate.add_argument("--job-attempts", type=int, default=3)
    generate.add_argument("--limit-rows", type=int)
    generate.add_argument("--resume", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    finalize.add_argument("--overwrite", action="store_true")

    refresh = subparsers.add_parser("refresh-openings")
    refresh.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    refresh.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    refresh.add_argument("--endpoint-file", type=Path, required=True)
    refresh.add_argument("--model", default=DEFAULT_MODEL)
    refresh.add_argument("--workers", type=int, default=128)
    refresh.add_argument("--batch-size", type=int, default=512)
    refresh.add_argument("--prefix-words", type=int, default=8)
    refresh.add_argument("--max-prefix-count", type=int, default=50)
    refresh.add_argument("--timeout", type=float, default=600.0)
    refresh.add_argument("--generation-max-tokens", type=int, default=4096)
    refresh.add_argument("--max-rounds", type=int, default=4)
    refresh.add_argument("--job-attempts", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        result = generate_annotations(
            args.input_root,
            args.output_root,
            endpoint_file=args.endpoint_file,
            model=args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            generation_max_tokens=args.generation_max_tokens,
            max_rounds=args.max_rounds,
            job_attempts=args.job_attempts,
            limit_rows=args.limit_rows,
            resume=args.resume,
        )
    elif args.command == "refresh-openings":
        result = refresh_repetitive_openings(
            args.input_root,
            args.output_root,
            endpoint_file=args.endpoint_file,
            model=args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            prefix_words=args.prefix_words,
            max_prefix_count=args.max_prefix_count,
            timeout=args.timeout,
            generation_max_tokens=args.generation_max_tokens,
            max_rounds=args.max_rounds,
            job_attempts=args.job_attempts,
        )
    else:
        result = finalize_dataset(
            args.input_root,
            args.output_root,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
