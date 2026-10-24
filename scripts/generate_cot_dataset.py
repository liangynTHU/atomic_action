#!/usr/bin/env python3
"""Generate and finalize five Qwen3.8 CoT styles for normal RoboTwin."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.cot_data.generation import (  # noqa: E402
    DEFAULT_MODEL,
    finalize_dataset,
    generate_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--input-vlm", type=Path, required=True)
    generate.add_argument("--annotations", type=Path, required=True)
    generate.add_argument("--endpoint-file", type=Path, required=True)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--workers", type=int, default=64)
    generate.add_argument("--batch-size", type=int, default=256)
    generate.add_argument("--timeout", type=float, default=600.0)
    generate.add_argument("--max-tokens", type=int, default=3072)
    generate.add_argument("--max-rounds", type=int, default=4)
    generate.add_argument("--job-attempts", type=int, default=3)
    generate.add_argument("--limit-rows", type=int)
    generate.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Global inclusive source row for an independently written shard.",
    )
    generate.add_argument(
        "--end-row",
        type=int,
        help="Global exclusive source row for an independently written shard.",
    )
    generate.add_argument("--resume", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--input-root", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--annotations", type=Path, required=True)
    finalize.add_argument("--overwrite", action="store_true")
    finalize.add_argument(
        "--allow-preserved-style-allocation",
        action="store_true",
        help="Allow a near-balanced preserved style assignment after row merging.",
    )
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_annotations(
            input_vlm=args.input_vlm,
            annotation_path=args.annotations,
            endpoint_file=args.endpoint_file,
            model=args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_rounds=args.max_rounds,
            job_attempts=args.job_attempts,
            limit_rows=args.limit_rows,
            start_row=args.start_row,
            end_row=args.end_row,
            resume=args.resume,
        )
    else:
        result = finalize_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            annotation_path=args.annotations,
            overwrite=args.overwrite,
            require_modulo_style_allocation=(not args.allow_preserved_style_allocation),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
