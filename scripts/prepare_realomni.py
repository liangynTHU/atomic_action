#!/usr/bin/env python3
"""Inspect complete RealOmni episodes and resume atomic action segmentation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_seg.pretrain.realomni.config import Settings
from atomic_seg.pretrain.realomni.runner import run_preparation, write_progress
from atomic_seg.pretrain.realomni.storage import lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", default=Settings.source_root)
    parser.add_argument("--quality-policy", choices=["strict", "trim_edges_1s"], default="trim_edges_1s")
    parser.add_argument("--episode", action="append", help="source-relative MCAP path; repeat for a fixed selection")
    parser.add_argument("--max-episodes", type=int, help="maximum unfinished episodes to process this invocation")
    parser.add_argument("--report-only", action="store_true", help="rebuild JSON progress from durable episode checkpoints")
    args = parser.parse_args()
    if args.report_only:
        with lock(args.output_root / "run.lock"):
            report = write_progress(args.output_root, args.output_root / "inventory/episodes.jsonl")
    else:
        settings = Settings(source_root=str(Path(args.source_root).resolve()), quality_policy=args.quality_policy)
        report = run_preparation(args.output_root, settings, selection=args.episode, max_episodes=args.max_episodes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
