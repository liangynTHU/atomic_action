#!/usr/bin/env python3
"""Build a recovery-shaped action segmentation manifest for normal CoT data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.cot_data.action_manifest import build_action_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-version", default="robotwin_cot_current")
    parser.add_argument(
        "--segmentation-policy",
        default=(
            "nonterminal both-still absorbed; one >=8-row terminal "
            "both-still stage allowed"
        ),
    )
    parser.add_argument("--confirmed-corrections", type=int)
    args = parser.parse_args()
    result = build_action_manifest(
        vla_path=args.vla,
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        report_path=args.report,
        code_version=args.code_version,
        confirmed_corrections=args.confirmed_corrections,
        segmentation_policy=args.segmentation_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
