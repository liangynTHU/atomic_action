#!/usr/bin/env python3
"""Remove standalone nonterminal both-still fragments from normal VLM/VLA."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.cot_data.still_cleanup import rewrite_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-vlm", type=Path, required=True)
    parser.add_argument("--input-vla", type=Path, required=True)
    parser.add_argument("--output-vlm", type=Path, required=True)
    parser.add_argument("--output-vla", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--minimum-terminal-still-frames", type=int, default=8)
    args = parser.parse_args()
    report = rewrite_dataset(
        input_vlm=args.input_vlm,
        input_vla=args.input_vla,
        output_vlm=args.output_vlm,
        output_vla=args.output_vla,
        report_path=args.report,
        mapping_path=args.mapping,
        minimum_terminal_still_frames=args.minimum_terminal_still_frames,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
