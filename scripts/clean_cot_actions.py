#!/usr/bin/env python3
"""Prepare normal action labels and canonical sub-tasks for visual CoT generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.cot_data.semantic_cleanup import (  # noqa: E402
    rewrite_semantic_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-vlm", type=Path, required=True)
    parser.add_argument("--input-vla", type=Path, required=True)
    parser.add_argument("--output-vlm", type=Path, required=True)
    parser.add_argument("--output-vla", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--confirmed-corrections", type=Path)
    args = parser.parse_args()
    report = rewrite_semantic_dataset(
        input_vlm=args.input_vlm,
        input_vla=args.input_vla,
        output_vlm=args.output_vlm,
        output_vla=args.output_vla,
        report_path=args.report,
        corrections_path=args.corrections,
        confirmed_corrections_path=args.confirmed_corrections,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
