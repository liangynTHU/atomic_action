#!/usr/bin/env python3
"""Rebuild old recovery segmentation using the mandatory current quality policy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.recovery_cot.refinement import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    prepare_still_refinement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    report = prepare_still_refinement(
        args.input_root,
        args.output_root,
        workers=args.workers,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
