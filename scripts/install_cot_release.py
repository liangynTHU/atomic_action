#!/usr/bin/env python3
"""Explicit installation entrypoint; dataset contents are never pruned here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from atomic_seg.cot_data.install import install_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            install_release(
                args.release_root,
                args.dataset_root,
                replace=args.replace,
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
