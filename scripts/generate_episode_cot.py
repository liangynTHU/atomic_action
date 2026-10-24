#!/usr/bin/env python3
"""Single-episode convenience alias for generate_cot_data.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_cot_data import main

if __name__ == "__main__":
    if "--manifest" in sys.argv:
        raise SystemExit("Use generate_cot_data.py for manifest batches")
    if (
        "--episode" not in sys.argv
        and "--help" not in sys.argv
        and "-h" not in sys.argv
    ):
        raise SystemExit("--episode is required")
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "artifacts/episode_cot"])
    main()
