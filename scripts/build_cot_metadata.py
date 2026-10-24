#!/usr/bin/env python3
"""Build the semantic/segmentation metadata report for RoboTwin."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from datetime import datetime, timezone

from atomic_seg.common.artifacts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_root
    previous = args.previous_root
    merge = json.loads((base / "SHORT_SEGMENT_MERGE_REPORT.json").read_text())
    action_manifest = json.loads((base / "ACTION_MANIFEST_REPORT.json").read_text())
    corrections = sum(
        1
        for line in (base / "metadata/ACTION_LABEL_CORRECTIONS.jsonl").open()
        if line.strip()
    )
    report = {
        "schema_version": "robotwin_v29_semantic_and_boundary_cleanup_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "action_language": "canonical actuator-grounded subtask language",
            "confirmed_action_label_corrections": corrections,
            "both_still": (
                "no initial or middle both-still; one terminal both-still stage "
                "allowed only when at least 8 action rows"
            ),
            "short_segments": (
                "a segment shorter than 8 action rows is merged only when an "
                "adjacent segment has the identical left/right action pair; "
                "different-action transitions are preserved"
            ),
            "cot": (
                "pending four-stage visual annotation; observed object names, "
                "colors, contact and holding are allowed when supported by images; "
                "both independent critics and request receipts required for release"
            ),
        },
        "counts": {
            **merge["counts"],
            "action_label_corrections": corrections,
            "manifest_segments": int(action_manifest["segments"]),
        },
        "inputs": {
            "previous_root": str(previous),
            "vlm_sha256": sha256_file(previous / "data/vlm/train.jsonl"),
            "vla_sha256": sha256_file(previous / "data/vla/train.json"),
        },
        "outputs": {
            "base_root": str(base),
            "vlm_sha256": sha256_file(base / "data/vlm/train.jsonl"),
            "vla_sha256": sha256_file(base / "data/vla/train.json"),
            "manifest_sha256": sha256_file(base / "metadata/source_manifest.jsonl"),
            "short_segment_mapping_sha256": sha256_file(
                base / "metadata/SHORT_SEGMENT_MAPPING.jsonl"
            ),
            "action_label_corrections_sha256": sha256_file(
                base / "metadata/ACTION_LABEL_CORRECTIONS.jsonl"
            ),
        },
    }
    output = base / "SEMANTIC_CLEANUP_REPORT.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
