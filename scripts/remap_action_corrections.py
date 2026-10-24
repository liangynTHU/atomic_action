#!/usr/bin/env python3
"""Remap action-correction segment indices after short-segment merging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lookup: dict[tuple[int, int], dict] = {}
    with args.mapping.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            episode = int(item["episode_index"])
            for old_segment in item["source_segment_indices"]:
                key = (episode, int(old_segment))
                if key in lookup:
                    raise ValueError(f"duplicate old segment mapping: {key}")
                lookup[key] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        args.corrections.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (int(item["episode_index"]), int(item["segment_index"]))
            mapped = lookup.get(key)
            if mapped is None:
                raise ValueError(f"missing correction mapping: {key}")
            output = dict(item)
            output["pre_short_merge_segment_index"] = int(item["segment_index"])
            output["segment_index"] = int(mapped["new_segment_index"])
            output["frame_index"] = int(mapped["start_frame"])
            output["short_merge_source_segment_indices"] = [
                int(value) for value in mapped["source_segment_indices"]
            ]
            destination.write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    print(json.dumps({"corrections": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
