#!/usr/bin/env python3
"""Strict CoT-language checks for the post-short-merge RoboTwin release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone

from atomic_seg.common.artifacts import iter_jsonl as rows
from atomic_seg.common.artifacts import sha256_file
from atomic_seg.cot_data.validation import (  # noqa: E402
    _parse_reasoning,
    _semantic_issues,
)
from atomic_seg.cot_data.evidence import QUALITY_POLICY, validate_annotation

THINK_RE = re.compile(r"^<think>\n(.*)\n</think>\n\n(\{.*\})$", re.DOTALL)
STYLE_MARKERS = {
    "evidence_first": re.compile(
        r"\b(?:evidence|camera|view|visible|state (?:shows|indicates|reveals))\b",
        re.IGNORECASE,
    ),
    "temporal_contrast": re.compile(
        r"\b(?:previous boundary|compared with|across the boundary|since the previous|changed|unchanged|state delta)\b",
        re.IGNORECASE,
    ),
    "goal_decomposition": re.compile(
        r"\b(?:subgoal|prerequisite|unresolved|requirement|remaining objective)\b",
        re.IGNORECASE,
    ),
    "alternative_elimination": re.compile(
        r"\b(?:alternative|rather than|instead of|repeating|repeat|unnecessary|reject)\b",
        re.IGNORECASE,
    ),
    "uncertainty_calibrated": re.compile(
        r"\b(?:uncertain|uncertainty|cannot confirm|not confirm|unverified|ambiguous|ambiguity|occluded|cannot verify|not assumed)\b",
        re.IGNORECASE,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    annotations = rows(args.annotations)
    styles: Counter[str] = Counter()
    style_marker_hits: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    hashes: set[str] = set()
    word_counts: list[int] = []
    count = 0
    with args.vlm.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            annotation = next(annotations, None)
            if annotation is None:
                raise ValueError("annotations ended before VLM")
            if int(annotation["row_index"]) != index:
                raise ValueError("annotation indices are not contiguous")
            frozen = validate_annotation(annotation, row, journal_root=args.annotations.parent)
            match = THINK_RE.fullmatch(str(row["messages"][2]["content"]))
            if match is None:
                raise ValueError(f"row {index}: assistant format mismatch")
            reasoning_with_close = match.group(1).strip()
            marker = " Therefore, the next action should be:"
            split = reasoning_with_close.rfind(marker)
            if split < 0:
                raise ValueError(f"row {index}: final action sentence missing")
            reasoning = reasoning_with_close[:split].strip()
            if reasoning_with_close != annotation["reasoning"] + " " + frozen["closing_sentence"]:
                raise ValueError(f"row {index}: final action sentence mismatch")
            if reasoning != str(annotation["reasoning"]).strip():
                raise ValueError(f"row {index}: annotation/VLM reasoning mismatch")
            parsed = _parse_reasoning({"reasoning": reasoning})
            if parsed != reasoning:
                raise ValueError(f"row {index}: reasoning normalization mismatch")
            semantic_issues = _semantic_issues(
                reasoning,
                row["target"],
                int(annotation["style_id"]),
            )
            if semantic_issues:
                raise ValueError(
                    f"row {index}: CoT/action semantic mismatch: {semantic_issues}"
                )
            if (
                int(annotation["accepted_round"]) <= 0
                or annotation.get("fallback") is True
            ):
                raise ValueError(f"row {index}: fallback or unaccepted annotation")
            digest = hashlib.sha256(reasoning.encode()).hexdigest()
            if digest != str(annotation["reasoning_sha256"]):
                raise ValueError(f"row {index}: reasoning hash mismatch")
            hashes.add(digest)
            style = str(annotation["style_name"])
            styles[style] += 1
            marker = STYLE_MARKERS.get(style)
            if marker is None:
                raise ValueError(f"row {index}: unknown style {style}")
            if marker.search(reasoning):
                style_marker_hits[style] += 1
            word_counts.append(int(annotation["word_count"]))
            words = re.findall(r"[A-Za-z0-9']+", reasoning.lower())
            openings[" ".join(words[:8])] += 1
            count += 1
    if next(annotations, None) is not None:
        raise ValueError("annotations have extra rows")
    if count != args.expected_rows:
        raise ValueError(f"row count {count} != {args.expected_rows}")
    if count == 0:
        raise ValueError("Cannot release an empty annotation set")
    if len(hashes) != count:
        raise ValueError("duplicate full reasoning survived")
    if max(styles.values()) - min(styles.values()) > max(1, (count + 999) // 1000):
        raise ValueError("five styles are not near-balanced")
    style_marker_rates = {
        style: round(style_marker_hits[style] / styles[style], 6)
        for style in sorted(styles)
    }
    if len(openings) < min(count, max(10_000, count // 10)):
        raise ValueError("opening diversity is too low")
    if max(openings.values()) > math.ceil(count * 0.02):
        raise ValueError("one opening prefix is over-represented")
    report = {
        "schema_version": "robotwin_grounded_cot_quality_audit_v1",
        "quality_policy": QUALITY_POLICY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "counts": {
            "rows": count,
            "styles": dict(styles),
            "unique_reasoning_hashes": len(hashes),
            "fallback_rows": 0,
            "metadata_leak_rows": 0,
            "observer_and_two_critics_verified_rows": count,
            "style_marker_hits": dict(style_marker_hits),
        },
        "style_marker_rates": style_marker_rates,
        "style_markers_are_diagnostics_only": True,
        "word_count": {
            "min": min(word_counts),
            "mean": round(statistics.mean(word_counts), 3),
            "max": max(word_counts),
        },
        "opening_prefix_8_words": {
            "unique": len(openings),
            "max_count": max(openings.values()),
        },
        "hashes": {
            "annotations": sha256_file(args.annotations),
            "vlm": sha256_file(args.vlm),
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
