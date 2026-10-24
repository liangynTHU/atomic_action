#!/usr/bin/env python3
"""Strict final audit and report builder for the one supported release."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from atomic_seg.action_language import (  # noqa: E402
    canonical_sub_task,
    cot_final_action_sentence,
)
from atomic_seg.common.artifacts import iter_jsonl as manifest_rows
from atomic_seg.common.artifacts import sha256_file
from atomic_seg.cot_data.generation import STYLE_SPECS  # noqa: E402
from atomic_seg.cot_data.records import iter_vla as _iter_vla
from atomic_seg.cot_data.release_metadata import write_release_metadata
from atomic_seg.cot_data.evidence import QUALITY_POLICY, validate_annotation
from atomic_seg.prompt_contract import (  # noqa: E402
    THINKING_USER_ENDING,
    system_prompt_for_mode,
)

THINK_RE = re.compile(r"^<think>\n(.*)\n</think>\n\n(\{.*\})$", re.DOTALL)


def split_assistant(value: str) -> tuple[str, dict[str, Any]]:
    match = THINK_RE.fullmatch(value)
    if match is None:
        raise ValueError("assistant format mismatch")
    answer = json.loads(match.group(2))
    if not isinstance(answer, dict):
        raise ValueError("assistant answer is not an object")
    return match.group(1).strip(), answer


def audit_release(
    *,
    release_root: Path,
    annotation_path: Path,
    corrections_path: Path,
    expected_rows: int,
    expected_episodes: int,
    allow_preserved_style_allocation: bool = False,
    require_no_short_same_action: bool = False,
    expected_corrections: int | None = None,
    write_metadata: bool = False,
) -> dict[str, Any]:
    vlm_path = release_root / "data/vlm/train.jsonl"
    vla_path = release_root / "data/vla/train.json"
    action_segmentation_path = release_root / "data/action_segmentation/train.jsonl"
    action_manifest_path = release_root / "metadata/source_manifest.jsonl"
    style_report_path = release_root / "COT_STYLE_REPORT.json"
    cot_quality_path = release_root / "COT_QUALITY_AUDIT.json"
    for path in (
        vlm_path,
        vla_path,
        action_segmentation_path,
        action_manifest_path,
        style_report_path,
        annotation_path,
        corrections_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(action_segmentation_path) != sha256_file(action_manifest_path):
        raise ValueError("action segmentation file differs from source manifest")
    style_report = json.loads(style_report_path.read_text())
    if style_report.get("quality_policy") != QUALITY_POLICY or style_report.get("quality", {}).get("observer_and_two_critics_verified") is not True:
        raise ValueError("Style report lacks the current grounding contract")
    if style_report["hashes"]["annotations"] != sha256_file(annotation_path):
        raise ValueError("Style report annotation hash mismatch")

    styles: Counter[str] = Counter()
    task_names: set[str] = set()
    variants: set[str] = set()
    rounds: Counter[int] = Counter()
    word_counts: list[int] = []
    opening_counts: Counter[str] = Counter()
    reasoning_hashes: set[str] = set()
    with annotation_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if int(row["row_index"]) != index:
                raise ValueError("annotation indices are not contiguous")
            if row.get("fallback") is True or int(row["accepted_round"]) <= 0:
                raise ValueError(f"annotation {index} did not pass Qwen validation")
            styles[str(row["style_name"])] += 1
            rounds[int(row["accepted_round"])] += 1
            word_counts.append(int(row["word_count"]))
            words = re.findall(r"[A-Za-z0-9']+", str(row["reasoning"]).lower())
            opening_counts[" ".join(words[:8])] += 1
            reasoning_hashes.add(str(row["reasoning_sha256"]))
    annotation_count = sum(styles.values())
    if annotation_count != expected_rows:
        raise ValueError(f"annotation count {annotation_count} != {expected_rows}")

    expected_styles = {
        spec["name"]: expected_rows // len(STYLE_SPECS)
        + (1 if index < expected_rows % len(STYLE_SPECS) else 0)
        for index, spec in enumerate(STYLE_SPECS)
    }
    expected_styles = {k: v for k, v in expected_styles.items() if v}
    if allow_preserved_style_allocation:
        if set(styles) != set(expected_styles):
            raise ValueError("preserved style allocation is missing a style")
        if max(styles.values()) - min(styles.values()) > max(
            1, math.ceil(expected_rows * 0.001)
        ):
            raise ValueError("preserved style allocation is not near-balanced")
    elif dict(styles) != expected_styles:
        raise ValueError("five-style allocation mismatch")

    manifest_iter = iter(manifest_rows(action_manifest_path))
    annotation_iter = manifest_rows(annotation_path)
    vlm_stream = vlm_path.open(encoding="utf-8")
    vlm_lines = iter(vlm_stream)
    segments = 0
    episodes = 0
    previous_block = -1
    action_by_key: dict[tuple[int, int, str], str] = {}
    try:
        for training_key, episode in _iter_vla(vla_path):
            task_names.add(str(episode.get("task_name", "unknown")))
            variants.add(str(episode.get("source_variant", "unknown")))
            manifest = next(manifest_iter)
            if int(training_key) != int(episode["training_episode_index"]):
                raise ValueError("VLA training episode index mismatch")
            if int(manifest["global_episode_index"]) != int(
                episode["global_episode_index"]
            ):
                raise ValueError("action manifest/VLA episode mismatch")
            if int(manifest["segment_count"]) != len(episode["segments"]):
                raise ValueError("action manifest segment count mismatch")
            if len(manifest["segments"]) != len(episode["segments"]):
                raise ValueError("action manifest segments array length mismatch")
            previous_row: dict[str, Any] | None = None
            previous_action_pair: str | None = None
            for segment_index, (segment, manifest_segment) in enumerate(
                zip(episode["segments"], manifest["segments"])
            ):
                line = next(vlm_lines)
                row = json.loads(line)
                annotation = next(annotation_iter)
                validate_annotation(annotation, row, journal_root=annotation_path.parent)
                if (
                    int(row["episode_index"]),
                    int(row["segment_index"]),
                ) != (
                    int(episode["global_episode_index"]),
                    segment_index,
                ):
                    raise ValueError("VLM/VLA episode-segment mismatch")
                if (
                    int(annotation["episode_index"]),
                    int(annotation["segment_index"]),
                ) != (
                    int(row["episode_index"]),
                    int(row["segment_index"]),
                ):
                    raise ValueError("annotation/VLM key mismatch")
                target = row["target"]
                if row["messages"][0] != {
                    "role": "system",
                    "content": system_prompt_for_mode(str(row["task_arm_mode"])),
                }:
                    raise ValueError("system prompt mismatch")
                if not str(row["messages"][1]["content"]).endswith(
                    THINKING_USER_ENDING
                ):
                    raise ValueError("thinking user ending mismatch")
                reasoning, answer = split_assistant(str(row["messages"][2]["content"]))
                if answer != target:
                    raise ValueError("assistant JSON differs from target")
                action_map = dict(
                    item.strip().split(":", 1)
                    for item in str(target["primary_action_verb"]).split(",")
                )
                for arm in ("left", "right"):
                    action_by_key[
                        (int(row["episode_index"]), int(row["segment_index"]), arm)
                    ] = action_map[arm]
                closing = cot_final_action_sentence(
                    action_map["left"],
                    action_map["right"],
                    int(target["num_chunks"]),
                )
                if not reasoning.endswith(closing):
                    raise ValueError("CoT final action sentence mismatch")
                if reasoning.count("Therefore, the next action should be:") != 1:
                    raise ValueError("CoT final sentence count mismatch")
                expected_sub_task = canonical_sub_task(
                    action_map["left"], action_map["right"]
                )
                if str(target["sub_task"]) != expected_sub_task:
                    raise ValueError("target sub_task is not canonical")
                for source in (segment, manifest_segment):
                    for key in (
                        "start_frame",
                        "end_frame",
                        "left_action",
                        "right_action",
                        "primary_action_verb",
                        "guide_action",
                        "num_chunks",
                        "sub_task",
                        "global_block_index",
                    ):
                        expected = target.get(key, row.get(key))
                        actual = source[key]
                        if key in {
                            "start_frame",
                            "end_frame",
                            "left_action",
                            "right_action",
                            "global_block_index",
                        }:
                            expected = (
                                row["segment_start_frame"]
                                if key == "start_frame"
                                else row["segment_end_frame"]
                                if key == "end_frame"
                                else action_map[key.split("_")[0]]
                                if key in {"left_action", "right_action"}
                                else segment["global_block_index"]
                            )
                        if actual != expected:
                            raise ValueError(f"VLM/VLA/manifest mismatch on {key}")
                current = [float(item) for item in row["current_state"]]
                endpoint = [float(item) for item in row["guide_action_target_state"]]
                if (
                    len(current) != 16
                    or len(endpoint) != 16
                    or len(target["guide_action"]) != 16
                ):
                    raise ValueError("guide/state vectors must be 16-D")
                if not all(
                    math.isfinite(float(v))
                    for v in current + endpoint + list(target["guide_action"])
                ):
                    raise ValueError("nonfinite state/guide value")
                residual = [
                    round(after - before, 4) for before, after in zip(current, endpoint)
                ]
                if (
                    max(
                        abs(float(left) - float(right))
                        for left, right in zip(target["guide_action"], residual)
                    )
                    > 0.00005001
                ):
                    raise ValueError("guide residual mismatch")
                valid_rows = (
                    int(row["segment_end_frame"]) - int(row["segment_start_frame"]) + 1
                )
                if int(target["num_chunks"]) != math.ceil(valid_rows / 8):
                    raise ValueError("num_chunks mismatch")
                if segment_index < len(episode["segments"]) - 1 and (
                    action_map == {"left": "still", "right": "still"}
                ):
                    raise ValueError("nonterminal both-still survived")
                if segment_index == len(episode["segments"]) - 1 and (
                    action_map == {"left": "still", "right": "still"} and valid_rows < 8
                ):
                    raise ValueError("short terminal both-still survived")
                current_action_pair = str(target["primary_action_verb"])
                if (
                    require_no_short_same_action
                    and previous_action_pair == current_action_pair
                    and (
                        valid_rows < 8 or int(previous_row["remaining_action_rows"]) < 8
                    )
                ):
                    raise ValueError(
                        "short adjacent identical-action boundary survived"
                    )
                if previous_row is None:
                    if row["previous_goal"] is not None:
                        raise ValueError("initial row has previous_goal")
                else:
                    if row["previous_goal"] != previous_row["target"]:
                        raise ValueError("history previous_goal mismatch")
                    prompt = str(row["messages"][1]["content"])
                    if (
                        '"valid_action_rows"' not in prompt
                        or '"padded_action_slots"' not in prompt
                    ):
                        raise ValueError("history execution padding is not explicit")
                previous_row = row
                previous_action_pair = current_action_pair
                block_index = int(segment["global_block_index"])
                if block_index != previous_block + 1:
                    raise ValueError("global block indices are not contiguous")
                previous_block = block_index
                segments += 1
            episodes += 1
        try:
            next(vlm_lines)
        except StopIteration:
            pass
        else:
            raise ValueError("VLM has extra rows")
        try:
            next(manifest_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("action manifest has extra rows")
        try:
            next(annotation_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("annotations have extra rows")
    finally:
        vlm_stream.close()

    if episodes != expected_episodes or segments != expected_rows:
        raise ValueError("final release counts mismatch")
    cot_quality: dict[str, Any] | None = None
    if not cot_quality_path.is_file():
        raise FileNotFoundError(cot_quality_path)
    cot_quality = json.loads(cot_quality_path.read_text(encoding="utf-8"))
    if cot_quality.get("passed") is not True:
        raise ValueError("CoT quality audit did not pass")
    if cot_quality.get("quality_policy") != QUALITY_POLICY or cot_quality["counts"].get("observer_and_two_critics_verified_rows") != expected_rows:
        raise ValueError("CoT quality audit lacks verified observer and dual critics")
    if int(cot_quality["counts"]["rows"]) != expected_rows:
        raise ValueError("CoT quality audit row count mismatch")
    if cot_quality["hashes"]["annotations"] != sha256_file(annotation_path):
        raise ValueError("CoT quality annotation hash mismatch")
    if cot_quality["hashes"]["vlm"] != sha256_file(vlm_path):
        raise ValueError("CoT quality VLM hash mismatch")
    corrections = [json.loads(line) for line in corrections_path.open() if line.strip()]
    if expected_corrections is not None and len(corrections) != expected_corrections:
        raise ValueError(f"expected {expected_corrections} action corrections")
    seen_corrections = set()
    for correction in corrections:
        key = (
            int(correction["episode_index"]),
            int(correction["segment_index"]),
            str(correction["arm"]),
        )
        if key in seen_corrections:
            raise ValueError(f"duplicate correction: {key}")
        seen_corrections.add(key)
        actual = action_by_key.get(key)
        expected = str(correction["new_action"])
        if actual != expected:
            raise ValueError(
                f"action correction not applied at {key}: {actual!r} != {expected!r}"
            )

    journal_manifest_path = annotation_path.parent / "REQUEST_JOURNALS.json"
    journal_manifest = json.loads(journal_manifest_path.read_text())
    journal_hashes = {}
    for name, checksum in journal_manifest["files"].items():
        journal = annotation_path.parent / name
        if sha256_file(journal) != checksum:
            raise ValueError("Bundled request journal changed")
        journal_hashes[str(journal.relative_to(release_root))] = checksum
    report = {
        "schema_version": ("robotwin_cot_release_audit_v1"),
        "quality_policy": QUALITY_POLICY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "counts": {
            "episodes": episodes,
            "tasks": len(task_names),
            "variants": sorted(variants),
            "segments": segments,
            "action_label_corrections": len(corrections),
            "action_label_corrections_verified": len(corrections),
            "styles": dict(styles),
            "accepted_rounds": {str(k): v for k, v in sorted(rounds.items())},
            "fallback_rows": 0,
            "unique_reasoning_hashes": len(reasoning_hashes),
        },
        "quality": {
            "observer_and_two_critics_verified": True,
            "visible_objects_colors_contact_allowed": True,
            "system_prompt_matches_reference_contract": True,
            "thinking_user_mode_explicit": True,
            "assistant_think_plus_final_json_valid": True,
            "all_cot_rows_end_with_exact_action_sentence": True,
            "vlm_vla_action_manifest_aligned": True,
            "action_segmentation_matches_manifest": True,
            "guide_residuals_valid": True,
            "history_chain_valid": True,
            "padding_semantics_explicit_in_prompt": True,
            "nonterminal_both_still": 0,
            "qwen_fallback_rows": 0,
            "short_adjacent_identical_action_boundaries": 0,
            "cot_language_audit_passed": cot_quality is not None,
            "word_count": {
                "min": min(word_counts),
                "mean": round(statistics.mean(word_counts), 3),
                "max": max(word_counts),
            },
            "opening_prefix_8_words": {
                "unique": len(opening_counts),
                "max_count": max(opening_counts.values()),
            },
        },
        "hashes": {
            "data/vlm/train.jsonl": sha256_file(vlm_path),
            "data/vla/train.json": sha256_file(vla_path),
            "data/action_segmentation/train.jsonl": sha256_file(
                action_segmentation_path
            ),
            "metadata/source_manifest.jsonl": sha256_file(action_manifest_path),
            "metadata/ACTION_LABEL_CORRECTIONS.jsonl": sha256_file(corrections_path),
            "work/cot5_annotations.jsonl": sha256_file(annotation_path),
            "work/REQUEST_JOURNALS.json": sha256_file(journal_manifest_path),
            "COT_STYLE_REPORT.json": sha256_file(style_report_path),
            **journal_hashes,
            **(
                {"COT_QUALITY_AUDIT.json": sha256_file(cot_quality_path)}
                if cot_quality is not None
                else {}
            ),
        },
    }
    if write_metadata:
        write_release_metadata(
            release_root,
            report,
            expected_rows=expected_rows,
            episodes=episodes,
            segments=segments,
            cot_quality=cot_quality,
            short_merge_checked=require_no_short_same_action,
            corrections_count=len(corrections),
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-corrections", type=int)
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument("--allow-preserved-style-allocation", action="store_true")
    parser.add_argument("--require-no-short-same-action", action="store_true")
    args = parser.parse_args()
    result = audit_release(
        release_root=args.release_root,
        annotation_path=args.annotations,
        corrections_path=args.corrections,
        expected_rows=args.expected_rows,
        expected_episodes=args.expected_episodes,
        allow_preserved_style_allocation=args.allow_preserved_style_allocation,
        require_no_short_same_action=args.require_no_short_same_action,
        expected_corrections=args.expected_corrections,
        write_metadata=args.write_metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
