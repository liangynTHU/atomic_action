#!/usr/bin/env python3
"""Merge only short RoboTwin segments into an identical-action neighbor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from atomic_seg.action_language import canonical_sub_task
from atomic_seg.common.artifacts import sha256_file
from atomic_seg.cot_data.records import CHUNK_SIZE, _compact_json
from atomic_seg.cot_data.records import current_images as _current_images
from atomic_seg.cot_data.records import delta as _delta
from atomic_seg.cot_data.records import episode_vlm_rows as _episode_vlm_rows
from atomic_seg.cot_data.records import iter_vla as _iter_vla
from atomic_seg.cot_data.records import iter_vlm as _iter_vlm
from atomic_seg.cot_data.records import sample_id as _sample_id
from atomic_seg.cot_data.records import user_prompt as _user_prompt

SCHEMA_VERSION = "robotwin_official_v29_short_same_action_merge_v1"


def segment_length(segment: Mapping[str, Any]) -> int:
    return int(segment["end_frame"]) - int(segment["start_frame"]) + 1


def short_same_action_groups(
    segments: Sequence[Mapping[str, Any]], minimum_rows: int
) -> list[list[int]]:
    """Absorb a short group only when an adjacent group has the same action pair."""

    groups = [[index] for index in range(len(segments))]
    changed = True
    while changed:
        changed = False
        for group_index, group in enumerate(groups):
            rows = sum(segment_length(segments[index]) for index in group)
            if rows >= minimum_rows:
                continue
            pair = str(segments[group[0]]["primary_action_verb"])
            previous_matches = (
                group_index > 0
                and str(segments[groups[group_index - 1][0]]["primary_action_verb"])
                == pair
            )
            next_matches = (
                group_index + 1 < len(groups)
                and str(segments[groups[group_index + 1][0]]["primary_action_verb"])
                == pair
            )
            if previous_matches:
                groups[group_index - 1].extend(group)
                del groups[group_index]
                changed = True
                break
            if next_matches:
                groups[group_index + 1] = group + groups[group_index + 1]
                del groups[group_index]
                changed = True
                break
    return groups


def rewrite(
    *,
    input_root: Path,
    output_root: Path,
    minimum_rows: int,
) -> dict[str, Any]:
    input_vlm = input_root / "data/vlm/train.jsonl"
    input_vla = input_root / "data/vla/train.json"
    output_vlm = output_root / "data/vlm/train.jsonl"
    output_vla = output_root / "data/vla/train.json"
    mapping_path = output_root / "metadata/SHORT_SEGMENT_MAPPING.jsonl"
    report_path = output_root / "SHORT_SEGMENT_MERGE_REPORT.json"
    for path in (output_vlm, output_vla, mapping_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_vlm = output_vlm.with_name(f".{output_vlm.name}.tmp-{os.getpid()}")
    tmp_vla = output_vla.with_name(f".{output_vla.name}.tmp-{os.getpid()}")
    tmp_map = mapping_path.with_name(f".{mapping_path.name}.tmp-{os.getpid()}")

    vlm_iterator = _iter_vlm(input_vlm)
    pending_vlm = next(vlm_iterator, None)
    counts: Counter[str] = Counter()
    old_global_row = 0
    new_global_row = 0
    previous_episode = -1
    try:
        with (
            tmp_vlm.open("w", encoding="utf-8") as vlm_output,
            tmp_vla.open("w", encoding="utf-8") as vla_output,
            tmp_map.open("w", encoding="utf-8") as mapping_output,
        ):
            vla_output.write("{\n")
            first_episode = True
            for training_key, episode in _iter_vla(input_vla):
                episode_index = int(episode["global_episode_index"])
                if episode_index <= previous_episode:
                    raise ValueError("VLA episodes are not globally ordered")
                source_rows, pending_vlm = _episode_vlm_rows(
                    vlm_iterator, pending_vlm, episode_index
                )
                source_segments = list(episode["segments"])
                if len(source_rows) != len(source_segments):
                    raise ValueError(f"episode {episode_index}: VLM/VLA count mismatch")
                old_indices = list(
                    range(old_global_row, old_global_row + len(source_segments))
                )
                old_global_row += len(source_segments)
                groups = short_same_action_groups(source_segments, minimum_rows)
                episode_changed = any(len(group) > 1 for group in groups)
                output_segments: list[dict[str, Any]] = []
                output_rows: list[dict[str, Any]] = []
                previous_row: dict[str, Any] | None = None

                for new_segment_index, group in enumerate(groups):
                    first_index, last_index = group[0], group[-1]
                    first_segment = source_segments[first_index]
                    first_row = source_rows[first_index]
                    last_row = source_rows[last_index]
                    pair = str(first_segment["primary_action_verb"])
                    if any(
                        str(source_segments[index]["primary_action_verb"]) != pair
                        for index in group
                    ):
                        raise AssertionError("attempted to merge unlike action pairs")
                    action_map = dict(
                        item.strip().split(":", 1) for item in pair.split(",")
                    )
                    start = int(first_segment["start_frame"])
                    end = int(source_segments[last_index]["end_frame"])
                    current_state = [float(item) for item in first_row["current_state"]]
                    endpoint_state = [
                        float(item) for item in last_row["guide_action_target_state"]
                    ]
                    valid_rows = end - start + 1
                    num_chunks = int(math.ceil(valid_rows / CHUNK_SIZE))
                    sub_task = canonical_sub_task(
                        action_map["left"], action_map["right"]
                    )
                    target = {
                        "guide_action": _delta(
                            current_state, endpoint_state, rounded=True
                        ),
                        "primary_action_verb": pair,
                        "sub_task": sub_task,
                        "num_chunks": num_chunks,
                    }

                    segment = copy.deepcopy(first_segment)
                    segment.update(
                        {
                            "sample_id": _sample_id(
                                str(first_segment["sample_id"]), new_segment_index
                            ),
                            "source_segment_id": new_segment_index,
                            "source_step_index": new_segment_index + 1,
                            "start_frame": start,
                            "end_frame": end,
                            "timestamp_formatted": f"{start} - {end}",
                            "left_action": action_map["left"],
                            "right_action": action_map["right"],
                            "primary_action_verb": pair,
                            "sub_task": sub_task,
                            "guide_action": copy.deepcopy(target["guide_action"]),
                            "num_chunks": num_chunks,
                            "global_block_index": new_global_row,
                            "source_global_block_index": int(
                                first_segment.get(
                                    "source_global_block_index",
                                    first_segment["global_block_index"],
                                )
                            ),
                            "step_index": new_segment_index + 1,
                            "merged_source_segment_indices": list(group),
                        }
                    )
                    output_segments.append(segment)

                    row = copy.deepcopy(first_row)
                    previous_state = (
                        None
                        if previous_row is None
                        else [float(item) for item in previous_row["current_state"]]
                    )
                    previous_goal = (
                        None
                        if previous_row is None
                        else copy.deepcopy(previous_row["target"])
                    )
                    previous_frame = (
                        None
                        if previous_row is None
                        else int(previous_row["segment_start_frame"])
                    )
                    if previous_row is None:
                        images = _current_images(first_row)
                        source_action_rows = 0
                        executed_chunks = 0
                        executed_actions = 0
                        padded_action_rows = 0
                        observed_delta = None
                    else:
                        images = [
                            *_current_images(previous_row),
                            *_current_images(first_row),
                        ]
                        source_action_rows = start - int(
                            previous_row["segment_start_frame"]
                        )
                        executed_chunks = int(
                            math.ceil(source_action_rows / CHUNK_SIZE)
                        )
                        executed_actions = executed_chunks * CHUNK_SIZE
                        padded_action_rows = executed_actions - source_action_rows
                        observed_delta = _delta(
                            previous_state, current_state, rounded=False
                        )
                    prompt_required_arm = (
                        row["required_arm"]
                        if row["task_arm_mode"] == "single"
                        and row["required_arm_source"] == "task_instruction"
                        else None
                    )
                    user_prompt = _user_prompt(
                        task=str(row["task"]),
                        current_state=current_state,
                        image_count=3 if previous_row is None else 6,
                        prompt_required_arm=prompt_required_arm,
                        task_arm_mode=str(row["task_arm_mode"]),
                        previous_state=previous_state,
                        previous_goal=previous_goal,
                        executed_chunks=executed_chunks,
                        executed_actions=executed_actions,
                        source_action_rows=source_action_rows,
                        padded_action_slots=padded_action_rows,
                        thinking=True,
                    )
                    aligned = num_chunks * CHUNK_SIZE
                    padded = aligned - valid_rows
                    row.update(
                        {
                            "images": images,
                            "segment_index": new_segment_index,
                            "global_block_index": new_global_row,
                            "source_global_block_index": int(
                                first_row.get(
                                    "source_global_block_index",
                                    first_row.get(
                                        "global_block_index", old_indices[first_index]
                                    ),
                                )
                            ),
                            "frame_index": start,
                            "segment_start_frame": start,
                            "segment_end_frame": end,
                            "action_start_frame": start,
                            "action_end_frame_inclusive": end,
                            "guide_action_origin_frame": start,
                            "guide_action_origin_state_frame": start,
                            "guide_action_target_frame": end + 1,
                            "guide_action_target_state_frame": end + 1,
                            "remaining_action_rows": valid_rows,
                            "target_valid_action_rows": valid_rows,
                            "target_num_chunks": num_chunks,
                            "target_aligned_action_slots": aligned,
                            "target_padded_action_rows": padded,
                            "target_action_slot_mask": [1] * valid_rows + [0] * padded,
                            "target_action_padding_mode": "masked_zero_slots_no_state_transition",
                            "target_action_padding_value": 0.0,
                            "origin_offset": 0,
                            "vla_max_origin_offset": min(
                                max(0, valid_rows - 1),
                                max(0, int(math.floor(valid_rows * 0.3))),
                            ),
                            "within_vla_origin_distribution": True,
                            "sample_type": (
                                "initial_boundary"
                                if new_segment_index == 0
                                else "subtask_transition"
                            ),
                            "history_outcome": (
                                "none"
                                if new_segment_index == 0
                                else "successful_transition"
                            ),
                            "current_state": current_state,
                            "guide_action_target_state": endpoint_state,
                            "previous_boundary_state": previous_state,
                            "history_anchor_state": copy.deepcopy(previous_state),
                            "previous_boundary_frame": previous_frame,
                            "history_anchor_frame": previous_frame,
                            "observed_state_delta": observed_delta,
                            "previous_goal": previous_goal,
                            "history_images_included": previous_row is not None,
                            "history_state_label": (
                                "previous executed planning origin"
                                if previous_row is not None
                                else None
                            ),
                            "executed_chunks_since_history": executed_chunks,
                            "executed_actions_since_history": executed_actions,
                            "source_action_rows_since_history": source_action_rows,
                            "padded_action_rows_since_history": padded_action_rows,
                            "aligned_action_slots_since_history": executed_actions,
                            "boundary_jitter_action_rows": 0,
                            "action_padding_mode": (
                                "masked_zero_slots_no_state_transition"
                                if previous_row is not None
                                else None
                            ),
                            "action_padding_value": 0.0
                            if previous_row is not None
                            else None,
                            "target": target,
                            "cot_annotation_required": True,
                        }
                    )
                    row["messages"] = [
                        copy.deepcopy(first_row["messages"][0]),
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ]
                    output_rows.append(row)
                    mapping_output.write(
                        _compact_json(
                            {
                                "new_row_index": new_global_row,
                                "episode_index": episode_index,
                                "new_segment_index": new_segment_index,
                                "source_row_indices": [
                                    old_indices[index] for index in group
                                ],
                                "source_segment_indices": list(group),
                                "merged": len(group) > 1,
                                "episode_changed": episode_changed,
                                "start_frame": start,
                                "end_frame": end,
                                "primary_action_verb": pair,
                            }
                        )
                        + "\n"
                    )
                    new_global_row += 1
                    previous_row = row

                total_steps = len(output_segments)
                for segment in output_segments:
                    segment["source_total_steps"] = total_steps
                    segment["total_steps"] = total_steps
                output_episode = copy.deepcopy(episode)
                output_episode["source_total_steps"] = total_steps
                output_episode["total_steps"] = total_steps
                output_episode["segments"] = output_segments
                for row in output_rows:
                    vlm_output.write(_compact_json(row) + "\n")
                if not first_episode:
                    vla_output.write(",\n")
                first_episode = False
                vla_output.write(
                    json.dumps(str(training_key)) + ":" + _compact_json(output_episode)
                )
                counts["episodes"] += 1
                counts["input_segments"] += len(source_segments)
                counts["output_segments"] += total_steps
                counts["removed_boundaries"] += len(source_segments) - total_steps
                counts["merged_groups"] += sum(len(group) > 1 for group in groups)
                counts["changed_episodes"] += int(episode_changed)
                counts["remaining_short_segments"] += sum(
                    segment_length(segment) < minimum_rows
                    for segment in output_segments
                )
                previous_episode = episode_index
                if counts["episodes"] % 1000 == 0:
                    print(
                        _compact_json(
                            {
                                "event": "short_merge_progress",
                                "episodes": counts["episodes"],
                                "output_segments": counts["output_segments"],
                            }
                        ),
                        flush=True,
                    )
            vla_output.write("\n}\n")
        if pending_vlm is not None or next(vlm_iterator, None) is not None:
            raise ValueError("VLM contains rows after final VLA episode")
        os.replace(tmp_vlm, output_vlm)
        os.replace(tmp_vla, output_vla)
        os.replace(tmp_map, mapping_path)
    except BaseException:
        tmp_vlm.unlink(missing_ok=True)
        tmp_vla.unlink(missing_ok=True)
        tmp_map.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "minimum_rows": minimum_rows,
            "merge_condition": (
                "segment shorter than minimum_rows and immediately adjacent "
                "to a segment with the identical left/right action pair"
            ),
            "different_action_short_segments": "preserved",
            "gripper_action_short_segments": "preserved unless identical neighbor",
            "cot": "action_json_pending_visual_observer_writer_and_dual_critics",
        },
        "counts": dict(counts),
        "inputs": {
            "root": str(input_root),
            "vlm_sha256": sha256_file(input_vlm),
            "vla_sha256": sha256_file(input_vla),
        },
        "outputs": {
            "root": str(output_root),
            "vlm_sha256": sha256_file(output_vlm),
            "vla_sha256": sha256_file(output_vla),
            "mapping_sha256": sha256_file(mapping_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=8)
    args = parser.parse_args()
    result = rewrite(
        input_root=args.input_root,
        output_root=args.output_root,
        minimum_rows=args.minimum_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
