"""Rewrite a RoboTwin VLM/VLA pair with the shared still policy.

The two training files are processed together so their segment boundaries,
guide-action targets, chunk metadata, execution history and CoT final JSON stay
aligned.  Existing image assets are reused; no future frame is introduced.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_seg.common.artifacts import compact_json as _compact_json
from atomic_seg.common.artifacts import sha256_file as _sha256
from atomic_seg.cot_data.records import chunks as _chunks
from atomic_seg.cot_data.records import current_images as _current_images
from atomic_seg.cot_data.records import delta as _delta
from atomic_seg.cot_data.records import episode_vlm_rows as _episode_vlm_rows
from atomic_seg.cot_data.records import iter_vla as _iter_vla
from atomic_seg.cot_data.records import iter_vlm as _iter_vlm
from atomic_seg.cot_data.records import sample_id as _sample_id
from atomic_seg.cot_data.records import user_prompt as _user_prompt
from atomic_seg.still_policy import plan_still_segment_groups

CHUNK_SIZE = 8
STATE_DIMENSION = 16


def _refine_episode(
    episode: Mapping[str, Any],
    vlm_rows: Sequence[Mapping[str, Any]],
    *,
    global_block_start: int,
    minimum_terminal_still_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Counter]:
    source_segments = list(episode["segments"])
    if len(source_segments) != len(vlm_rows):
        raise ValueError("VLM/VLA segment counts differ")
    for index, (segment, row) in enumerate(zip(source_segments, vlm_rows)):
        if int(segment["source_segment_id"]) != index:
            raise ValueError("VLA source_segment_id is not contiguous")
        if int(row["segment_index"]) != index:
            raise ValueError("VLM segment_index is not contiguous")
        if int(segment["start_frame"]) != int(row["segment_start_frame"]) or int(
            segment["end_frame"]
        ) != int(row["segment_end_frame"]):
            raise ValueError("VLM/VLA frame ranges differ")

    groups = plan_still_segment_groups(
        source_segments,
        minimum_terminal_still_frames=minimum_terminal_still_frames,
    )
    refined_vla_segments: list[dict[str, Any]] = []
    refined_vlm_rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    counts: Counter = Counter()
    previous_refined_row: dict[str, Any] | None = None

    for new_index, group in enumerate(groups):
        owner_segment = source_segments[group.owner_index]
        owner_row = vlm_rows[group.owner_index]
        first_row = vlm_rows[group.source_indices[0]]
        last_row = vlm_rows[group.source_indices[-1]]
        start = group.start_frame
        end = group.end_frame
        current_state = [float(item) for item in first_row["current_state"]]
        endpoint_state = [float(item) for item in last_row["guide_action_target_state"]]
        valid_rows = end - start + 1
        num_chunks = _chunks(valid_rows)
        target = {
            "guide_action": _delta(current_state, endpoint_state, rounded=True),
            "primary_action_verb": str(owner_segment["primary_action_verb"]),
            "sub_task": str(owner_segment["sub_task"]),
            "num_chunks": num_chunks,
        }

        segment = copy.deepcopy(owner_segment)
        segment.update(
            {
                "sample_id": _sample_id(str(segment["sample_id"]), new_index),
                "source_segment_id": new_index,
                "source_step_index": new_index + 1,
                "start_frame": start,
                "end_frame": end,
                "timestamp_formatted": f"{start} - {end}",
                "guide_action": copy.deepcopy(target["guide_action"]),
                "num_chunks": num_chunks,
                "global_block_index": global_block_start + new_index,
                "step_index": new_index + 1,
            }
        )
        refined_vla_segments.append(segment)

        row = copy.deepcopy(owner_row)
        previous_state = (
            None
            if previous_refined_row is None
            else [float(item) for item in previous_refined_row["current_state"]]
        )
        previous_goal = (
            None
            if previous_refined_row is None
            else copy.deepcopy(previous_refined_row["target"])
        )
        previous_frame = (
            None
            if previous_refined_row is None
            else int(previous_refined_row["segment_start_frame"])
        )
        if previous_refined_row is None:
            images = _current_images(first_row)
            source_action_rows = 0
            executed_chunks = 0
            executed_actions = 0
            padded_action_rows = 0
            observed_delta = None
        else:
            images = [
                *_current_images(previous_refined_row),
                *_current_images(first_row),
            ]
            source_action_rows = start - int(
                previous_refined_row["segment_start_frame"]
            )
            executed_chunks = _chunks(source_action_rows)
            executed_actions = executed_chunks * CHUNK_SIZE
            padded_action_rows = executed_actions - source_action_rows
            observed_delta = _delta(
                previous_state,
                current_state,
                rounded=False,
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
            image_count=len(images),
            prompt_required_arm=prompt_required_arm,
            task_arm_mode=str(row["task_arm_mode"]),
            previous_state=previous_state,
            previous_goal=previous_goal,
            executed_chunks=executed_chunks,
            executed_actions=executed_actions,
            source_action_rows=source_action_rows,
            padded_action_slots=padded_action_rows,
        )
        aligned = num_chunks * CHUNK_SIZE
        padded = aligned - valid_rows
        row.update(
            {
                "images": images,
                "segment_index": new_index,
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
                "origin_offset": 0,
                "vla_max_origin_offset": min(
                    max(0, valid_rows - 1),
                    max(0, int(math.floor(valid_rows * 0.3))),
                ),
                "within_vla_origin_distribution": True,
                "sample_type": (
                    "initial_boundary" if new_index == 0 else "subtask_transition"
                ),
                "history_outcome": (
                    "none" if new_index == 0 else "successful_transition"
                ),
                "current_state": current_state,
                "guide_action_target_state": endpoint_state,
                "previous_boundary_state": previous_state,
                "history_anchor_state": copy.deepcopy(previous_state),
                "previous_boundary_frame": previous_frame,
                "history_anchor_frame": previous_frame,
                "observed_state_delta": observed_delta,
                "previous_goal": previous_goal,
                "history_images_included": previous_refined_row is not None,
                "history_state_label": (
                    "previous executed planning origin"
                    if previous_refined_row is not None
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
                    if previous_refined_row is not None
                    else None
                ),
                "action_padding_value": (
                    0.0 if previous_refined_row is not None else None
                ),
                "target": target,
                "cot_annotation_required": True,
            }
        )
        row["messages"] = [
            copy.deepcopy(owner_row["messages"][0]),
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
        refined_vlm_rows.append(row)
        previous_refined_row = row
        counts["cot_annotation_required"] += 1

        absorbed_lookup = dict(
            zip(group.absorbed_still_indices, group.absorbed_still_roles)
        )
        for old_index in group.source_indices:
            role = absorbed_lookup.get(old_index) or (
                "owner" if old_index == group.owner_index else "merged"
            )
            mapping.append(
                {
                    "episode_index": int(episode["global_episode_index"]),
                    "old_segment_index": old_index,
                    "new_segment_index": new_index,
                    "role": role,
                    "old_start_frame": int(source_segments[old_index]["start_frame"]),
                    "old_end_frame": int(source_segments[old_index]["end_frame"]),
                    "new_start_frame": start,
                    "new_end_frame": end,
                }
            )
            if old_index in absorbed_lookup:
                counts[f"absorbed_{absorbed_lookup[old_index]}"] += 1

    total_steps = len(refined_vla_segments)
    for segment in refined_vla_segments:
        segment["source_total_steps"] = total_steps
        segment["total_steps"] = total_steps
    refined_episode = copy.deepcopy(episode)
    refined_episode["source_total_steps"] = total_steps
    refined_episode["total_steps"] = total_steps
    refined_episode["segments"] = refined_vla_segments

    for index, row in enumerate(refined_vlm_rows):
        if int(row["segment_index"]) != index:
            raise AssertionError("refined VLM indices are not contiguous")
        if index < len(refined_vlm_rows) - 1 and (
            row["target"]["primary_action_verb"] == "left:still, right:still"
        ):
            raise AssertionError("nonterminal both-still VLM row survived")
    return refined_episode, refined_vlm_rows, mapping, counts


def rewrite_dataset(
    *,
    input_vlm: Path,
    input_vla: Path,
    output_vlm: Path,
    output_vla: Path,
    report_path: Path,
    mapping_path: Path,
    minimum_terminal_still_frames: int = 8,
) -> dict[str, Any]:
    """Stream-rewrite and validate the aligned training pair."""

    for output in (output_vlm, output_vla, report_path, mapping_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    vlm_iterator = _iter_vlm(input_vlm)
    pending_vlm = next(vlm_iterator, None)
    temporary_vlm = output_vlm.with_name(f".{output_vlm.name}.tmp-{os.getpid()}")
    temporary_vla = output_vla.with_name(f".{output_vla.name}.tmp-{os.getpid()}")
    temporary_mapping = mapping_path.with_name(
        f".{mapping_path.name}.tmp-{os.getpid()}"
    )
    counts: Counter = Counter()
    global_block_index = 0
    previous_global_episode = -1

    try:
        with (
            temporary_vlm.open("w", encoding="utf-8") as vlm_output,
            temporary_vla.open("w", encoding="utf-8") as vla_output,
            temporary_mapping.open("w", encoding="utf-8") as mapping_output,
        ):
            vla_output.write("{\n")
            first_episode = True
            for training_key, episode in _iter_vla(input_vla):
                global_episode = int(episode["global_episode_index"])
                if global_episode <= previous_global_episode:
                    raise ValueError("VLA episodes are not globally ordered")
                episode_rows, pending_vlm = _episode_vlm_rows(
                    vlm_iterator,
                    pending_vlm,
                    global_episode,
                )
                refined_episode, refined_rows, mapping, episode_counts = (
                    _refine_episode(
                        episode,
                        episode_rows,
                        global_block_start=global_block_index,
                        minimum_terminal_still_frames=(minimum_terminal_still_frames),
                    )
                )
                counts.update(episode_counts)
                counts["input_episodes"] += 1
                counts["input_segments"] += len(episode["segments"])
                counts["output_segments"] += len(refined_rows)
                if len(refined_rows) != len(episode["segments"]):
                    counts["changed_episodes"] += 1
                for row in refined_rows:
                    vlm_output.write(_compact_json(row) + "\n")
                for item in mapping:
                    mapping_output.write(_compact_json(item) + "\n")
                if not first_episode:
                    vla_output.write(",\n")
                first_episode = False
                vla_output.write(
                    json.dumps(str(training_key)) + ":" + _compact_json(refined_episode)
                )
                global_block_index += len(refined_rows)
                previous_global_episode = global_episode
                if counts["input_episodes"] % 1000 == 0:
                    print(
                        "[still-cleanup] "
                        f"episodes={counts['input_episodes']} "
                        f"segments={counts['output_segments']}",
                        flush=True,
                    )
            vla_output.write("\n}\n")
        if pending_vlm is not None or next(vlm_iterator, None) is not None:
            raise ValueError("VLM contains rows after the final VLA episode")
        os.replace(temporary_vlm, output_vlm)
        os.replace(temporary_vla, output_vla)
        os.replace(temporary_mapping, mapping_path)
    except BaseException:
        temporary_vlm.unlink(missing_ok=True)
        temporary_vla.unlink(missing_ok=True)
        temporary_mapping.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "robotwin_v28_both_still_cleanup_v1",
        "policy": {
            "initial_both_still": "absorb_into_next_action_as_pre_context",
            "middle_both_still": "absorb_into_previous_action_as_post_settle",
            "terminal_both_still": (
                "retain_one_stage_only_when_at_least_minimum_frames"
            ),
            "minimum_terminal_still_frames": minimum_terminal_still_frames,
            "active_short_segments": "preserved_unless_they_are_both_still",
            "cot": "action_json_pending_visual_observer_writer_and_dual_critics",
        },
        "counts": dict(counts),
        "inputs": {
            "vlm": str(input_vlm),
            "vla": str(input_vla),
            "vlm_sha256": _sha256(input_vlm),
            "vla_sha256": _sha256(input_vla),
        },
        "outputs": {
            "vlm": str(output_vlm),
            "vla": str(output_vla),
            "mapping": str(mapping_path),
            "vlm_sha256": _sha256(output_vlm),
            "vla_sha256": _sha256(output_vla),
            "mapping_sha256": _sha256(mapping_path),
        },
    }
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report
