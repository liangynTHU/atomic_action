"""Make RoboTwin action text structurally consistent with guide actions."""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_seg.action_language import canonical_sub_task
from atomic_seg.common.artifacts import sha256_file as _sha256
from atomic_seg.cot_data.records import _compact_json
from atomic_seg.cot_data.records import current_images as _current_images
from atomic_seg.cot_data.records import delta as _delta
from atomic_seg.cot_data.records import episode_vlm_rows as _episode_vlm_rows
from atomic_seg.cot_data.records import iter_vla as _iter_vla
from atomic_seg.cot_data.records import iter_vlm as _iter_vlm
from atomic_seg.cot_data.records import user_prompt as _user_prompt
from atomic_seg.prompt_contract import system_prompt_for_mode

CHUNK_SIZE = 8


def _quaternion_angle(first: Sequence[Any], second: Sequence[Any]) -> float:
    a = [float(item) for item in first]
    b = [float(item) for item in second]
    norm_a = math.sqrt(sum(item * item for item in a))
    norm_b = math.sqrt(sum(item * item for item in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("zero-norm quaternion")
    dot = abs(sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _arm_metrics(
    current: Sequence[Any],
    target: Sequence[Any],
    offset: int,
) -> tuple[float, float, float, float, float]:
    delta_xyz = [
        float(target[offset + index]) - float(current[offset + index])
        for index in range(3)
    ]
    translation = math.sqrt(sum(item * item for item in delta_xyz))
    horizontal = math.sqrt(delta_xyz[0] ** 2 + delta_xyz[1] ** 2)
    vertical = delta_xyz[2]
    rotation = _quaternion_angle(
        current[offset + 3 : offset + 7],
        target[offset + 3 : offset + 7],
    )
    gripper = float(target[offset + 7]) - float(current[offset + 7])
    return translation, horizontal, vertical, rotation, gripper


def _large_motion(metrics: Sequence[float]) -> bool:
    translation, _horizontal, _vertical, rotation, gripper = metrics
    return translation >= 0.02 or rotation >= 0.15 or abs(gripper) >= 0.20


def _classify_pose_action(metrics: Sequence[float]) -> str:
    translation, horizontal, vertical, rotation, gripper = metrics
    if gripper <= -0.20:
        return "close_gripper"
    if gripper >= 0.20:
        return "open_gripper"
    if abs(vertical) > max(0.015, 0.60 * translation) and abs(vertical) > horizontal:
        return "lift" if vertical > 0.0 else "lower"
    if rotation / 0.10 > 4.5 * max(translation / 0.08, 1e-6):
        return "turn"
    return "move"


def corrected_actions(
    segment: Mapping[str, Any],
    row: Mapping[str, Any],
    confirmed_corrections: Mapping[tuple[int, int, str], str] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Correct only hard action/guide contradictions."""

    current = row["current_state"]
    target = row["guide_action_target_state"]
    actions = {
        "left": str(segment["left_action"]),
        "right": str(segment["right_action"]),
    }
    metrics = {
        "left": _arm_metrics(current, target, 0),
        "right": _arm_metrics(current, target, 8),
    }
    corrections: list[dict[str, Any]] = []
    for arm in ("left", "right"):
        old = actions[arm]
        new = old
        correction_key = (
            int(row["episode_index"]),
            int(row["segment_index"]),
            arm,
        )
        confirmed = (
            None
            if confirmed_corrections is None
            else confirmed_corrections.get(correction_key)
        )
        if confirmed is not None:
            new = str(confirmed)
        elif old == "still" and _large_motion(metrics[arm]):
            new = _classify_pose_action(metrics[arm])
        elif old == "lift" and metrics[arm][2] < -0.02:
            new = _classify_pose_action(metrics[arm])
        elif old == "lower" and metrics[arm][2] > 0.02:
            new = _classify_pose_action(metrics[arm])
        elif old == "open_gripper" and metrics[arm][4] < -0.05:
            new = "close_gripper"
        elif old == "close_gripper" and metrics[arm][4] > 0.05:
            new = "open_gripper"
        if new != old:
            actions[arm] = new
            corrections.append(
                {
                    "arm": arm,
                    "old_action": old,
                    "new_action": new,
                    "translation_m": metrics[arm][0],
                    "vertical_delta_m": metrics[arm][2],
                    "rotation_rad": metrics[arm][3],
                    "gripper_delta": metrics[arm][4],
                    "source": (
                        "confirmed_raw_trajectory_audit"
                        if confirmed is not None
                        else "hard_endpoint_consistency_rule"
                    ),
                }
            )
    return actions["left"], actions["right"], corrections


def _rewrite_episode(
    episode: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    confirmed_corrections: Mapping[tuple[int, int, str], str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    segments = list(episode["segments"])
    if len(segments) != len(rows):
        raise ValueError("VLM/VLA segment counts differ")
    output_episode = copy.deepcopy(episode)
    output_segments: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    previous_row: dict[str, Any] | None = None

    for index, (source_segment, source_row) in enumerate(zip(segments, rows)):
        if int(source_segment["source_segment_id"]) != index:
            raise ValueError("VLA segment indices are not contiguous")
        if int(source_row["segment_index"]) != index:
            raise ValueError("VLM segment indices are not contiguous")
        left_action, right_action, action_corrections = corrected_actions(
            source_segment,
            source_row,
            confirmed_corrections,
        )
        verb = f"left:{left_action}, right:{right_action}"
        sub_task = canonical_sub_task(left_action, right_action)
        num_chunks = int(source_row["target"]["num_chunks"])
        target = {
            "guide_action": copy.deepcopy(source_row["target"]["guide_action"]),
            "primary_action_verb": verb,
            "sub_task": sub_task,
            "num_chunks": num_chunks,
        }

        segment = copy.deepcopy(source_segment)
        segment.update(
            {
                "left_action": left_action,
                "right_action": right_action,
                "primary_action_verb": verb,
                "sub_task": sub_task,
            }
        )
        output_segments.append(segment)

        row = copy.deepcopy(source_row)
        previous_state = (
            None
            if previous_row is None
            else [float(item) for item in previous_row["current_state"]]
        )
        previous_goal = (
            None if previous_row is None else copy.deepcopy(previous_row["target"])
        )
        previous_frame = (
            None if previous_row is None else int(previous_row["segment_start_frame"])
        )
        current_state = [float(item) for item in row["current_state"]]
        if previous_row is None:
            images = _current_images(row)
            source_action_rows = 0
            executed_chunks = 0
            executed_actions = 0
            padded_action_rows = 0
            observed_delta = None
        else:
            images = [*_current_images(previous_row), *_current_images(row)]
            source_action_rows = int(row["segment_start_frame"]) - previous_frame
            executed_chunks = int(math.ceil(source_action_rows / CHUNK_SIZE))
            executed_actions = executed_chunks * CHUNK_SIZE
            padded_action_rows = executed_actions - source_action_rows
            observed_delta = _delta(previous_state, current_state, rounded=False)
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
            thinking=True,
        )
        row.update(
            {
                "images": images,
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
                "action_padding_mode": (
                    "masked_zero_slots_no_state_transition"
                    if previous_row is not None
                    else None
                ),
                "action_padding_value": 0.0 if previous_row is not None else None,
                "target": target,
                "cot_annotation_required": True,
            }
        )
        row["messages"] = [
            {
                "role": "system",
                "content": system_prompt_for_mode(str(row["task_arm_mode"])),
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
        output_rows.append(row)
        previous_row = row

        for correction in action_corrections:
            corrections.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "segment_index": index,
                    "frame_index": int(row["frame_index"]),
                    "task": str(row["task"]),
                    **correction,
                }
            )

    output_episode["segments"] = output_segments
    return output_episode, output_rows, corrections


def rewrite_semantic_dataset(
    *,
    input_vlm: Path,
    input_vla: Path,
    output_vlm: Path,
    output_vla: Path,
    report_path: Path,
    corrections_path: Path,
    confirmed_corrections_path: Path | None = None,
) -> dict[str, Any]:
    """Rewrite all action language and preserve numerical supervision."""

    for output in (output_vlm, output_vla, report_path, corrections_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_vlm = output_vlm.with_name(f".{output_vlm.name}.tmp-{os.getpid()}")
    temporary_vla = output_vla.with_name(f".{output_vla.name}.tmp-{os.getpid()}")
    temporary_corrections = corrections_path.with_name(
        f".{corrections_path.name}.tmp-{os.getpid()}"
    )
    vlm_iterator = _iter_vlm(input_vlm)
    pending_vlm = next(vlm_iterator, None)
    counts: Counter = Counter()
    confirmed_corrections: dict[tuple[int, int, str], str] = {}
    if confirmed_corrections_path is not None:
        for line_number, line in enumerate(
            confirmed_corrections_path.open(encoding="utf-8"), 1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                int(item["episode_index"]),
                int(item["segment_index"]),
                str(item["arm"]),
            )
            proposed = str(item["proposed_action"])
            if key in confirmed_corrections:
                raise ValueError(
                    f"duplicate confirmed correction at line {line_number}: {key}"
                )
            confirmed_corrections[key] = proposed
    try:
        with (
            temporary_vlm.open("w", encoding="utf-8") as vlm_output,
            temporary_vla.open("w", encoding="utf-8") as vla_output,
            temporary_corrections.open("w", encoding="utf-8") as correction_output,
        ):
            vla_output.write("{\n")
            first_episode = True
            for training_key, episode in _iter_vla(input_vla):
                episode_index = int(episode["global_episode_index"])
                episode_rows, pending_vlm = _episode_vlm_rows(
                    vlm_iterator,
                    pending_vlm,
                    episode_index,
                )
                refined_episode, refined_rows, corrections = _rewrite_episode(
                    episode,
                    episode_rows,
                    confirmed_corrections,
                )
                for row in refined_rows:
                    vlm_output.write(_compact_json(row) + "\n")
                for correction in corrections:
                    correction_output.write(_compact_json(correction) + "\n")
                if not first_episode:
                    vla_output.write(",\n")
                first_episode = False
                vla_output.write(
                    json.dumps(str(training_key)) + ":" + _compact_json(refined_episode)
                )
                counts["episodes"] += 1
                counts["segments"] += len(refined_rows)
                counts["action_label_corrections"] += len(corrections)
                if counts["episodes"] % 1000 == 0:
                    print(
                        "[semantic-cleanup] "
                        f"episodes={counts['episodes']} "
                        f"segments={counts['segments']}",
                        flush=True,
                    )
            vla_output.write("\n}\n")
        if pending_vlm is not None or next(vlm_iterator, None) is not None:
            raise ValueError("VLM contains rows after the final VLA episode")
        os.replace(temporary_vlm, output_vlm)
        os.replace(temporary_vla, output_vla)
        os.replace(temporary_corrections, corrections_path)
    except BaseException:
        temporary_vlm.unlink(missing_ok=True)
        temporary_vla.unlink(missing_ok=True)
        temporary_corrections.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "robotwin_action_preparation_v1",
        "policy": {
            "sub_task": "deterministic_actuator_grounded_language",
            "cot": "requires_visual_observer_writer_and_dual_critics",
            "assistant": "action_json_pending_cot_annotation",
            "action_label_correction": (
                "only hard guide contradictions: large motion under still, "
                "opposite lift/lower direction, or opposite gripper sign"
            ),
        },
        "counts": dict(counts),
        "inputs": {
            "vlm": str(input_vlm),
            "vla": str(input_vla),
            "vlm_sha256": _sha256(input_vlm),
            "vla_sha256": _sha256(input_vla),
            "confirmed_corrections": (
                None
                if confirmed_corrections_path is None
                else str(confirmed_corrections_path)
            ),
        },
        "outputs": {
            "vlm": str(output_vlm),
            "vla": str(output_vla),
            "corrections": str(corrections_path),
            "vlm_sha256": _sha256(output_vlm),
            "vla_sha256": _sha256(output_vla),
            "corrections_sha256": _sha256(corrections_path),
        },
    }
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report
