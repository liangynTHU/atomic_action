"""Mandatory native recovery action policy, independent of CoT generation."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from atomic_seg.still_policy import is_both_still, plan_still_segment_groups

QUALITY_POLICY = "recovery_cot_quality_20260930"
MINIMUM_ACTION_ROWS = 8


def _length(step: Mapping[str, Any]) -> int:
    return int(step["end_frame"]) - int(step["start_frame"]) + 1


def _pair(step: Mapping[str, Any]) -> tuple[str, str]:
    return str(step["left_action"]), str(step["right_action"])


def normalize_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep all action rows; preserve removed decision boundaries as provenance.

    Native decision IDs are provenance, not an exemption from the planner's
    no-nonterminal-still / no-short-identical-action policy. Different primitive
    transitions remain explicit, even when brief (especially gripper events).
    """
    values = copy.deepcopy(list(segments))
    for index, step in enumerate(values):
        step.setdefault(
            "source_atomic_spans",
            [
                {
                    "source_segment_index": index,
                    **{
                        key: step[key]
                        for key in (
                            "start_frame",
                            "end_frame",
                            "left_action",
                            "right_action",
                            "native_group_index",
                            "native_semantic_action",
                            "native_event_stage",
                        )
                        if key in step
                    },
                }
            ],
        )
    result = []
    for group in plan_still_segment_groups(
        values, minimum_terminal_still_frames=MINIMUM_ACTION_ROWS
    ):
        step = copy.deepcopy(values[group.owner_index])
        step["start_frame"] = group.start_frame
        step["end_frame"] = group.end_frame
        step["source_atomic_spans"] = [
            span
            for index in group.source_indices
            for span in values[index]["source_atomic_spans"]
        ]
        result.append(step)
    merged: list[dict[str, Any]] = []
    for step in result:
        merged.append(step)
        while (
            len(merged) > 1
            and _pair(merged[-2]) == _pair(merged[-1])
            and (
                min(_length(merged[-2]), _length(merged[-1])) < MINIMUM_ACTION_ROWS
                or merged[-2].get("native_group_index")
                == merged[-1].get("native_group_index")
            )
        ):
            right = merged.pop()
            merged[-1]["end_frame"] = right["end_frame"]
            merged[-1]["source_atomic_spans"].extend(right["source_atomic_spans"])
    for index, step in enumerate(merged):
        step["segment_index"] = index
        step["primary_action_verb"] = (
            f"left:{step['left_action']}, right:{step['right_action']}"
        )
        step["short_segment_reason"] = (
            "distinct_primitive_transition"
            if _length(step) < MINIMUM_ACTION_ROWS
            else None
        )
        step["quality_policy"] = QUALITY_POLICY
    validate_segments(merged)
    return merged


def validate_segments(segments: Sequence[Mapping[str, Any]]) -> None:
    if not segments or int(segments[0]["start_frame"]) != 0:
        raise ValueError("recovery actions must cover the episode from row zero")
    previous = None
    for index, step in enumerate(segments):
        length = _length(step)
        if length < 1:
            raise ValueError("empty recovery action")
        if (
            step.get("primary_action_verb")
            != f"left:{step['left_action']}, right:{step['right_action']}"
        ):
            raise ValueError("recovery action pair differs from per-arm labels")
        if is_both_still(step):
            if index != len(segments) - 1:
                raise ValueError("nonterminal both-still recovery action")
            if length < MINIMUM_ACTION_ROWS:
                # An all-still short episode has no meaningful action owner.
                # Fail explicitly, never invent an active label or publish it.
                raise ValueError(
                    "short terminal/all-still recovery episode requires review"
                )
        if previous is not None:
            if int(step["start_frame"]) != int(previous["end_frame"]) + 1:
                raise ValueError("recovery action timeline is not contiguous")
            if (
                _pair(step) == _pair(previous)
                and min(length, _length(previous)) < MINIMUM_ACTION_ROWS
            ):
                raise ValueError("short adjacent identical-action recovery boundary")
        previous = step
