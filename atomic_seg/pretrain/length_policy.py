"""Merge adjacent primitives only when both arms keep the same action labels."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np


MERGE_POLICY = "identical_bimanual_actions_v1"


@dataclass(frozen=True)
class LengthPolicy:
    minimum_frames: int | None = None
    preferred_minimum_frames: int = 32
    preferred_maximum_frames: int = 64
    maximum_merged_frames: int = 128

    def __post_init__(self):
        values = [self.preferred_minimum_frames, self.preferred_maximum_frames, self.maximum_merged_frames]
        if any(type(v) is not int or v < 1 for v in values) or values != sorted(values):
            raise ValueError("length thresholds must be positive ordered integers")
        if self.minimum_frames is not None and (
            type(self.minimum_frames) is not int or not 1 <= self.minimum_frames <= self.preferred_minimum_frames
        ):
            raise ValueError("minimum_frames must be null or a positive ordered threshold")


def length_statistics(segments):
    lengths = [s["end_frame"] - s["start_frame"] + 1 for s in segments]
    if not lengths:
        raise ValueError("no segments to summarize")
    return {
        "segments": len(lengths), "action_rows": sum(lengths),
        "minimum_frames": min(lengths), "maximum_frames": max(lengths),
        "median_frames": float(np.median(lengths)),
        "below_16": sum(n < 16 for n in lengths), "from_16_to_31": sum(16 <= n < 32 for n in lengths),
        "at_least_32": sum(n >= 32 for n in lengths),
        "num_chunks_distribution": dict(sorted(Counter(math.ceil(n / 8) for n in lengths).items())),
    }


def _action_pair(segment):
    return segment["left_action"], segment["right_action"]


def validate_action_primitives(segment):
    """Reject mixed-action groups, allowing explicitly recorded idle context."""
    expected = _action_pair(segment)
    for primitive in segment.get("primitive_segments", [segment]):
        actual = _action_pair(primitive)
        if actual == expected:
            continue
        if actual == ("still", "still") and any(
            span["start_frame"] <= primitive["start_frame"]
            and primitive["end_frame"] <= span["end_frame"]
            for span in segment.get("absorbed_still_spans", [])
        ):
            continue
        raise ValueError(
            "mixed-action primitive group; regenerate segmentation and annotations "
            "from the original primitives under the identical-action merge policy"
        )


def _opposed_grippers(primitives):
    return any(
        {"open_gripper", "close_gripper"} <= {
            p[arm + "_action"] for s in primitives for p in s.get("primitive_segments", [s])
        }
        for arm in ("left", "right")
    )


def coarsen_segments(episode, segments, policy: LengthPolicy):
    """Only remove boundaries; never duplicate, discard, pad, or reorder action rows.

    By default there is no hard length floor. The preference penalizes fragments,
    subordinate to every change in either arm's action and source transitions.
    Previously absorbed both-still context stays attached to its owning action.
    An explicit minimum_frames cannot override these protected boundaries.
    """
    if episode["profile"] not in {"abc_yam", "realomni_umi", "agibot_a2d"}:
        raise ValueError("this length policy requires a supported bimanual profile")
    if not segments:
        raise ValueError("cannot group an empty episode")
    cursor = 0
    for segment in segments:
        if segment["start_frame"] != cursor or segment["end_frame"] < cursor:
            raise ValueError("source primitives must cover an ordered complete episode")
        validate_action_primitives(segment)
        cursor = segment["end_frame"] + 1
    if cursor != len(episode["states"]):
        raise ValueError("source primitive coverage differs from native rows")
    count = len(segments)
    cuts = [s["start_frame"] for s in segments] + [cursor]
    protected = {}
    action_boundaries = {i for i in range(1, count)
                         if _action_pair(segments[i - 1]) != _action_pair(segments[i])}
    terminal_hold_boundaries = {i for i, s in enumerate(segments) if i > 0 and s.get("terminal_hold")}
    for annotation in episode.get("source_annotations", []):
        index = min(range(len(cuts)), key=lambda k: abs(cuts[k] - annotation["start_frame"]))
        if 0 < index < count:
            protected.setdefault(index, []).append({
                "source_frame": annotation["start_frame"], "retained_frame": cuts[index],
                "snap_offset_frames": cuts[index] - annotation["start_frame"], "text": annotation["text"],
            })
    protected_indices = set(protected) | terminal_hold_boundaries | action_boundaries
    costs = [math.inf] * (count + 1)
    previous = [None] * (count + 1)
    costs[0] = 0.0
    group_start_floor = 0
    for end in range(1, count + 1):
        if end - 1 in protected_indices:
            group_start_floor = end - 1
        for start in range(end - 1, group_start_floor - 1, -1):
            length = cuts[end] - cuts[start]
            if policy.minimum_frames is not None and length < policy.minimum_frames:
                continue
            if length > policy.maximum_merged_frames and end - start > 1:
                break
            shortfall = max(0, policy.preferred_minimum_frames - length) / (policy.preferred_minimum_frames / 2)
            excess = max(0, length - policy.preferred_maximum_frames) / policy.preferred_minimum_frames
            loss = 0.8 + 4 * shortfall**2 + excess**2
            if costs[start] + loss < costs[end]:
                costs[end], previous[end] = costs[start] + loss, start
    if not math.isfinite(costs[-1]):
        raise ValueError("minimum frame count conflicts with protected boundaries; episode requires review")
    groups, end = [], count
    while end:
        start = previous[end]
        groups.append((start, end))
        end = start
    groups.reverse()
    output = []
    for start, end in groups:
        primitives = segments[start:end]
        segment = {"segment_index": len(output), "start_frame": cuts[start], "end_frame": cuts[end] - 1}
        sequences = {}
        leaves = [p for primitive in primitives for p in primitive.get("primitive_segments", [primitive])]
        for arm in ("left", "right"):
            sequence = []
            for primitive in leaves:
                label = primitive[arm + "_action"]
                if not sequence or sequence[-1] != label:
                    sequence.append(label)
            segment[arm + "_action"] = primitives[0][arm + "_action"]
            sequences[arm] = sequence
        segment["primary_action_verb"] = f"left:{segment['left_action']}, right:{segment['right_action']}"
        segment["primitive_segments"] = [
            {key: p[key] for key in ("segment_index", "start_frame", "end_frame", "left_action", "right_action")}
            for p in leaves
        ]
        segment["absorbed_still_spans"] = [span for p in primitives for span in p.get("absorbed_still_spans", [])]
        segment["terminal_hold"] = end == count and all(p["left_action"] == p["right_action"] == "still" for p in leaves)
        segment["motion_sequence"] = sequences
        segment["boundary_context"] = {"start": protected.get(start, []), "end": protected.get(end, [])}
        segment["grouping_reason"] = "adjacent identical left/right actions; action and source transitions retained"
        if cuts[end] - cuts[start] < policy.preferred_minimum_frames:
            reasons = []
            for name, index, neighbor in (("previous", start, (start - 1, end)), ("next", end, (start, end + 1))):
                if index in (0, count):
                    reasons.append(name + ":episode_edge")
                elif index in protected:
                    reasons.append(name + ":source_annotation_boundary")
                elif index in terminal_hold_boundaries:
                    reasons.append(name + ":terminal_hold_boundary")
                elif _opposed_grippers(segments[neighbor[0]:neighbor[1]]):
                    reasons.append(name + ":opposing_gripper_actions")
                elif index in action_boundaries:
                    reasons.append(name + ":different_action_labels")
                else:
                    reasons.append(name + ":preserved_motion_boundary_in_global_length_fit")
            segment["short_segment_reason"] = "; ".join(reasons)
        validate_action_primitives(segment)
        output.append(segment)
    audit = {"policy": asdict(policy), "before": length_statistics(segments), "after": length_statistics(output),
             "length_mode": "soft_preference" if policy.minimum_frames is None else "explicit_hard_minimum",
             "hard_minimum_enforced": policy.minimum_frames is not None,
             "terminal_hold_boundaries": [cuts[i] for i in sorted(terminal_hold_boundaries)],
             "merge_policy": MERGE_POLICY,
             "action_boundaries": [cuts[i] for i in sorted(action_boundaries)],
             "identical_action_labels_in_group": True,
             "protected_boundaries": [item for items in protected.values() for item in items],
             "cost": costs[-1], "source_annotation_policy": "retain nearest existing motion boundary; offsets recorded",
             "no_opposing_gripper_labels_in_group": True}
    return output, audit
