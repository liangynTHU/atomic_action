"""RoboTwin-style student messages with explicit native state representations."""

from __future__ import annotations

import json
import math

import numpy as np

from atomic_seg.common.numeric import round_vector
from atomic_seg.prompt_contract import NO_THINKING_USER_ENDING
from atomic_seg.pretrain.segmentation import target_for


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def named_state(profile, values, *, delta=False):
    v = round_vector(values)
    if profile == "oxe_bridge" and len(v) == 7:
        names = ("delta_xyz", "delta_rpy", "delta_gripper") if delta else ("xyz", "rpy", "gripper")
        return {"arm": {names[0]: v[:3], names[1]: v[3:6], names[2]: v[6]}}
    if profile == "abc_yam" and len(v) == 14:
        joints, gripper = ("delta_joint_positions", "delta_gripper") if delta else ("joint_positions", "gripper")
        return {side: {joints: v[offset:offset + 6], gripper: v[offset + 6]}
                for side, offset in (("left", 0), ("right", 7))}
    if profile == "realomni_umi" and len(v) == 16:
        names = ("delta_xyz", "delta_quat_wxyz", "delta_gripper") if delta else ("xyz", "quat_wxyz", "gripper")
        return {side: {names[0]: v[offset:offset + 3], names[1]: v[offset + 3:offset + 7], names[2]: v[offset + 7]}
                for side, offset in (("left", 0), ("right", 8))}
    raise ValueError(f"state dimensions do not match {profile}")


def cameras_in_order(episode):
    names = list(episode["camera_paths"])
    preferred = {"image": 0, "top": 0, "top_left": 0, "top_right": 1,
                 "left_wrist": 2, "right_wrist": 3}
    if any(name not in preferred for name in names):
        raise ValueError("unknown camera role; add an explicit student camera mapping")
    return sorted(names, key=lambda name: (preferred[name], name))


def build_user(episode, segment, previous):
    profile, start = episode["profile"], segment["start_frame"]
    cameras = cameras_in_order(episode)
    frames = ([] if previous is None else [("previous-boundary", previous["start_frame"])]) + [("current", start)]
    roles = [f"{when} {camera.replace('_', ' ')}" for when, _ in frames for camera in cameras]
    lines = ["<image>" * len(roles) + "Task: " + episode["task"]]
    if profile in {"abc_yam", "realomni_umi"}:
        lines.append("Task arm mode: multi-arm; both arms may participate or hand off as required by the task.")
    lines.append("Images: " + ", ".join(roles) + ".")
    execution = None
    observed_delta = None
    if previous is None:
        lines.append("History boundary: none; this is the initial planning boundary.")
    else:
        previous_start = previous["start_frame"]
        if previous["end_frame"] + 1 != start:
            raise ValueError("history must use the immediately preceding complete segment")
        valid_rows = start - previous_start
        chunks = math.ceil(valid_rows / 8)
        execution = {"executed_chunks": chunks, "valid_action_rows": valid_rows,
                     "aligned_action_slots": chunks * 8, "padded_action_slots": chunks * 8 - valid_rows}
        lines += ["Previous command: " + json_text(target_for(episode, previous)),
                  "Previous execution: " + json_text(execution),
                  "Previous boundary state: " + json_text(named_state(profile, episode["states"][previous_start]))]
        observed_delta = named_state(profile, np.asarray(episode["states"][start]) - episode["states"][previous_start], delta=True)
    lines.append("Current state: " + json_text(named_state(profile, episode["states"][start])))
    if observed_delta is not None:
        lines.append("Observed state delta since previous boundary: " + json_text(observed_delta))
    lines.append(NO_THINKING_USER_ENDING)
    return "\n".join(lines), roles, frames, execution
